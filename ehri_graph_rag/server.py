from typing import Generator, Any
from flask import Flask, request, Response, jsonify, render_template
from ehri_graph_rag.models.model_manager import LlamaCpp, MistralAPI
import asyncio
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

app = Flask(__name__, template_folder="../conf/web/templates")
mistral_api = MistralAPI()
llama_cpp = LlamaCpp()
logging.getLogger("ehri_graph_rag").setLevel(logging.WARN)
logging.basicConfig(level=logging.WARN)

@app.route("/chat/completions", methods=["POST"])
def generate_response():
    if not request.is_json:
        return Response("Invalid input: JSON required", status=400, mimetype="text/plain")

    data = request.get_json()
    messages = data.get("messages", "")
    last_message = messages[-1].get("content", "")
    model_input = data.get("model", "qwen3-vl:2b-instruct-q4_K_M")
    mode_input = data.get("mode", "graphrag")

    logger.info(f"Received message \"{last_message}\" to be resolved using mode {mode_input} and model {model_input}")

    match model_input.lower():
        case "ministral-3b-2512":
            model = mistral_api
        case "qwen3-vl:2b-instruct-q4_K_M":
            model = llama_cpp
        case _:
            model = llama_cpp

    async def generate():
        match mode_input:
            case "rag":
                streamer = await model.get_results_llm_with_rag(last_message, model = model_input.lower())
            case "graphrag":
                streamer = await model.get_results_llm_with_graphrag(last_message, model = model_input.lower())
            case "mcp":
                streamer = await model.get_results_llm_with_mcp(last_message, model = model_input.lower())
            case _:
                streamer = await model.get_results_llm(last_message, model = model_input.lower())
        async for chunk in streamer:
            yield chunk
    return __iter_over_async(generate())

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

@app.route('/')
def home():
    return render_template("web.html")

# extracted from: https://medium.com/@mr.murga/streaming-ai-responses-with-flask-a-practical-guide-677c15e82cdd
def __iter_over_async(async_generator) -> Generator[str, None, None]:
    """
    Iterates over the async iterable and yields formatted chunks.

    Yields:
        str: Formatted chunk of response text.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    iterator = async_generator.__aiter__()

    async def get_next() -> tuple[bool, Any]:
        """
        Retrieves the next chunk from the iterator.

        Returns:
            tuple[bool, Any]: A tuple with a boolean indicating if the iteration is done and the chunk.
        """
        try:
            obj = await iterator.__anext__()
            return False, obj
        except StopAsyncIteration:
            return True, None
        except Exception as e:
            print(f"Error in get_next: {e}")
            # Handle exceptions from callable_fn or chunk_fnc
            return True, None

    try:
        while True:
            done, obj = loop.run_until_complete(get_next())
            if done:
                break
            yield obj
    finally:
        loop.close()