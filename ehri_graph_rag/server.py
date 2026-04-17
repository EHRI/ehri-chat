from typing import Generator, Any
from flask import Flask, request, Response, jsonify, render_template
from ehri_graph_rag.database.database_manager import DatabaseManager, ActivityRecord
from ehri_graph_rag.models.model_manager import LlamaCpp, MistralAPI, GeminiAPI, LLMGenerationOptions
import asyncio
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

app = Flask(__name__, template_folder="../conf/web/templates")
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
    top_k = data.get("top_k", LLMGenerationOptions().top_k)
    temperature = data.get("temperature", LLMGenerationOptions().temperature)
    max_tokens = data.get("max_tokens", LLMGenerationOptions().max_tokens)
    llm_generation_options = LLMGenerationOptions(
        model=model_input.lower(),
        temperature=temperature,
        top_k=top_k,
        max_tokens=max_tokens
    )

    logger.info(f"Received message \"{last_message}\" to be resolved using mode {mode_input} and model {model_input}")

    match model_input.lower():
        case "mistral-small-latest":
            model = MistralAPI()
        case "gemini-2.5-flash":
            model = GeminiAPI()
        case "qwen3-vl:2b-instruct-q4_K_M":
            model = LlamaCpp()
        case _:
            model = LlamaCpp()

    async def generate():
        output = ""
        error = None
        try:
            match mode_input:
                case "rag":
                    streamer = await model.get_results_llm_with_rag(last_message, generation_options = llm_generation_options)
                case "graphrag":
                    streamer = await model.get_results_llm_with_graphrag(last_message, generation_options = llm_generation_options)
                case "mcp":
                    streamer = await model.get_results_llm_with_mcp(last_message, generation_options = llm_generation_options)
                case _:
                    streamer = await model.get_results_llm(last_message, generation_options = llm_generation_options)

            async for chunk in streamer:
                output += chunk
                yield chunk
        except Exception as e:
            error = str(e)
        finally:
            DatabaseManager().insert_activity(ActivityRecord(last_message, mode_input, model_input, temperature, top_k, output, error))

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