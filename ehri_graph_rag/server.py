from flask import Flask, request, Response, jsonify, render_template
from ehri_graph_rag.models.model_manager import LlamaCpp, MistralAPI
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
    model_input = data.get("model", "qwen3-vl-2b-instruct")
    mode_input = data.get("mode", "graphrag")

    logger.info(f"Received message \"{last_message}\" to be resolved using mode {mode_input} and model {model_input}")

    match model_input.lower():
        case "ministral-3b-2512":
            model = mistral_api
        case "qwen3-vl-2b-instruct":
            model = llama_cpp
        case _:
            model = llama_cpp

    def generate():
        match mode_input:
            case "rag":
                streamer = model.get_results_llm_with_rag(last_message)
            case "graphrag":
                streamer = model.get_results_llm_with_graphrag(last_message)
            case _:
                streamer = model.get_results_llm(last_message)
        for chunk in streamer:
            yield chunk
        return Response("Invalid combination of options", status=400, mimetype="text/plain")

    return generate()

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

@app.route('/')
def home():
    return render_template("web.html")