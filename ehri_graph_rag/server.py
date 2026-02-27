from flask import Flask, request, Response, jsonify, render_template
from ehri_graph_rag.models.model_manager import LlamaCpp, MistralAPI

app = Flask(__name__, template_folder="../conf/web/templates")

@app.route("/chat/completions", methods=["POST"])
def generate_response():
    if not request.is_json:
        return Response("Invalid input: JSON required", status=400, mimetype="text/plain")

    data = request.get_json()
    messages = data.get("messages", "")
    last_message = messages[-1].get("content", "")
    model_input = data.get("model", "qwen3-vl-2b-instruct")
    mode_input = data.get("mode", "graphrag")

    match model_input.lower():
        case "ministral-3b-2512":
            model = MistralAPI()
        case "qwen3-vl-2b-instruct":
            model = LlamaCpp()
        case _:
            model = LlamaCpp()

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