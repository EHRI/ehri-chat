from flask import Flask, request, Response, jsonify, render_template
from ehri_graph_rag.models.model_manager import Ministral3B, MistralSmallLatestAPI

app = Flask(__name__, template_folder="../conf/web/templates")

@app.route("/chat/completions", methods=["POST"])
def generate_response():
    if not request.is_json:
        return Response("Invalid input: JSON required", status=400, mimetype="text/plain")

    data = request.get_json()
    messages = data.get("messages", "")
    last_message = messages[-1].get("content", "")
    model_input = data.get("model", "mistral")
    mode_input = data.get("mode", "graphrag")

    def generate():
        if model_input == "mistral":
            model = MistralSmallLatestAPI()
            if mode_input == "rag":
                streamer = model.get_results_llm_with_rag(last_message)
            elif mode_input == "graphrag":
                streamer = model.get_results_llm_with_graphrag(last_message)
            else:
                streamer = model.get_results_llm(last_message)
            # answer_bits = [chunk.data.choices[0].delta.content for chunk in streamer]
            # answer = "".join(answer_bits)
            for chunk in streamer:
                yield chunk.data.choices[0].delta.content
        return Response("Invalid combination of options", status=400, mimetype="text/plain")

    return generate()

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

@app.route('/')
def home():
    return render_template("web.html")