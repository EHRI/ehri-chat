from ehri_graph_rag.models.model_manager import LlamaCpp, MistralAPI
import click
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARN)
logging.basicConfig(level=logging.WARN)

class ModelLoader:
    def __init__(self, model):
        match model:
            case "mistral":
                self.model = MistralAPI()
            case "qwen":
                self.model = LlamaCpp()
            case _:
                self.model = LlamaCpp()

@click.command()
@click.option('--prompt',
              help="Prompt for the model",
              required=True)
@click.option('--model',
              type=click.Choice(['qwen', 'mistral']),
              required = False,
              help="Model to use, right now mistral for Ministral-3B-2512 or qwen for Qwen3-VL-2B-Instruct",
              default="qwen",
              show_default=True)
@click.option("--mode",
              type=click.Choice(['GraphRAG', 'RAG', 'vanilla']),
              required = False,
              help="The enhancement mode to use",
              default="vanilla",
              show_default=True)
def ehri_graph_rag(prompt, model, mode):
    model = ModelLoader(model).model
    match mode:
        case "GraphRAG":
            streamer = model.get_results_llm_with_graphrag(prompt)
        case "RAG":
            streamer = model.get_results_llm_with_rag(prompt)
        case _:
            streamer = model.get_results_llm(prompt)
    for chunk in streamer:
        print(chunk, end="", flush=True)

if __name__ == "__main__":
    logging.getLogger("ehri_graph_rag").setLevel(logging.WARN)
    ehri_graph_rag()

