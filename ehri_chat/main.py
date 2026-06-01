from ehri_chat.models.model_manager import LlamaCpp, MistralAPI, GeminiAPI, LLMGenerationOptions
import asyncio
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
            case "gemini":
                self.model = GeminiAPI()
            case "qwen":
                self.model = LlamaCpp()
            case _:
                self.model = LlamaCpp()

@click.command()
@click.option('--prompt',
              help="Prompt for the model",
              required=True)
@click.option('--model',
              type=click.Choice(['qwen', 'mistral', 'gemini']),
              required = False,
              help="Model to use, right now mistral for Ministral-3B-2512 or qwen for Qwen3-VL-2B-Instruct",
              default="qwen",
              show_default=True)
@click.option("--mode",
              type=click.Choice(['GraphRAG', 'RAG', 'MCP', 'vanilla']),
              required = False,
              help="The enhancement mode to use",
              default="vanilla",
              show_default=True)
@click.option("--temperature",
              help="Temperature used for the generation",
              type=click.FloatRange(0.0, 1.0),
              required=False,
              default=0.1,
              show_default=True)
@click.option("--top_k",
              help="Top_k used for the generation",
              type=click.IntRange(1, 100),
              required=False,
              default=20,
              show_default=True)
@click.option("--max_tokens",
              help="Max tokens to generate",
              type=click.IntRange(1024, 10240),
              required=False,
              default=2048,
              show_default=True)
def ehri_chat(prompt, model, mode, temperature, top_k, max_tokens):
    generation_options = LLMGenerationOptions(
        model=None,# for now the model is not passed as we make use of a simplistic referencing method in the CLI
        temperature=temperature,
        top_k=top_k,
        max_tokens=max_tokens
    )
    asyncio.run(run(prompt, model, mode, generation_options))

async def run(prompt, model, mode, generation_options: LLMGenerationOptions = LLMGenerationOptions()):
    model = ModelLoader(model).model
    match mode:
        case "GraphRAG":
            streamer = await model.get_results_llm_with_graphrag(prompt, generation_options=generation_options)
        case "RAG":
            streamer = await model.get_results_llm_with_rag(prompt, generation_options=generation_options)
        case "MCP":
            streamer = await model.get_results_llm_with_mcp(prompt, generation_options=generation_options)
        case _:
            streamer = await model.get_results_llm(prompt, generation_options=generation_options)
    async for chunk in streamer:
        print(chunk, end="", flush=True)

if __name__ == "__main__":
    logging.getLogger("ehri_chat").setLevel(logging.WARN)
    ehri_chat()