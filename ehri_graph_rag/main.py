from ehri_graph_rag.models.model_manager import Ministral3B, MistralSmallLatestAPI
import sys

class ModelLoader():
    def __init__(self):
        self.model = None
    
    def get_model(self):
        if(self.model is None):
            self.model = MistralSmallLatestAPI()
        return self.model

if __name__ == "__main__":
    model_loader = ModelLoader()
    if(len(sys.argv) >=2  and sys.argv[1] == "RAG"):
        if(len(sys.argv) >= 3):
            prompt = sys.argv[2]
            print("Answering query with RAG:")
            #streamer, thread = model_loader.get_model().get_results_llm_with_rag(prompt)
            streamer = model_loader.get_model().get_results_llm_with_rag(prompt)
            #model_loader.get_model().print_streamer(streamer, thread)
            model_loader.get_model().print_streamer(streamer)
        else:
            print("Please provide a prompt as an argument.")
    elif(len(sys.argv) >=2  and sys.argv[1] == "GraphRAG"):
        if(len(sys.argv) >= 3):
            prompt = sys.argv[2]
            print("Answering query with GraphRAG:")
            #streamer, thread = model_loader.get_model().get_results_llm_with_rag(prompt)
            streamer = model_loader.get_model().get_results_llm_with_graphrag(prompt)
            #model_loader.get_model().print_streamer(streamer, thread)
            model_loader.get_model().print_streamer(streamer)
        else:
            print("Please provide a prompt as an argument.")
    else:
        if(len(sys.argv) >= 2):
            prompt = sys.argv[1]
            print("Answering query without RAG:")
            #streamer, thread = model_loader.get_model().get_results_llm(prompt)
            streamer = model_loader.get_model().get_results_llm(prompt)
            #model_loader.get_model().print_streamer(streamer, thread)
            model_loader.get_model().print_streamer(streamer)
        else:
            print("Please provide a prompt as an argument.")

