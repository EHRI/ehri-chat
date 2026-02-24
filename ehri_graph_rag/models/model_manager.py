from transformers import Mistral3ForConditionalGeneration, MistralCommonBackend
from transformers import TextIteratorStreamer
from threading import Thread
from ehri_graph_rag.embeddings.embeddings_manager import RagEmbeddingsManager
from ehri_graph_rag.rag.graphrag_context_manager import GraphRagContextManager
from mistralai import Mistral
import os

class LLMModel():
    pass

class Ministral3B(LLMModel):
    def __init__(self):
        super().__init__()
        self.model_id = "mistralai/Ministral-3-3B-Instruct-2512-BF16"


    def get_results_llm(self, user_prompt):
        tokenizer = MistralCommonBackend.from_pretrained(self.model_id)
        model = Mistral3ForConditionalGeneration.from_pretrained(self.model_id, device_map="auto")
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_prompt,
                    }
                ],
            },
        ]

        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        tokenized = tokenizer.apply_chat_template(messages, return_tensors="pt", return_dict=True)

        generation_kwargs = dict(tokenized, streamer=streamer, max_new_tokens=1024)

        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()
    
        return streamer, thread

    def get_results_llm_with_rag(self, user_prompt):
        relevant_context = "\n\n".join(RagEmbeddingsManager().retrieve_relevant_chunks(user_prompt))
        prompt_with_context = user_prompt + f"""\n\n
        # CONTEXT
        {relevant_context}
        """
        return self.get_results_llm(prompt_with_context)
    
    def print_streamer(self, streamer, thread):
        for new_text in streamer:
            print(new_text, end="", flush=True)
        thread.join()
    
class MistralSmallLatestAPI(LLMModel):
    def __init__(self):
        super().__init__()
        self.client = Mistral(api_key=os.getenv("MISTRAL_API_KEY", ""))
        self.system_prompt = """"You are Mistral Small 3.1, a Large Language Model (LLM) created by Mistral AI, a French startup headquartered in Paris.
You power an AI assistant called Le Chat.
Your knowledge base was last updated on 2023-10-01.
The current date is {today}.
You are now being used in a Retrieval Augmented Generation (RAG) set up using data from the EHRI Portal which will feed some contextual information.
Whenever possible try to put the links to the provided context so users can easily expand their searches.
If this contextual information does not provide good answers just follow the general behaviour defined below.

When you're not sure about some information, you say that you don't have the information and don't make up anything.
If the user's question is not clear, ambiguous, or does not provide enough context for you to accurately answer the question, you do not try to answer it right away and you rather ask the user to clarify their request (e.g. "What are some good restaurants around me?" => "Where are you?" or "When is the next flight to Tokyo" => "Where do you travel from?").
You are always very attentive to dates, in particular you try to resolve dates (e.g. "yesterday" is {yesterday}) and when asked about information at specific dates, you discard information that is at another date.
You follow these instructions in all languages, and always respond to the user in the language they use or request.
Next sections describe the capabilities that you have."""

    def print_streamer(self, streamer):
        for chunk in streamer:
            print(chunk.data.choices[0].delta.content, end="", flush=True)

    def get_results_llm(self, user_prompt):
        return self.client.chat.stream(model="mistral-small-latest", 
            messages=[
                {
                    "role": "system",
                    "content": self.system_prompt
                },
                {
                    "content": user_prompt,
                    "role": "user"
                },
            ])

    def get_results_llm_with_rag(self, user_prompt):
        relevant_context = "\n\n".join(RagEmbeddingsManager().retrieve_relevant_chunks(user_prompt))
        prompt_with_context = f"""
Context information is below.
---------------------
{relevant_context}
---------------------
Given the context information and not prior knowledge, answer the query.
Query: {user_prompt}
Answer:
"""
        # prompt_with_context = user_prompt + f"""\n\n
        # # CONTEXT
        # {relevant_context}
        # """
        return self.get_results_llm(prompt_with_context)
    
    def get_results_llm_with_graphrag(self, user_prompt):
        relevant_context = "\n\n".join(GraphRagContextManager().retrieve_relevant_context(user_prompt))
        prompt_with_context = f"""
Context information is below.
---------------------
{relevant_context}
---------------------
Given the context information and not prior knowledge, answer the query.
Query: {user_prompt}
Answer:
"""
        # prompt_with_context = user_prompt + f"""\n\n
        # # CONTEXT
        # {relevant_context}
        # """
        return self.get_results_llm(prompt_with_context)