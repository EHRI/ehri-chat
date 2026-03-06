from ehri_graph_rag.embeddings.embeddings_manager import RagEmbeddingsManager, GraphRagEmbeddingsManager
from ehri_graph_rag.rag.graphrag_context_manager import GraphRagContextManager
from mistralai import Mistral
from datetime import datetime
import http.client
import json
import os
import logging
logger = logging.getLogger("ehri_graph_rag")

class LLModel:
    def __init__(self):
        self.rag_embeddings_manager = RagEmbeddingsManager()
        self.graphrag_context_manager = GraphRagContextManager()
        self.today = datetime.today()
        self.yesterday = datetime(self.today.year, self.today.month, self.today.day - 1)
        self.system_prompt = f"""You are a Large Language Model (LLM).
The current date is {self.today.strftime('%Y-%m-%d')}.
You are now being used in a Retrieval Augmented Generation (RAG) set up using data from the EHRI Portal which will feed some contextual information.
Whenever possible try to put the links to the provided context so users can easily expand their searches.
If this contextual information does not provide good answers just follow the general behaviour defined below.

When you're not sure about some information, you say that you don't have the information and don't make up anything.
If the user's question is not clear, ambiguous, or does not provide enough context for you to accurately answer the question, you do not try to answer it right away and you rather ask the user to clarify their request (e.g. "What are some good restaurants around me?" => "Where are you?" or "When is the next flight to Tokyo" => "Where do you travel from?").
You are always very attentive to dates, in particular you try to resolve dates (e.g. "yesterday" is {self.yesterday.strftime('%Y-%m-%d')}) and when asked about information at specific dates, you discard information that is at another date.
You follow these instructions in all languages, and always respond to the user in the language they use or request.
Next sections describe the capabilities that you have."""

    def generate_messages(self, user_prompt):
        return [{
                    "role": "system",
                    "content": self.system_prompt
                },
                {
                    "content": user_prompt,
                    "role": "user"
                }]

    def generate_rag_prompt(self, user_prompt, relevant_context):
        return f"""
Context information is below.
---------------------
{relevant_context}
---------------------
Given the context information and not prior knowledge, answer the query.
Query: {user_prompt}
Answer:
"""

    def get_results_llm_with_rag(self, user_prompt):
        relevant_context = "\n\n".join(self.rag_embeddings_manager.retrieve_relevant_chunks(user_prompt))
        prompt_with_context = self.generate_rag_prompt(user_prompt, relevant_context)
        return self.get_results_llm(prompt_with_context)

    def get_results_llm_with_graphrag(self, user_prompt):
        relevant_context = "\n\n".join(self.graphrag_context_manager.retrieve_relevant_context(user_prompt))
        prompt_with_context = self.generate_rag_prompt(user_prompt, relevant_context)
        return self.get_results_llm(prompt_with_context)

    
class MistralAPI(LLModel):
    def __init__(self):
        super().__init__()
        self.client = Mistral(api_key=os.getenv("MISTRAL_API_KEY", ""))

    def get_results_llm(self, user_prompt):
        logger.debug(f"Generated prompt: {user_prompt}")
        response = self.client.chat.stream(model="mistral-small-latest", messages=self.generate_messages(user_prompt))
        def generate():
            for chunk in response:
                yield chunk.data.choices[0].delta.content
        return generate()

class LlamaCpp(LLModel):
    def __init__(self):
        super().__init__()
        self.endpoint = "localhost:8080" 
        self.path = "/v1/chat/completions"

    def get_results_llm(self, user_prompt):
        logger.debug(f"Generated prompt: {user_prompt}")
        conn = http.client.HTTPConnection(self.endpoint)
        headers = {'Content-type': 'application/json'}
        json_data = json.dumps({
            "messages": self.generate_messages(user_prompt),
            "stream": True
        })
        def generate():
            conn.request('POST', self.path, json_data, headers)
            for event in conn.getresponse():
                if event.startswith(b"data:") and b"data: [DONE]" not in event:
                    chunk = json.loads(event.decode().rstrip("\n").replace("data:", "", 1))
                    if chunk['choices'][0]['delta'] is not None:
                        message_part = chunk['choices'][0]['delta'].get('content', "")
                        yield message_part if message_part is not None else ""
        return generate()