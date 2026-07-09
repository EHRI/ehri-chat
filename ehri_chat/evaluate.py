import dataclasses
import time

from pydantic import BaseModel, Field
from enum import Enum
from mistralai.client import Mistral
import os
import asyncio
import datetime
import json

from ehri_chat.main import ModelLoader
from ehri_chat.models.model_manager import Evaluation, LLMGenerationOptions

# Based on https://docs.mistral.ai/resources/cookbooks/mistral-evaluation-rag_evaluation


# Define Enum for scores
class Score(str, Enum):
    no_relevance = "0"
    low_relevance = "1"
    medium_relevance = "2"
    high_relevance = "3"

# Define a constant for the score description
SCORE_DESCRIPTION = (
    "Score as a string between '0' and '3'. "
    "0: No relevance/Not grounded/Irrelevant - The context/answer is completely unrelated or not based on the context. "
    "1: Low relevance/Low groundedness/Somewhat relevant - The context/answer has minimal relevance or grounding. "
    "2: Medium relevance/Medium groundedness/Mostly relevant - The context/answer is somewhat relevant or grounded. "
    "3: High relevance/High groundedness/Fully relevant - The context/answer is highly relevant or grounded."
)

# Define separate classes for each criterion with detailed descriptions
class ContextRelevance(BaseModel):
    explanation: str = Field(..., description=("Step-by-step reasoning explaining how the retrieved context aligns with the user's query. "
                    "Consider the relevance of the information to the query's intent and the appropriateness of the context "
                    "in providing a coherent and useful response. In case the context is empty give the lowest score possible for this metric and disregard its relation to the other metrics."))
    score: Score = Field(..., description=SCORE_DESCRIPTION)

class AnswerRelevance(BaseModel):
    explanation: str = Field(..., description=("Step-by-step reasoning explaining how well the generated answer addresses the user's original query. "
                    "Consider the helpfulness and on-point nature of the answer, aligning with the user's intent and providing valuable insights."))
    score: Score = Field(..., description=SCORE_DESCRIPTION)

class Groundedness(BaseModel):
    explanation: str = Field(..., description=("Step-by-step reasoning explaining how faithful the generated answer is to the retrieved context. "
                    "Consider the factual accuracy and reliability of the answer, ensuring it is grounded in the retrieved information."))
    score: Score = Field(..., description=SCORE_DESCRIPTION)

class RAGEvaluation(BaseModel):
    context_relevance: ContextRelevance = Field(..., description="Evaluation of the context relevance to the query, considering how well the retrieved context aligns with the user's intent." )
    answer_relevance: AnswerRelevance = Field(..., description="Evaluation of the answer relevance to the query, assessing how well the generated answer addresses the user's original query." )
    groundedness: Groundedness = Field(..., description="Evaluation of the groundedness of the generated answer, ensuring it is faithful to the retrieved context." )

# Function to evaluate RAG metrics
def evaluate_rag(query: str, retrieved_context: str, generated_answer: str):
    # Initialize the Mistral client with the API key
    client = Mistral(api_key=os.getenv("MISTRAL_API_KEY", ""))
    model = "mistral-large-latest"
    chat_response = client.chat.parse(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a judge for evaluating a Retrieval-Augmented Generation (RAG) system. "
                    "Evaluate the context relevance, answer relevance, and groundedness based on the following criteria: "
                    "Provide a reasoning and a score as a string between '0' and '3' for each criterion. "
                    "Context Relevance: How relevant is the retrieved context to the query? "
                    "Answer Relevance: How relevant is the generated answer to the query? "
                    "Groundedness: How faithful is the generated answer to the retrieved context?"
                )
            },
            {
                "role": "user",
                "content": f"Query: {query}\nRetrieved Context: {retrieved_context}\nGenerated Answer: {generated_answer}"
            },
        ],
        response_format=RAGEvaluation,
        temperature=0
    )
    return chat_response.choices[0].message.parsed

async def generator_to_str(generator):
    output = ""
    async for chunk in generator:
        output += str(chunk)
    return output

async def evaluate(query, provider, model, method, evaluation_data):

    model_with_rag = ModelLoader(provider).model

    llm_generation_options = LLMGenerationOptions(
        model = model,
        temperature = 0.1,
        top_k = 20,
        max_tokens = 4096
    )
    response = ""
    match method:
        case "vanilla":
            response = await model_with_rag.get_results_llm(query, generation_options=llm_generation_options, evaluation=evaluation_data)
        case "rag":
            response = await model_with_rag.get_results_llm_with_rag(query, generation_options=llm_generation_options, evaluation=evaluation_data)
        case "graphrag":
            response = await model_with_rag.get_results_llm_with_graphrag(query, generation_options=llm_generation_options, evaluation=evaluation_data)
        case "mcp":
            prompt = query + " Use the EHRI-KG MCP server." # to force the use of MCP
            response = await model_with_rag.get_results_llm_with_mcp(prompt, generation_options=llm_generation_options, evaluation=evaluation_data)
    evaluation_data.generated_response = await generator_to_str(response)
    return evaluate_rag(query, evaluation_data.relevant_context, evaluation_data.generated_response)

if __name__ == "__main__":
    async def run_evaluation():
        providers = ["mistral", "gemini"]
        methods = ["vanilla", "rag", "graphrag", "mcp"]
        queries = {
            "situation_belgium": "What was the situation of Belgium during the Holocaust?",
            "roma_fate_france": "What was the fate of Roma in France during the Holocaust?",
            "comparison_countries": "Can you establish a relation between France and Italy in terms of collaboration with the Nazis?",
            "listing_institutions": "List the most relevant institutions for the Holocaust in Belgium.",
            "museums_holocaust": "Can you locate museums for the Holocaust in the Netherlands?",
            "camps_with_archive_germany": "Locate camp sites that hold an archive within their premises in Germany these days.",
            "deportations_antwerp": "I need to retrieve information about deportations of Jews in Antwerp.",
            "raids_amsterdam": "Can you locate archival material relating to raids in Amsterdam.",
            "visas_countries": "Do you have information on what countries issued visas for Jewish refugees during the War.",
            "person_information": "Who was Hermann Göring?"
        }

        print("Evaluating...")
        for method in methods:
            for key, query in queries.items():
                for provider in providers:
                    model = ""
                    match provider:
                        case "mistral":
                            model = "mistral-small-latest"
                        case "gemini":
                            model = "gemini-2.5-flash"
                    evaluation_data = Evaluation()
                    print(f"Query {key} against model {model} and method {method}...")
                    report = await evaluate(query, provider, model, method, evaluation_data)
                    time_str = str(datetime.datetime.now()).replace(' ', '_').replace(':', '.')
                    filename_llm_report = f"evaluations/llm_as_a_judge/query_{key}_{method}_{provider}_{time_str}_report.json"
                    filename_tokens_report = f"evaluations/tokens_usage/query_{key}_{method}_{provider}_{time_str}_tokens_usage.json"
                    with open(filename_llm_report, "w+") as f:
                        f.write(report.model_dump_json(indent=2))
                    with open(filename_tokens_report, "w+") as f:
                        f.write(json.dumps(dataclasses.asdict(evaluation_data)))
                    time.sleep(30)  # wait to avoid exceeding quotas

    asyncio.run(run_evaluation())