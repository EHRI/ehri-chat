import uuid
from dataclasses import dataclass
from functools import reduce
from json import JSONDecodeError
from ehri_chat.embeddings.embeddings_manager import RagEmbeddingsManager, GraphRagEmbeddingsManager
from ehri_chat.mcp.client import MCPClient
from ehri_chat.rag.graphrag_context_manager import GraphRagContextManager
from mistralai.client import Mistral
from mistralai.extra.run.context import RunContext
from mistralai.extra.mcp.sse import MCPClientSSE, SSEServerParams
from mistralai.extra.run.result import RunResult
from datetime import datetime, timedelta
import http.client
import json
from google import genai
from google.genai import types
import os
import logging
logger = logging.getLogger("ehri_chat")

@dataclass
class LLMGenerationOptions:
    model: str = None
    temperature: float = 0.1
    top_k: int = 20
    max_tokens: int = 2048

@dataclass
class Evaluation:
    relevant_context: str = ""
    generated_response: str = ""
    input_tokens: int = 0
    output_tokens: int = 0

class LLModel:
    def __init__(self):
        self.today = datetime.today()
        self.yesterday = self.today - timedelta(days=1)

    def generate_rag_system_prompt(self):
        return f"""You are a Large Language Model (LLM).
The current date is {self.today.strftime('%Y-%m-%d')}.
You are now being used in a Retrieval Augmented Generation (RAG) set up using data from the EHRI Portal which will feed some contextual information.
Whenever possible try to put the links to the provided context so users can easily expand their searches.
If this contextual information does not provide good answers just follow the general behaviour defined below.

When you're not sure about some information, you say that you don't have the information and don't make up anything.
If the user's question is not clear, ambiguous, or does not provide enough context for you to accurately answer the question, you do not try to answer it right away and you rather ask the user to clarify their request (e.g. "What are some good restaurants around me?" => "Where are you?" or "When is the next flight to Tokyo" => "Where do you travel from?").
You are always very attentive to dates, in particular you try to resolve dates (e.g. "yesterday" is {self.yesterday.strftime('%Y-%m-%d')}) and when asked about information at specific dates, you discard information that is at another date.
You follow these instructions in all languages, and always respond to the user in the language they use or request.
Next sections describe the capabilities that you have."""

    def generate_mcp_system_prompt(self):
        return f"""You are a Large Language Model (LLM).
    The current date is {self.today.strftime('%Y-%m-%d')}.
    You have the ability to consult an Model Context Protocol (MCP) server with data from the EHRI Portal.
    While the MCP is based on the EHRI-KG and semantic web technologies, you do not have to conform to any specific syntax and you must only use free text to consult it.
    Whenever possible try to put the links to the provided context so users can easily expand their searches.
    If this additional information does not provide good answers just follow the general behaviour defined below.

    When you're not sure about some information, you say that you don't have the information and don't make up anything.
    If the user's question is not clear, ambiguous, or does not provide enough context for you to accurately answer the question, you do not try to answer it right away and you rather ask the user to clarify their request (e.g. "What are some good restaurants around me?" => "Where are you?" or "When is the next flight to Tokyo" => "Where do you travel from?").
    You are always very attentive to dates, in particular you try to resolve dates (e.g. "yesterday" is {self.yesterday.strftime('%Y-%m-%d')}) and when asked about information at specific dates, you discard information that is at another date.
    You follow these instructions in all languages, and always respond to the user in the language they use or request.
    Next sections describe the capabilities that you have."""

    def generate_messages(self, user_prompt, history: list = None, mcp = False):
        last_message = [{
                    "role": "system",
                    "content": self.generate_rag_system_prompt() if not mcp else self.generate_mcp_system_prompt()
                },
                {
                    "content": user_prompt,
                    "role": "user"
                }]
        previous_messages = history if history is not None else []
        return previous_messages + last_message

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

    async def get_results_llm_with_rag(self, user_prompt: str, history: list = None, generation_options: LLMGenerationOptions = LLMGenerationOptions(), evaluation: Evaluation = None):
        relevant_context = "\n\n".join(RagEmbeddingsManager().retrieve_relevant_chunks(user_prompt))
        prompt_with_context = self.generate_rag_prompt(user_prompt, relevant_context)
        if evaluation is not None:
            evaluation.relevant_context = relevant_context
        return await self.get_results_llm(prompt_with_context, history = history, generation_options = generation_options, evaluation = evaluation)

    async def get_results_llm_with_graphrag(self, user_prompt, history: list = None, generation_options: LLMGenerationOptions = LLMGenerationOptions(), evaluation: Evaluation = None):
        relevant_context = "\n\n".join(GraphRagContextManager().retrieve_relevant_context(user_prompt))
        prompt_with_context = self.generate_rag_prompt(user_prompt, relevant_context)
        if evaluation is not None:
            evaluation.relevant_context = relevant_context
        return await self.get_results_llm(prompt_with_context, history = history, generation_options = generation_options, evaluation = evaluation)

    async def get_results_llm_with_mcp(self, user_prompt, history: list = None, generation_options: LLMGenerationOptions = LLMGenerationOptions(), evaluation: Evaluation = None):
        return await self.get_results_llm(user_prompt, history = history, mcp = MCPClient(), generation_options = generation_options, evaluation = evaluation)
    
class MistralAPI(LLModel):
    def __init__(self):
        super().__init__()
        self.client = Mistral(api_key=os.getenv("MISTRAL_API_KEY", ""))

    async def get_results_llm(self, user_prompt, history: list = None, mcp = None, generation_options: LLMGenerationOptions = LLMGenerationOptions(), evaluation: Evaluation = None):
        logger.debug(f"Generated prompt: {user_prompt}")
            # This is not yet supported for streamable http MCP servers
            # if mcp:
            #     mcp_client = MCPClientSSE(sse_params=SSEServerParams(url=self.mcp_client.server, timeout=100))
            #     async with RunContext(
            #             model=model if model is not None else "mistral-small-latest",
            #     ) as run_ctx:
            #         await run_ctx.register_mcp_client(mcp_client=mcp_client)
            #         response = await self.client.beta.conversations.run_stream_async(
            #             run_ctx=run_ctx,
            #             inputs=self.generate_messages(user_prompt),
            #         )
            #         run_result = None
            #         async for event in response:
            #             if isinstance(event, RunResult):
            #                 run_result = event
            #             else:
            #                 print(event)
            #         for entry in run_result.output_entries:
            #             yield entry
            # else:
        tools = None
        if mcp:
            await mcp.connect()
            tools = mcp.mcp_tools_to_openai(await mcp.get_tools())
        async def generate():
            try:
                agent_loop = True # first execution
                messages = self.generate_messages(user_prompt, history = history, mcp = mcp is not None)
                while agent_loop:
                    agent_loop = tools is not None
                    if mcp:
                        response = await self.client.chat.stream_async(
                            model=generation_options.model if generation_options.model is not None else "mistral-small-latest",
                            messages=messages,
                            tools=tools,
                            tool_choice="auto",
                            temperature=generation_options.temperature,
                            max_tokens=generation_options.max_tokens,
                            #top_k is not provided
                        )
                    else:
                        response = await self.client.chat.stream_async(
                            model=generation_options.model if generation_options.model is not None else "mistral-small-latest",
                            messages=messages,
                            temperature=generation_options.temperature,
                            max_tokens=generation_options.max_tokens,
                            # top_k is not provided
                        )
                    async for chunk in response:
                        finish_reason = chunk.data.choices[0].finish_reason
                        if chunk.data.usage is not None and evaluation is not None:
                            evaluation.input_tokens += chunk.data.usage.prompt_tokens
                            evaluation.output_tokens += chunk.data.usage.completion_tokens
                        if finish_reason not in ["tool_content", None, "tool_calls"]:
                            agent_loop = False
                        if chunk.data.choices[0].delta.tool_calls is not None and chunk.data.choices[0].delta.tool_calls:
                            tools_calls = []
                            tools_results = []
                            for tc in chunk.data.choices[0].delta.tool_calls:
                                args = json.loads(tc.function.arguments)
                                tools_calls.append({
                                    "id": tc.id,
                                    "function": {
                                        "name": tc.function.name,
                                        "arguments": args
                                    }
                                })
                                logger.debug(f"  → calling tool {tc.function.name}({args})")
                                tool_result = await mcp.call_mcp_tool(tc.function.name, args)
                                if evaluation is not None:
                                    evaluation.relevant_context += tool_result
                                tools_results.append({
                                    "role": "tool",
                                    "name": tc.function.name,
                                    "tool_call_id": tc.id,
                                    "content": tool_result,
                                })
                            messages.append({
                                "role": "assistant",
                                "content": chunk.data.choices[0].delta.content,
                                "tool_calls": tools_calls
                            })  # This is specifically needed for Mistral (see order of messages for Mistral API).
                            for tr in tools_results:
                                messages.append(tr)
                        else:
                            message_part = chunk.data.choices[0].delta.content
                            if message_part is not None and message_part.strip():
                                yield message_part
            finally:
                if mcp:
                    await mcp.close()
        return generate()

class GeminiAPI(LLModel):
    def __init__(self):
        super().__init__()

    async def get_results_llm(self, user_prompt, mcp = None, history: list = None, generation_options: LLMGenerationOptions = LLMGenerationOptions(), evaluation: Evaluation = None):
        client = genai.Client(
            api_key=os.environ.get("GEMINI_API_KEY"),
        )
        model = generation_options.model if generation_options.model is not None else "gemini-2.5-flash"
        history = [types.Content(
                role=message['role'] if message['role'] != "assistant" else "model",
                parts=[
                    types.Part.from_text(text=message['content']),
                ],
            ) for message in history] if history is not None else []
        new_message = [types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=user_prompt),
                ],
            ),
        ]
        contents = history + new_message
        if mcp:
            await mcp.connect()
        #tools = mcp.mcp_tools_to_gemini(await mcp.get_tools()) if mcp else []
        tools = [mcp.session] if mcp else []
        generate_content_config = types.GenerateContentConfig(
            temperature=generation_options.temperature,
            top_k=generation_options.top_k,
            max_output_tokens=generation_options.max_tokens,
            thinking_config=types.ThinkingConfig(
                thinking_budget=0,
            ),
            tools=tools,
            system_instruction=[
                types.Part.from_text(text=self.generate_mcp_system_prompt() if mcp else self.generate_rag_system_prompt()),
            ],
        )

        async def generate():
            try:
                async for chunk in await client.aio.models.generate_content_stream(
                        model=model,
                        contents=contents,
                        config=generate_content_config,
                ):
                    if chunk.usage_metadata and evaluation is not None:
                        evaluation.input_tokens = chunk.usage_metadata.prompt_token_count
                        evaluation.output_tokens = chunk.usage_metadata.total_token_count - chunk.usage_metadata.prompt_token_count
                    if evaluation is not None:
                        for candidate in chunk.candidates:
                            for part in candidate.content.parts:
                                if part.function_call is not None:
                                    tool_response = await mcp.call_mcp_tool(part.function_call.name, part.function_call.args)
                                    evaluation.relevant_context += tool_response
                    yield chunk.text
            finally:
                if mcp:
                    await mcp.close()
        return generate()

class LlamaCpp(LLModel):
    def __init__(self):
        super().__init__()
        self.endpoint = "localhost:11434"
        self.path = "/v1/chat/completions"
        self.models_list_path = "/v1/models"
        self.key = None
        self.https = False

    async def get_available_models(self):
        logger.debug(f"Obtaining available models for: {self.endpoint}")
        conn = http.client.HTTPConnection(self.endpoint) if not self.https else http.client.HTTPSConnection(self.endpoint)
        headers = {}
        if self.key is not None:
            headers['Authorization'] = f"Bearer {self.key}"
        conn.request('GET', self.models_list_path, None, headers)
        return json.loads(conn.getresponse().read())

    async def get_results_llm(self, user_prompt, mcp = None, history: list = None, generation_options: LLMGenerationOptions = LLMGenerationOptions(), evaluation: Evaluation = None):
        tools = None
        logger.debug(f"Generated prompt: {user_prompt}")
        conn = http.client.HTTPConnection(self.endpoint) if not self.https else http.client.HTTPSConnection(self.endpoint)
        headers = {'Content-type': 'application/json'}
        if self.key is not None:
            headers['Authorization'] = f"Bearer {self.key}"
        dict_initial_data = {
            "model": generation_options.model if generation_options.model is not None else "qwen3-vl:2b-instruct-q4_K_M",
            "messages": self.generate_messages(user_prompt, history = history, mcp = mcp is not None),
            "stream": True,
            "options": {
                "temperature": generation_options.temperature,
                "top_k": generation_options.top_k,
                "num_predict": generation_options.max_tokens
            }
        }
        if mcp:
            await mcp.connect()
            tools = mcp.mcp_tools_to_openai(await mcp.get_tools())
            dict_initial_data['tools'] = tools
        json_initial_data = json.dumps(dict_initial_data)
        async def generate(json_data):
            try:
                agent_loop = True # first execution
                iterations = 0
                while agent_loop and iterations <= 20:
                    iterations += 1
                    agent_loop = tools is not None
                    print(json_data)
                    conn.request('POST', self.path, json_data, headers)
                    previous_reasoning_state = False
                    chunks_history = []
                    for event in conn.getresponse():
                        tools_calls = []
                        tools_results = []
                        print(event)
                        if event.startswith(b"data:") and b"data: [DONE]" not in event:
                            chunk = json.loads(event.decode().rstrip("\n").replace("data:", "", 1))
                            if chunk['choices'][0]['delta'] is not None:
                                finish_reason = chunk['choices'][0].get('finish_reason', None)
                                if finish_reason not in ["tool_content", None, "tool_calls"]:
                                    agent_loop = False
                                if chunk['choices'][0]['delta'].get('tool_calls', None) is not None:
                                    chunks_history.append(chunk)
                                    for tc in chunk['choices'][0]['delta']['tool_calls']:
                                        if tc.get('id') is None or tc.get('function') is None or tc['function'].get('name') is None or tc['function'].get('arguments') is None:
                                            result = self.try_to_merge_with_history(chunk, chunks_history)
                                            if result is not None:
                                                tc = result
                                        arguments = tc['function']['arguments'] if tc['function']['arguments'] is not None else "{}"
                                        random_id = f"call_{uuid.uuid4()}"
                                        call_id = tc['id'] if tc['id'] is not None else random_id
                                        name = tc['function']['name'] if tc['function'] is not None and tc['function']['name'] is not None else "Unknown"
                                        try:
                                            args = json.loads(arguments)
                                            logger.debug(f"  → calling tool {tc['function']['name']}({arguments})")
                                            tool_result = await mcp.call_mcp_tool(name, args)
                                        except JSONDecodeError as e:
                                            tool_result = f"Error while parsing arguments as JSON. Try again and provide the mandatory arguments."
                                        if tc['id'] is None or tc['function']['name'] is None or tc['function']['arguments'] is None:
                                            tool_result = f"The tool calling request has no valid id or name."
                                        tools_calls.append({
                                            "id": call_id,
                                            "type": tc['type'] if hasattr(tc, 'type') else "function",
                                            "function": {
                                                "name": name,
                                                "arguments": arguments
                                            }
                                        })
                                        tools_results.append({
                                            "role": "tool",
                                            "name": name,
                                            "tool_call_id": call_id,
                                            "content": tool_result,
                                        })
                                    json_payload = json.loads(json_data)
                                    json_payload['messages'].append({
                                        "role": "assistant",
                                        "content": None,
                                        "tool_calls": tools_calls
                                    })
                                    for tr in tools_results:
                                        json_payload['messages'].append(tr)
                                    json_data = json.dumps(json_payload)
                                    with open("test_messages", "a") as f:
                                        f.write("\n\n" + json_data)
                                else:
                                    reasoning_part = chunk['choices'][0]['delta'].get("reasoning_content", "")
                                    message_part = chunk['choices'][0]['delta'].get('content', "")
                                    if reasoning_part:
                                        if previous_reasoning_state:
                                            yield reasoning_part if reasoning_part is not None else ""
                                        else:
                                            previous_reasoning_state = True
                                            yield f"<think>{reasoning_part}" if reasoning_part is not None else "<think>"

                                    if message_part:
                                        if previous_reasoning_state:
                                            previous_reasoning_state = False
                                            yield f"</think>\n{message_part}" if message_part is not None else "</think>"
                                        else:
                                            yield message_part if message_part is not None else ""
            finally:
                if mcp:
                    await mcp.close()
        return generate(json_initial_data)

    def try_to_merge_with_history(self, current_chunk: dict, chunks_history: list[dict]) -> dict | None:
        candidates = list(filter(lambda i: i['id'] == current_chunk['id'] and i['choices'][0]['delta'].get('tool_calls') is not None, chunks_history))
        if len(candidates) > 0:
            tc_candidates = list(map(lambda i: i['choices'][0]['delta']['tool_calls'][0], candidates))
            tc = reduce(lambda a, b: self.merge_with_lower_level(a, b) , tc_candidates)
            return tc if tc['id'] is not None and tc['function']['name'] is not None and tc['function']['arguments'] is not None else None
        else:
            return None

    def merge_with_lower_level(self, a, b):
        if a.get('id') is None or not a['id']:
            a['id'] = b['id']
        if a['function'].get('arguments') is None or not a['function']['arguments']:
            a['function']['arguments'] = b['function']['arguments']
            if str(a['function']['arguments']).startswith("{") and not str(a['function']['arguments']).endswith("}"):
                a['function']['arguments'] = a['function']['arguments'] + "}"
        if a['function'].get('name') is None or not a['function']['name']:
            a['function']['name'] = b['function']['name']
        return a