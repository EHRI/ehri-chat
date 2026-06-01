from dataclasses import dataclass
from ehri_graph_rag.embeddings.embeddings_manager import RagEmbeddingsManager, GraphRagEmbeddingsManager
from ehri_graph_rag.mcp.client import MCPClient
from ehri_graph_rag.rag.graphrag_context_manager import GraphRagContextManager
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
logger = logging.getLogger("ehri_graph_rag")

@dataclass
class LLMGenerationOptions:
    model: str = None,
    temperature: float = 0.1
    top_k: int = 20,
    max_tokens: int = 2048

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

    async def get_results_llm_with_rag(self, user_prompt: str, history: list = None, generation_options: LLMGenerationOptions = LLMGenerationOptions()):
        relevant_context = "\n\n".join(RagEmbeddingsManager().retrieve_relevant_chunks(user_prompt))
        prompt_with_context = self.generate_rag_prompt(user_prompt, relevant_context)
        return await self.get_results_llm(prompt_with_context, history = history, generation_options = generation_options)

    async def get_results_llm_with_graphrag(self, user_prompt, history: list = None, generation_options: LLMGenerationOptions = LLMGenerationOptions()):
        relevant_context = "\n\n".join(GraphRagContextManager().retrieve_relevant_context(user_prompt))
        prompt_with_context = self.generate_rag_prompt(user_prompt, relevant_context)
        return await self.get_results_llm(prompt_with_context, history = history, generation_options = generation_options)

    async def get_results_llm_with_mcp(self, user_prompt, history: list = None, generation_options: LLMGenerationOptions = LLMGenerationOptions()):
        return await self.get_results_llm(user_prompt, history = history, mcp = MCPClient(), generation_options = generation_options)
    
class MistralAPI(LLModel):
    def __init__(self):
        super().__init__()
        self.client = Mistral(api_key=os.getenv("MISTRAL_API_KEY", ""))

    async def get_results_llm(self, user_prompt, history: list = None, mcp = None, generation_options: LLMGenerationOptions = LLMGenerationOptions()):
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

    async def get_results_llm(self, user_prompt, mcp = None, history: list = None, generation_options: LLMGenerationOptions = LLMGenerationOptions()):
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

    async def get_results_llm(self, user_prompt, mcp = None, history: list = None, generation_options: LLMGenerationOptions = LLMGenerationOptions()):
        tools = None
        logger.debug(f"Generated prompt: {user_prompt}")
        conn = http.client.HTTPConnection(self.endpoint)
        headers = {'Content-type': 'application/json'}
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
                while agent_loop:
                    agent_loop = tools is not None
                    conn.request('POST', self.path, json_data, headers)
                    for event in conn.getresponse():
                        if event.startswith(b"data:") and b"data: [DONE]" not in event:
                            chunk = json.loads(event.decode().rstrip("\n").replace("data:", "", 1))
                            if chunk['choices'][0]['delta'] is not None:
                                finish_reason = chunk['choices'][0].get('finish_reason', None)
                                if finish_reason not in ["tool_content", None, "tool_calls"]:
                                    agent_loop = False
                                if chunk['choices'][0]['delta'].get('tool_calls', None) is not None:
                                    for tc in chunk['choices'][0]['delta']['tool_calls']:
                                        args = json.loads(tc['function']['arguments'])
                                        logger.debug(f"  → calling tool {tc['function']['name']}({args})")
                                        tool_result = await mcp.call_mcp_tool(tc['function']['name'], args)
                                        json_payload = json.loads(json_data)
                                        json_payload['messages'].append({
                                            "role": "tool",
                                            "tool_call_id": tc['id'],
                                            "content": tool_result,
                                        })
                                        json_data = json.dumps(json_payload)
                                else:
                                    message_part = chunk['choices'][0]['delta'].get('content', "")
                                    yield message_part if message_part is not None else ""
            finally:
                if mcp:
                    await mcp.close()
        return generate(json_initial_data)