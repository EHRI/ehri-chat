from ehri_graph_rag.embeddings.embeddings_manager import RagEmbeddingsManager, GraphRagEmbeddingsManager
from ehri_graph_rag.mcp.client import MCPClient
from ehri_graph_rag.rag.graphrag_context_manager import GraphRagContextManager
from mistralai.client import Mistral
from mistralai.extra.run.context import RunContext
from mistralai.extra.mcp.sse import MCPClientSSE, SSEServerParams
from mistralai.extra.run.result import RunResult
from datetime import datetime
import http.client
import json
from google import genai
from google.genai import types
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
        self.mcp_client = MCPClient()

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

    async def get_results_llm_with_rag(self, user_prompt, model = None):
        relevant_context = "\n\n".join(self.rag_embeddings_manager.retrieve_relevant_chunks(user_prompt))
        prompt_with_context = self.generate_rag_prompt(user_prompt, relevant_context)
        return await self.get_results_llm(prompt_with_context, model = model)

    async def get_results_llm_with_graphrag(self, user_prompt, model = None):
        relevant_context = "\n\n".join(self.graphrag_context_manager.retrieve_relevant_context(user_prompt))
        prompt_with_context = self.generate_rag_prompt(user_prompt, relevant_context)
        return await self.get_results_llm(prompt_with_context, model = model)

    async def get_results_llm_with_mcp(self, user_prompt, model = None):
        return await self.get_results_llm(user_prompt, mcp = True, model = model)
    
class MistralAPI(LLModel):
    def __init__(self):
        super().__init__()
        self.client = Mistral(api_key=os.getenv("MISTRAL_API_KEY", ""))

    async def get_results_llm(self, user_prompt, mcp = False, model = None):
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
            await self.mcp_client.connect()
            tools = self.mcp_client.mcp_tools_to_openai(await self.mcp_client.get_tools())
        async def generate():
            try:
                agent_loop = True # first execution
                messages = self.generate_messages(user_prompt)
                while agent_loop:
                    agent_loop = tools is not None
                    if mcp:
                        response = await self.client.chat.stream_async(
                            model=model if model is not None else "mistral-small-latest",
                            messages=messages,
                            tools=tools,
                            tool_choice="auto"
                        )
                    else:
                        response = await self.client.chat.stream_async(
                            model=model if model is not None else "mistral-small-latest",
                            messages=messages
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
                                tool_result = await self.mcp_client.call_mcp_tool(tc.function.name, args)
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
                            yield chunk.data.choices[0].delta.content
            finally:
                if mcp:
                    await self.mcp_client.close()
        return generate()

class GeminiAPI(LLModel):
    def __init__(self):
        super().__init__()

    async def get_results_llm(self, user_prompt, mcp = False, model = None):
        client = genai.Client(
            api_key=os.environ.get("GEMINI_API_KEY"),
        )
        model = "gemini-2.5-flash"
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=user_prompt),
                ],
            ),
        ]
        if mcp:
            await self.mcp_client.connect()
        #tools = self.mcp_client.mcp_tools_to_gemini(await self.mcp_client.get_tools()) if mcp else []
        tools = [self.mcp_client.session] if mcp else []
        generate_content_config = types.GenerateContentConfig(
            temperature=0.5,
            thinking_config=types.ThinkingConfig(
                thinking_budget=0,
            ),
            tools=tools,
            system_instruction=[
                types.Part.from_text(text=self.system_prompt),
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
                    await self.mcp_client.close()
        return generate()

class LlamaCpp(LLModel):
    def __init__(self):
        super().__init__()
        self.endpoint = "localhost:11434"
        self.path = "/v1/chat/completions"

    async def get_results_llm(self, user_prompt, mcp = False, model = None):
        tools = None
        logger.debug(f"Generated prompt: {user_prompt}")
        conn = http.client.HTTPConnection(self.endpoint)
        headers = {'Content-type': 'application/json'}
        dict_initial_data = {
            "messages": self.generate_messages(user_prompt),
            "stream": True,
        }
        if model is not None:
            dict_initial_data['model'] = model
        if mcp:
            await self.mcp_client.connect()
            tools = self.mcp_client.mcp_tools_to_openai(await self.mcp_client.get_tools())
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
                                        tool_result = await self.mcp_client.call_mcp_tool(tc['function']['name'], args)
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
                    await self.mcp_client.close()
        return generate(json_initial_data)