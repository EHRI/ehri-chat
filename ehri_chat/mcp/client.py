from contextlib import AsyncExitStack
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from google.genai import types
import re

class MCPClient:
    def __init__(self):
        self.exit_stack = AsyncExitStack()
        self.server = "https://lod.ehri-project-test.eu/mcp"
        self.transport = None
        self.session = None
        self.write = None
        self.read = None

    async def connect(self):
        self.transport = await self.exit_stack.enter_async_context(streamable_http_client(self.server))
        self.read, self.write, _ = self.transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.read, self.write))
        await self.session.initialize()

    def mcp_tools_to_openai(self, tools) -> list[dict]:
        """Converts MCP tool descriptors to the OpenAI tools schema."""
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": re.sub("```(.|\\n)+```", "", t.description), # removes the SPARQL queries from the description
                    "parameters": t.inputSchema,
                },
            }
            for t in tools
        ]

    async def mcp_tools_to_gemini(self, tools):
        return types.Tool(function_declarations=[
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.inputSchema,
            }
            for tool in tools
        ])

    async def call_mcp_tool(self, name: str, args: dict) -> str:
        result = await self.session.call_tool(name, args)
        if result.isError:
            return f"Error while calling tool, revise the function name, the passed arguments (and whether they are empty) and try again."
        else:
            return "\n".join(
                block.text for block in result.content if hasattr(block, "text")
            )

    async def get_tools(self):
        return (await self.session.list_tools()).tools

    async def close(self):
        await self.exit_stack.aclose()