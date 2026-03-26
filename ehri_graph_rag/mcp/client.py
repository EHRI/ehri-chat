from contextlib import AsyncExitStack
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
import re

class MCPClient:
    def __init__(self):
        self.exit_stack = AsyncExitStack()
        self.transport = None
        self.session = None
        self.write = None
        self.read = None

    async def connect(self):
        self.transport = await self.exit_stack.enter_async_context(streamable_http_client("https://lod.ehri-project-test.eu/mcp"))
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

    async def call_mcp_tool(self, name: str, args: dict) -> str:
        result = await self.session.call_tool(name, args)
        return "\n".join(
            block.text for block in result.content if hasattr(block, "text")
        )

    async def get_tools(self):
        return self.mcp_tools_to_openai((await self.session.list_tools()).tools)

    async def close(self):
        await self.exit_stack.aclose()