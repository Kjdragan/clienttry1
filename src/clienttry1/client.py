# client.py

import os
import logging
from typing import Optional, Dict, Any, List
from contextlib import AsyncExitStack
from datetime import datetime
from pathlib import Path
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = logging.getLogger(__name__)

TAVILY_MCP_PATH = Path(os.path.expanduser("~")) / "AppData/Roaming/npm/node_modules/tavily-mcp/build/index.js"

class MCPClient:
    """Dynamic MCP client implementation."""
    
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.capabilities: Dict[str, List[Dict[str, Any]]] = {
            'tools': [],
            'resources': [],
            'prompts': []
        }
        self.transport = None
        
    async def discover_capabilities(self) -> Dict[str, List[Dict[str, Any]]]:
        """Discover all available MCP capabilities."""
        if not self.session:
            raise ValueError("Client not initialized")
            
        try:
            # Discover tools
            tools_result = await self.session.list_tools()
            self.capabilities['tools'] = [
                {
                    'name': tool.name,
                    'description': tool.description,
                    'schema': tool.inputSchema
                }
                for tool in tools_result.tools
            ]
            logger.info(f"Discovered {len(self.capabilities['tools'])} tools")

            # Discover resources
            try:
                resources_result = await self.session.list_resources()
                self.capabilities['resources'] = [
                    {
                        'uri': resource.uri,
                        'name': resource.name,
                        'description': resource.description,
                        'mime_type': resource.mimeType
                    }
                    for resource in resources_result.resources
                ]
                logger.info(f"Discovered {len(self.capabilities['resources'])} resources")
            except Exception as e:
                logger.debug(f"Resource discovery failed: {e}")

            # Discover prompts
            try:
                prompts_result = await self.session.list_prompts()
                self.capabilities['prompts'] = [
                    {
                        'name': prompt.name,
                        'description': prompt.description,
                        'arguments': prompt.arguments
                    }
                    for prompt in prompts_result.prompts
                ]
                logger.info(f"Discovered {len(self.capabilities['prompts'])} prompts")
            except Exception as e:
                logger.debug(f"Prompt discovery failed: {e}")

            return self.capabilities

        except Exception as e:
            logger.error(f"Capability discovery failed: {e}")
            raise

    async def connect_to_server(self, command: str = "node", args: Optional[List[str]] = None, env: Optional[Dict[str, str]] = None):
        """Connect to an MCP server."""
        try:
            # Verify Tavily MCP server exists
            if not TAVILY_MCP_PATH.exists():
                raise FileNotFoundError(
                    f"Tavily MCP server not found at {TAVILY_MCP_PATH}. "
                    "Please install using: npm install -g tavily-mcp"
                )

            # Use the direct path to the Tavily MCP server
            server_args = [str(TAVILY_MCP_PATH)]
            server_env = env or {}
            
            logger.debug(f"Connecting to MCP server at: {TAVILY_MCP_PATH}")
            logger.debug(f"Server environment: {server_env}")
            
            server_params = StdioServerParameters(
                command=command,
                args=server_args,
                env=server_env
            )
            
            self.transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            
            self.stdio, self.write = self.transport
            self.session = await self.exit_stack.enter_async_context(
                ClientSession(self.stdio, self.write)
            )
            
            await self.session.initialize()
            
            # Discover capabilities
            await self.discover_capabilities()
            
            logger.info("Successfully connected to MCP server")
            
        except FileNotFoundError as e:
            logger.error(f"Server not found: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to connect to MCP server: {e}")
            raise

    async def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool with parameters."""
        if not self.session:
            raise ValueError("Client not initialized")
            
        tool = next(
            (t for t in self.capabilities['tools'] if t['name'] == tool_name),
            None
        )
        if not tool:
            raise ValueError(f"Tool not found: {tool_name}")
            
        try:
            logger.debug(f"Executing tool {tool_name} with parameters: {parameters}")
            result = await self.session.call_tool(tool_name, parameters)
            
            return {
                'tool': tool_name,
                'parameters': parameters,
                'result': result.content,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            raise

    async def cleanup(self):
        """Clean up resources."""
        try:
            if self.session:
                await self.exit_stack.aclose()
                logger.info("Cleaned up MCP client resources")
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            raise