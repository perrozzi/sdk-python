"""UTCP Tool Provider for Strands Integration.

This module provides the UTCPProvider class that integrates UTCP tools into the Strands framework
by creating AgentTool instances from UTCP providers.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

from ...types.tools import AgentTool
from .utcp_agent_tool import UTCPAgentTool
from .utcp_client import UTCPClient

logger = logging.getLogger(__name__)


class UTCPProvider:
    """Provider that creates Strands AgentTool instances from UTCP providers.
    
    This class acts as a bridge between UTCP and the Strands tool system, allowing
    UTCP tools to be used seamlessly within Strands agents.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the UTCP provider.
        
        Args:
            config: UTCP configuration dictionary containing providers_file_path and other settings
        """
        self.config = config
        self._utcp_client: Optional[UTCPClient] = None
        self._tools: List[AgentTool] = []
        logger.debug("initialized UTCP provider with config: %s", config)

    async def initialize(self) -> None:
        """Initialize the UTCP client and load tools."""
        try:
            logger.debug("initializing UTCP provider")
            self._utcp_client = UTCPClient(self.config)
            await self._utcp_client.start()
            
            # Load tools from UTCP client
            utcp_tools = self._utcp_client.list_tools_sync()
            self._tools = [UTCPAgentTool(tool, self._utcp_client) for tool in utcp_tools]
            
            logger.info("successfully initialized UTCP provider with %d tools", len(self._tools))
            
        except Exception as e:
            logger.exception("failed to initialize UTCP provider")
            raise

    async def cleanup(self) -> None:
        """Clean up the UTCP client resources."""
        if self._utcp_client:
            await self._utcp_client.stop()
            self._utcp_client = None
        self._tools = []
        logger.debug("cleaned up UTCP provider")

    def get_tools(self) -> List[AgentTool]:
        """Get all available UTCP tools as AgentTool instances.
        
        Returns:
            List of AgentTool instances created from UTCP tools
            
        Raises:
            RuntimeError: If the provider is not initialized
        """
        if self._utcp_client is None:
            raise RuntimeError("UTCP provider is not initialized. Call initialize() first.")
        
        return self._tools.copy()

    def get_tool_by_name(self, tool_name: str) -> Optional[AgentTool]:
        """Get a specific tool by name.
        
        Args:
            tool_name: Name of the tool to retrieve
            
        Returns:
            AgentTool instance if found, None otherwise
        """
        for tool in self._tools:
            if tool.tool_name == tool_name:
                return tool
        return None

    async def search_tools(self, query: str, max_results: int = 10) -> List[AgentTool]:
        """Search for tools using the UTCP client's search functionality.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            
        Returns:
            List of matching AgentTool instances
            
        Raises:
            RuntimeError: If the provider is not initialized
        """
        if self._utcp_client is None:
            raise RuntimeError("UTCP provider is not initialized. Call initialize() first.")
        
        try:
            utcp_search_results = await self._utcp_client.search_tools(query, max_results)
            return [UTCPAgentTool(tool, self._utcp_client) for tool in utcp_search_results]
        except Exception as e:
            logger.error("failed to search UTCP tools: %s", e)
            return []

    @property
    def is_initialized(self) -> bool:
        """Check if the provider is initialized."""
        return self._utcp_client is not None

    def __len__(self) -> int:
        """Return the number of available tools."""
        return len(self._tools)

    def __iter__(self):
        """Iterate over available tools."""
        return iter(self._tools)


def create_utcp_tools(config: Dict[str, Any]) -> List[AgentTool]:
    """Convenience function to create UTCP tools for use with Strands agents.
    
    This function provides a simple way to create UTCP tools that can be passed
    directly to a Strands agent's tools list.
    
    Args:
        config: UTCP configuration dictionary
        
    Returns:
        List of AgentTool instances from UTCP providers
        
    Example:
        ```python
        from strands.agent import Agent
        from strands.tools.utcp import create_utcp_tools
        
        utcp_config = {
            "providers_file_path": "/path/to/providers.json"
        }
        
        utcp_tools = create_utcp_tools(utcp_config)
        agent = Agent(tools=utcp_tools)
        ```
    """
    provider = UTCPProvider(config)
    
    # Run initialization in event loop
    try:
        loop = asyncio.get_running_loop()
        # If we're in an async context, we need to handle this differently
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, provider.initialize())
            future.result()
    except RuntimeError:
        # No running loop, safe to use asyncio.run
        asyncio.run(provider.initialize())
    
    return provider.get_tools()
