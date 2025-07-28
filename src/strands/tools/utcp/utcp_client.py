"""Universal Tool Calling Protocol (UTCP) client wrapper for Strands integration.

This module provides the UTCPClient class which wraps the native UTCP client to provide
a simplified interface for the Strands agent framework. UTCP doesn't require
complex transport management or background threads, making this integration simple.
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

from utcp.client.utcp_client import UtcpClient
from utcp.shared.tool import Tool as UTCPTool

from ...types import PaginatedList
from ...types.exceptions import UTCPClientInitializationError
from ...types.tools import ToolResult, ToolResultStatus
from .utcp_agent_tool import UTCPAgentTool

logger = logging.getLogger(__name__)


class UTCPClient:
    """Wrapper for UTCP client that provides Strands-compatible interface.

    This class provides a simplified interface to UTCP functionality, handling
    tool discovery and execution through the native UTCP client.
    UTCP doesn't require complex connection management or background threads.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize a new UTCP client wrapper.

        Args:
            config: Configuration dictionary for the UTCP client.
                   Should contain 'providers_file_path' and optionally 'load_variables_from'
        """
        self._config = config
        self._utcp_client: Optional[UtcpClient] = None
        logger.debug("initializing UTCPClient wrapper with config: %s", config)

    async def __aenter__(self) -> "UTCPClient":
        """Async context manager entry point which initializes the UTCP client."""
        return await self.start()

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit point that cleans up resources."""
        await self.stop()

    async def start(self) -> "UTCPClient":
        """Initialize the UTCP client.

        Returns:
            self: The UTCPClient instance

        Raises:
            UTCPClientInitializationError: If the client fails to initialize
        """
        try:
            logger.debug("starting UTCP client")
            # Use the native UTCP client factory method
            self._utcp_client = await UtcpClient.create(config=self._config)
            logger.debug("UTCP client initialized successfully")
            return self
        except Exception as e:
            logger.exception("UTCP client failed to initialize")
            raise UTCPClientInitializationError("Failed to initialize UTCP client") from e

    async def stop(self) -> None:
        """Stop the UTCP client and clean up resources."""
        logger.debug("stopping UTCP client")
        # UTCP client doesn't require explicit cleanup
        self._utcp_client = None

    def list_tools_sync(self, pagination_token: Optional[str] = None) -> PaginatedList[UTCPAgentTool]:
        """Synchronously retrieve the list of available tools from UTCP providers.

        Args:
            pagination_token: Optional pagination token (not used by UTCP currently)

        Returns:
            PaginatedList[UTCPAgentTool]: A list of available tools adapted to the AgentTool interface

        Raises:
            UTCPClientInitializationError: If the client is not initialized
        """
        if self._utcp_client is None:
            raise UTCPClientInitializationError("UTCP client is not initialized")

        logger.debug("listing UTCP tools synchronously")
        
        try:
            # Get all tools from the UTCP client
            # Check if we're in an async context
            try:
                loop = asyncio.get_running_loop()
                # We're in an async context, need to handle differently
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self._utcp_client.tool_repository.get_tools())
                    utcp_tools = future.result()
            except RuntimeError:
                # No running loop, safe to use asyncio.run
                utcp_tools = asyncio.run(self._utcp_client.tool_repository.get_tools())
        except Exception as e:
            logger.error("Failed to get tools from UTCP client: %s", e)
            utcp_tools = []

        logger.debug("received %d tools from UTCP client", len(utcp_tools))

        # Convert to UTCPAgentTool instances
        agent_tools = [UTCPAgentTool(tool, self) for tool in utcp_tools]
        logger.debug("successfully adapted %d UTCP tools", len(agent_tools))
        
        # UTCP doesn't currently support pagination, so nextCursor is always None
        return PaginatedList[UTCPAgentTool](agent_tools, token=None)

    async def call_tool_async(
        self,
        tool_use_id: str,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> ToolResult:
        """Call a tool asynchronously using the UTCP client.

        Args:
            tool_use_id: Unique identifier for this tool use
            tool_name: Name of the tool to call (format: provider_name.tool_name)
            arguments: Arguments to pass to the tool

        Returns:
            ToolResult: The result of the tool call

        Raises:
            UTCPClientInitializationError: If the client is not initialized
        """
        if self._utcp_client is None:
            raise UTCPClientInitializationError("UTCP client is not initialized")

        logger.debug("calling UTCP tool '%s' asynchronously with tool_use_id=%s", tool_name, tool_use_id)

        try:
            # Call the tool using the native UTCP client
            result = await self._utcp_client.call_tool(tool_name=tool_name, arguments=arguments)
            return self._handle_tool_result(tool_use_id, result)
        except Exception as e:
            logger.exception("UTCP tool execution failed")
            return self._handle_tool_execution_error(tool_use_id, e)

    def call_tool_sync(
        self,
        tool_use_id: str,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> ToolResult:
        """Call a tool synchronously using the UTCP client.

        Args:
            tool_use_id: Unique identifier for this tool use
            tool_name: Name of the tool to call (format: provider_name.tool_name)
            arguments: Arguments to pass to the tool

        Returns:
            ToolResult: The result of the tool call

        Raises:
            UTCPClientInitializationError: If the client is not initialized
        """
        if self._utcp_client is None:
            raise UTCPClientInitializationError("UTCP client is not initialized")

        logger.debug("calling UTCP tool '%s' synchronously with tool_use_id=%s", tool_name, tool_use_id)

        try:
            # Use asyncio.run to call the async method synchronously
            result = asyncio.run(self._utcp_client.call_tool(tool_name=tool_name, arguments=arguments))
            return self._handle_tool_result(tool_use_id, result)
        except Exception as e:
            logger.exception("UTCP tool execution failed")
            return self._handle_tool_execution_error(tool_use_id, e)

    async def search_tools(self, query: str, max_results: int = 10) -> List[UTCPAgentTool]:
        """Search for tools using the UTCP client's search functionality.

        Args:
            query: Search query string
            max_results: Maximum number of results to return

        Returns:
            List[UTCPAgentTool]: List of matching tools
        """
        if self._utcp_client is None:
            raise UTCPClientInitializationError("UTCP client is not initialized")

        logger.debug("searching UTCP tools with query: '%s'", query)

        try:
            # Use UTCP's built-in search functionality (now properly async in v0.1.8+)
            utcp_tools = await self._utcp_client.search_tools(query=query, limit=max_results)
        except Exception as e:
            logger.error("Failed to search UTCP tools: %s", e)
            utcp_tools = []

        logger.debug("found %d matching UTCP tools", len(utcp_tools))

        # Convert to UTCPAgentTool instances
        agent_tools = [UTCPAgentTool(tool, self) for tool in utcp_tools]
        logger.debug("successfully adapted %d UTCP search results", len(agent_tools))

        return agent_tools

    def _handle_tool_execution_error(self, tool_use_id: str, exception: Exception) -> ToolResult:
        """Create error ToolResult with consistent logging."""
        return ToolResult(
            status="error",
            toolUseId=tool_use_id,
            content=[{"text": f"UTCP tool execution failed: {str(exception)}"}],
        )

    def _handle_tool_result(self, tool_use_id: str, result: Any) -> ToolResult:
        """Convert UTCP tool result to Strands ToolResult format.

        Args:
            tool_use_id: The tool use identifier
            result: The raw result from UTCP tool execution

        Returns:
            ToolResult: Formatted result for Strands
        """
        logger.debug("processing UTCP tool result for tool_use_id=%s", tool_use_id)

        # UTCP returns results as plain Python objects, so we need to convert them
        # to the expected ToolResult format
        try:
            if isinstance(result, dict):
                # If result is a dict, convert to JSON text
                content_text = json.dumps(result, indent=2)
            elif isinstance(result, (list, tuple)):
                # If result is a list/tuple, convert to JSON text
                content_text = json.dumps(result, indent=2)
            elif isinstance(result, str):
                # If result is already a string, use as-is
                content_text = result
            else:
                # For other types, convert to string
                content_text = str(result)

            status: ToolResultStatus = "success"
            logger.debug("UTCP tool execution completed successfully")

            return ToolResult(status=status, toolUseId=tool_use_id, content=[{"text": content_text}])
        except Exception as e:
            logger.exception("Failed to process UTCP tool result")
            return self._handle_tool_execution_error(tool_use_id, e)
