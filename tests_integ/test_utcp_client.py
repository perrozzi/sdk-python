"""
UTCP Client Integration Tests

This module provides comprehensive integration tests for the UTCP (Universal Tool Calling Protocol)
client implementation. These tests validate the complete workflow from tool discovery through
execution using real UTCP servers.

The tests are architecture and enhanced capabilities of UTCP integration.
"""

import json
import os
import tempfile
import threading
import time
from typing import List

import pytest
import requests

from strands import Agent
from strands.tools.utcp.utcp_client import UTCPClient
from strands.types.content import Message
from strands.types.exceptions import UTCPClientInitializationError
from strands.types.tools import ToolUse

def start_utcp_calculator_server(port: int = 8002):
    """
    Initialize and start a UTCP calculator server for integration testing.

    This function creates a FastAPI server that provides calculator and image
    generation tools via UTCP protocol. The server uses HTTP transport for
    communication, making it accessible over standard HTTP requests.
    """
    from tests_integ.utcp_calculator_server import start_calculator_server

    start_calculator_server(port)

def start_utcp_echo_server(port: int = 8003):
    """
    Initialize and start a UTCP echo server for integration testing.

    This function creates a FastAPI server that provides an echo tool
    via UTCP protocol for testing basic communication flow.
    """
    from tests_integ.utcp_echo_server import start_echo_server

    start_echo_server(port)

def wait_for_server(url: str, timeout: int = 10):
    """Wait for a server to be ready by checking its health endpoint."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{url}/health", timeout=1)
            if response.status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(0.1)
    return False

def create_test_providers_config(calculator_port: int = 8002, echo_port: int = 8003) -> str:
    """Create a temporary providers.json file for testing."""
    providers_config = [
        {
            "name": "calculator_server",
            "provider_type": "http",
            "url": f"http://127.0.0.1:{calculator_port}/utcp",
            "http_method": "GET",
        },
        {
            "name": "echo_server",
            "provider_type": "http",
            "url": f"http://127.0.0.1:{echo_port}/utcp",
            "http_method": "GET",
        },
    ]    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(providers_config, f, indent=2)
        return f.name

@pytest.mark.asyncio
async def test_utcp_client():
    """
    Test UTCP client integration with calculator and echo tools.

    This test should yield output the simple UTCP integration approach:
    - No background threads required
    - Simple HTTP-based tool discovery
    - Direct async tool execution
    - Built-in search capabilities
    """
    # Start test servers
    calculator_thread = threading.Thread(target=start_utcp_calculator_server, kwargs={"port": 8002}, daemon=True)
    calculator_thread.start()

    echo_thread = threading.Thread(target=start_utcp_echo_server, kwargs={"port": 8003}, daemon=True)
    echo_thread.start()

    # Wait for servers to start
    assert wait_for_server("http://127.0.0.1:8002"), "Calculator server failed to start"
    assert wait_for_server("http://127.0.0.1:8003"), "Echo server failed to start"

    # Create providers configuration
    providers_file = create_test_providers_config()

    try:
        # Configure UTCP client
        utcp_config = {"providers_file_path": providers_file}

        # Test UTCP client integration
        async with UTCPClient(utcp_config) as utcp_client:
            # Get all available tools
            all_tools = utcp_client.list_tools_sync()

            # Verify we have the expected tools (with sanitized names for Bedrock compatibility)
            tool_names = [tool.tool_name for tool in all_tools]
            assert "calculator_server_calculator" in tool_names, f"Calculator tool not found in {tool_names}"
            assert "echo_server_echo" in tool_names, f"Echo tool not found in {tool_names}"

            # Create agent with UTCP tools
            agent = Agent(tools=all_tools)

            # Test the complete workflow: calculate and echo
            agent("add 1 and 2, then echo the result back to me")

            # Verify tool usage
            tool_use_content_blocks = _messages_to_content_blocks(agent.messages)
            tool_names_used = [block["name"] for block in tool_use_content_blocks]

            assert "calculator_server_calculator" in tool_names_used, f"Calculator tool was not used. Used: {tool_names_used}"
            assert "echo_server_echo" in tool_names_used, f"Echo tool was not used. Used: {tool_names_used}"

            # Test image generation and color detection
            image_prompt = """
            Generate a custom image, then tell me if the image is red, blue, yellow, pink, orange, or green. 
            RESPOND ONLY WITH THE COLOR
            """

            image_response = agent(image_prompt)

            # Check if the response contains "yellow" (our test image color)
            response_text = " ".join([block["text"] for block in image_response.message["content"] if "text" in block])

            assert "yellow" in response_text.lower(), f"Expected 'yellow' in response: {response_text}"

    finally:
        # Cleanup
        if os.path.exists(providers_file):
            os.unlink(providers_file)

@pytest.mark.asyncio
async def test_can_reuse_utcp_client():
    """
    Test that UTCP client can be reused multiple times.

    This demonstrates the simpler lifecycle management where no complex background thread cleanup is required.
    """
    # Start echo server
    echo_thread = threading.Thread(target=start_utcp_echo_server, kwargs={"port": 8004}, daemon=True)
    echo_thread.start()

    assert wait_for_server("http://127.0.0.1:8004"), "Echo server failed to start"

    # Create providers configuration for echo server only
    providers_config = [
        {"name": "echo_server", "provider_type": "http", "url": "http://127.0.0.1:8004/utcp", "http_method": "GET"}
    ]

    providers_file = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(providers_config, f, indent=2)
            providers_file = f.name

        utcp_config = {"providers_file_path": providers_file}
        utcp_client = UTCPClient(utcp_config)

        # First use
        async with utcp_client:
            tools = utcp_client.list_tools_sync()
            assert len(tools) > 0, "No tools found"

        # Second use - should work without issues
        async with utcp_client:
            agent = Agent(tools=utcp_client.list_tools_sync())
            agent("echo the following to me: DOG")

            tool_use_content_blocks = _messages_to_content_blocks(agent.messages)
            assert any([block["name"] == "echo_server_echo" for block in tool_use_content_blocks])

    finally:
        if providers_file and os.path.exists(providers_file):
            os.unlink(providers_file)

@pytest.mark.asyncio
async def test_utcp_tool_search_functionality():
    """
    Test UTCP's built-in tool search functionality.
    """
    # Start calculator server
    calculator_thread = threading.Thread(target=start_utcp_calculator_server, kwargs={"port": 8005}, daemon=True)
    calculator_thread.start()

    assert wait_for_server("http://127.0.0.1:8005"), "Calculator server failed to start"

    providers_config = [
        {
            "name": "calculator_server",
            "provider_type": "http",
            "url": "http://127.0.0.1:8005/utcp",
            "http_method": "GET",
        }
    ]

    providers_file = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(providers_config, f, indent=2)
            providers_file = f.name

        utcp_config = {"providers_file_path": providers_file}

        async with UTCPClient(utcp_config) as utcp_client:
            # Test search functionality
            math_tools = await utcp_client.search_tools("math", max_results=10)
            assert len(math_tools) > 0, "No math tools found"

            calculator_tools = await utcp_client.search_tools("calculator", max_results=10)
            assert len(calculator_tools) > 0, "No calculator tools found"

            # Verify search results contain expected tools (with sanitized names)
            tool_names = [tool.tool_name for tool in calculator_tools]
            assert "calculator_server_calculator" in tool_names, f"Calculator tool not found in search results: {tool_names}"

    finally:
        if providers_file and os.path.exists(providers_file):
            os.unlink(providers_file)

@pytest.mark.asyncio
async def test_utcp_multiple_provider_types():
    """
    Test UTCP with multiple provider types.
    """
    # Start HTTP server
    calculator_thread = threading.Thread(target=start_utcp_calculator_server, kwargs={"port": 8006}, daemon=True)
    calculator_thread.start()

    assert wait_for_server("http://127.0.0.1:8006"), "Calculator server failed to start"

    # Create configuration with multiple provider types
    providers_config = [
        {"name": "http_calculator", "provider_type": "http", "url": "http://127.0.0.1:8006/utcp", "http_method": "GET"},
        # Note: In a real scenario, you could add CLI, WebSocket, gRPC providers here
        # For this test, we'll focus on HTTP providers with different configurations
        {
            "name": "http_calculator_alt",
            "provider_type": "http",
            "url": "http://127.0.0.1:8006/utcp",
            "http_method": "GET",
            "content_type": "application/json",
        },
    ]

    providers_file = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(providers_config, f, indent=2)
            providers_file = f.name

        utcp_config = {"providers_file_path": providers_file}

        async with UTCPClient(utcp_config) as utcp_client:
            tools = utcp_client.list_tools_sync()

            # Should have tools from multiple providers
            assert len(tools) >= 2, f"Expected at least 2 tools, got {len(tools)}"

            # Test that tools work
            agent = Agent(tools=tools)
            agent("calculate 5 plus 3")

            tool_use_content_blocks = _messages_to_content_blocks(agent.messages)
            assert len(tool_use_content_blocks) > 0, "No tools were used"

    finally:
        if providers_file and os.path.exists(providers_file):
            os.unlink(providers_file)

@pytest.mark.asyncio
async def test_utcp_error_handling():
    """
    Test UTCP error handling with invalid configurations and server failures.
    """
    # Test with invalid providers file
    invalid_config = {"providers_file_path": "/nonexistent/file.json"}

    utcp_client = UTCPClient(invalid_config)

    # Should fail during start(), not during init
    with pytest.raises(UTCPClientInitializationError):
        async with utcp_client:
            pass

    # Test with unreachable server
    providers_config = [
        {
            "name": "unreachable_server",
            "provider_type": "http",
            "url": "http://127.0.0.1:9999/utcp",  # Non-existent server
            "http_method": "GET",
        }
    ]

    providers_file = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(providers_config, f, indent=2)
            providers_file = f.name

        utcp_config = {"providers_file_path": providers_file}

        # UTCP client handles unreachable servers gracefully by registering them with 0 tools
        # rather than raising an exception, so we test that it initializes successfully
        async with UTCPClient(utcp_config) as utcp_client:
            tools = utcp_client.list_tools_sync()
            # Should have 0 tools since the server is unreachable
            assert len(tools) == 0, f"Expected 0 tools from unreachable server, got {len(tools)}"

    finally:
        if providers_file and os.path.exists(providers_file):
            os.unlink(providers_file)

def _messages_to_content_blocks(messages: List[Message]) -> List[ToolUse]:
    """Extract tool use content blocks from agent messages."""
    return [block["toolUse"] for message in messages for block in message["content"] if "toolUse" in block]
