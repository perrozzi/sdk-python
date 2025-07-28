"""Simple tests for basic UTCP functionality."""

import pytest

from strands.tools.utcp.utcp_client_simple import UTCPClientSimple
from strands.types.exceptions import UTCPClientInitializationError

@pytest.mark.asyncio
async def test_simple_utcp_client_basic_functionality():
    """Test that the simple UTCP client basic functionality works."""
    config = {"test": "config"}
    
    async with UTCPClientSimple(config) as client:
        # Test tool listing
        tools = client.list_tools_sync()
        assert len(tools) == 2
        assert tools[0].tool_name == "mock_tool_1"
        assert tools[1].tool_name == "mock_tool_2"
        
        # Test tool calling
        result = await client.call_tool_async(
            tool_use_id="test-123",
            tool_name="mock_tool_1",
            arguments={"input": "test"}
        )
        
        assert result["status"] == "success"
        assert result["toolUseId"] == "test-123"
        assert len(result["content"]) == 1
        assert "Mock result for mock_tool_1" in result["content"][0]["text"]

def test_simple_utcp_client_not_initialized():
    """Test error when client is not initialized."""
    client = UTCPClientSimple({})
    
    with pytest.raises(UTCPClientInitializationError):
        client.list_tools_sync()

@pytest.mark.asyncio
async def test_simple_utcp_client_context_manager():
    """Test that context manager works properly."""
    config = {"test": "config"}
    
    client = UTCPClientSimple(config)
    assert client._utcp_client is None
    
    async with client:
        assert client._utcp_client is not None
    
    assert client._utcp_client is None

def test_simple_utcp_client_init():
    """Test client initialization."""
    config = {"providers_file_path": "/tmp/test.json"}
    client = UTCPClientSimple(config)
    
    assert client._config == config
    assert client._utcp_client is None
