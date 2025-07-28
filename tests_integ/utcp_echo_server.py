"""
UTCP Echo Server for Integration Testing

This module implements a simple echo server using the Universal Tool Calling Protocol (UTCP).
It provides a basic tool that echoes back any input string, which is useful for
testing the UTCP communication flow and validating that messages are properly
transmitted between the client and server.

The server runs as an HTTP server and provides a UTCP manual for tool discovery.
"""

from typing import Any, Dict

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from utcp.shared.provider import HttpProvider
from utcp.shared.tool import Tool, ToolInputOutputSchema

app = FastAPI(title="UTCP Echo Server", version="1.0.0")

# Global variable to store the current port
CURRENT_PORT = 8003

def create_echo_tool() -> Tool:
    """Create the echo tool for the UTCP server."""

    # HTTP provider for echo tool - use dynamic port
    provider = HttpProvider(name="echo_server", url=f"http://127.0.0.1:{CURRENT_PORT}", http_method="POST")

    # Echo tool input schema
    input_schema = ToolInputOutputSchema(
        type="object",
        properties={"to_echo": {"type": "string", "description": "Text to echo back"}},
        required=["to_echo"],
        description="Echo tool input parameters",
    )

    # Echo tool output schema
    output_schema = ToolInputOutputSchema(
        type="object",
        properties={"echoed_text": {"type": "string", "description": "The echoed text"}},
        description="Echo tool output",
    )

    tool = Tool(
        name="echo",
        description="Echoes response back to the user",
        inputs=input_schema,
        outputs=output_schema,
        tool_provider=provider,
        tags=["echo", "text", "utility"],
    )

    return tool

# Create the echo tool
ECHO_TOOL = create_echo_tool()

@app.get("/utcp")
def get_utcp_manual():
    """Return the UTCP manual with the echo tool."""
    manual = {
        "version": "1.0",
        "tools": [
            {
                "name": ECHO_TOOL.name,
                "description": ECHO_TOOL.description,
                "inputs": {
                    "type": ECHO_TOOL.inputs.type,
                    "properties": ECHO_TOOL.inputs.properties,
                    "required": ECHO_TOOL.inputs.required or [],
                    "description": ECHO_TOOL.inputs.description or "",
                },
                "outputs": {
                    "type": ECHO_TOOL.outputs.type,
                    "properties": ECHO_TOOL.outputs.properties,
                    "description": ECHO_TOOL.outputs.description or "",
                },
                "tags": ECHO_TOOL.tags,
                "tool_provider": {
                    "provider_type": "http",
                    "name": ECHO_TOOL.tool_provider.name,
                    "url": f"http://127.0.0.1:{CURRENT_PORT}/tools/echo",
                    "http_method": "POST",
                },
            }
        ],
    }
    return JSONResponse(content=manual)

@app.post("/tools/echo")
def echo_tool(request: Dict[str, Any]):
    """Execute the echo tool."""
    try:
        to_echo = request.get("to_echo")

        if to_echo is None:
            return JSONResponse(status_code=400, content={"error": "Missing required parameter 'to_echo'"})

        return {"echoed_text": str(to_echo)}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Echo error: {str(e)}"})

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "UTCP Echo Server"}

def start_echo_server(port: int = 8003):
    """Start the UTCP echo server."""
    global CURRENT_PORT
    CURRENT_PORT = port
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")

if __name__ == "__main__":
    start_echo_server()
