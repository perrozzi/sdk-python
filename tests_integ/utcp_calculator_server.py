"""
UTCP Calculator Server for Integration Testing

This module implements a simple calculator server using the Universal Tool Calling Protocol (UTCP).
It provides calculator tools for performing basic arithmetic operations and image generation,
which are useful for testing the UTCP communication flow and validating that tools work properly
with the Strands agent framework.

The server can run as both an HTTP server and provide a UTCP manual for tool discovery.
"""

import base64
from typing import Any, Dict

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from utcp.shared.provider import HttpProvider
from utcp.shared.tool import Tool, ToolInputOutputSchema

app = FastAPI(title="UTCP Calculator Server", version="1.0.0")

def create_calculator_tools() -> Dict[str, Tool]:
    """Create calculator tools for the UTCP server."""

    # HTTP provider for calculator tools
    provider = HttpProvider(name="calculator_server", url="http://127.0.0.1:8002", http_method="POST")

    # Calculator tool input schema
    calc_input = ToolInputOutputSchema(
        type="object",
        properties={
            "x": {"type": "integer", "description": "First number"},
            "y": {"type": "integer", "description": "Second number"},
        },
        required=["x", "y"],
        description="Calculator input parameters",
    )

    # Calculator tool output schema
    calc_output = ToolInputOutputSchema(
        type="object",
        properties={"result": {"type": "integer", "description": "Calculation result"}},
        description="Calculator result",
    )

    # Image generation input schema (no parameters needed)
    image_input = ToolInputOutputSchema(
        type="object", properties={}, description="No parameters required for image generation"
    )

    # Image generation output schema
    image_output = ToolInputOutputSchema(
        type="object",
        properties={
            "image_data": {"type": "string", "description": "Base64 encoded image"},
            "mime_type": {"type": "string", "description": "Image MIME type"},
        },
        description="Generated image data",
    )

    tools = {
        "calculator": Tool(
            name="calculator",
            description="Calculator tool which performs addition",
            inputs=calc_input,
            outputs=calc_output,
            tool_provider=provider,
            tags=["math", "calculator", "addition"],
        ),
        "generate_custom_image": Tool(
            name="generate_custom_image",
            description="Generates a custom yellow image",
            inputs=image_input,
            outputs=image_output,
            tool_provider=provider,
            tags=["image", "generation", "custom"],
        ),
    }

    return tools

# Create tools
TOOLS = create_calculator_tools()

@app.get("/utcp")
def get_utcp_manual():
    """Return the UTCP manual with available tools."""
    manual = {
        "version": "1.0",
        "tools": [
            {
                "name": tool.name,
                "description": tool.description,
                "inputs": {
                    "type": tool.inputs.type,
                    "properties": tool.inputs.properties,
                    "required": tool.inputs.required or [],
                    "description": tool.inputs.description or "",
                },
                "outputs": {
                    "type": tool.outputs.type,
                    "properties": tool.outputs.properties,
                    "description": tool.outputs.description or "",
                },
                "tags": tool.tags,
                "tool_provider": {
                    "provider_type": "http",
                    "name": tool.tool_provider.name,
                    "url": f"http://127.0.0.1:8002/tools/{tool.name}",
                    "http_method": "POST",
                },
            }
            for tool in TOOLS.values()
        ],
    }
    return JSONResponse(content=manual)

@app.post("/tools/calculator")
def calculator_tool(request: Dict[str, Any]):
    """Execute the calculator tool."""
    try:
        x = request.get("x")
        y = request.get("y")

        if x is None or y is None:
            return JSONResponse(status_code=400, content={"error": "Missing required parameters x and y"})

        result = int(x) + int(y)
        return {"result": result}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Calculator error: {str(e)}"})

@app.post("/tools/generate_custom_image")
def generate_custom_image_tool(request: Dict[str, Any]):
    """Execute the image generation tool."""
    try:
        # Try to read the yellow.png file
        try:
            with open("tests_integ/yellow.png", "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
                return {"image_data": encoded_image, "mime_type": "image/png", "color": "yellow"}
        except FileNotFoundError:
            # If file not found, create a simple response indicating yellow
            return {
                "image_data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChAGA",
                "mime_type": "image/png",
                "color": "yellow",
                "note": "Simulated yellow image",
            }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Image generation error: {str(e)}"})

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "UTCP Calculator Server"}

def start_calculator_server(port: int = 8002):
    """Start the UTCP calculator server."""
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")

if __name__ == "__main__":
    start_calculator_server()
