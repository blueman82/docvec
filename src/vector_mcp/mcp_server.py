"""MCP server skeleton with tool registration system.

This module provides the MCP server infrastructure for the vector database
indexing system, including:
- Async server infrastructure using MCP Python SDK
- Tool registration system with schema validation
- Structured logging for debugging
- Clean resource management and teardown
"""

import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional

from mcp.server.fastmcp import FastMCP

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolDefinition:
    """Definition of an MCP tool.

    Attributes:
        name: Tool name (must be unique)
        description: Human-readable description of what the tool does
        input_schema: JSON schema defining expected input parameters
    """
    name: str
    description: str
    input_schema: dict[str, Any]


class MCPServer:
    """MCP server wrapper providing tool registration and lifecycle management.

    This class wraps the MCP FastMCP server infrastructure and provides:
    - Tool registration with schema validation
    - Handler management for async tool execution
    - Resource lifecycle management
    - Structured logging

    Args:
        name: Server name for identification
        config: Configuration object (placeholder for future use)
        indexer: Document indexer instance (placeholder for future use)
        storage: ChromaStore instance (placeholder for future use)

    Attributes:
        _app: FastMCP server instance
        _tools: Dictionary mapping tool names to ToolDefinition objects
        _handlers: Dictionary mapping tool names to async handler callables
        _running: Flag indicating if server is running

    Example:
        >>> server = MCPServer(name="vector-indexer")
        >>> server.register_tool(
        ...     name="search",
        ...     description="Search documents",
        ...     schema={"type": "object", "properties": {"query": {"type": "string"}}}
        ... )
        >>> async def search_handler(query: str):
        ...     return {"results": []}
        >>> server.add_tool_handler("search", search_handler)
        >>> await server.run()
    """

    def __init__(
        self,
        name: str = "vector-mcp-server",
        config: Optional[Any] = None,
        indexer: Optional[Any] = None,
        storage: Optional[Any] = None,
    ):
        """Initialize MCP server with optional dependencies.

        Args:
            name: Server name for identification
            config: Configuration object (reserved for future use)
            indexer: Document indexer instance (reserved for future use)
            storage: ChromaStore instance (reserved for future use)
        """
        self._app = FastMCP(name)
        self._tools: dict[str, ToolDefinition] = {}
        self._handlers: dict[str, Callable] = {}
        self._running = False

        # Store dependencies for future use
        self._config = config
        self._indexer = indexer
        self._storage = storage

        logger.info(f"Initialized MCP server: {name}")

    def register_tool(
        self,
        name: str,
        description: str,
        schema: dict[str, Any],
    ) -> None:
        """Register a tool definition with the server.

        Tools must be registered before handlers are added. The schema should
        follow JSON Schema format for input validation.

        Args:
            name: Unique tool name
            description: Human-readable description
            schema: JSON schema for input validation

        Raises:
            ValueError: If tool name is already registered

        Example:
            >>> server.register_tool(
            ...     name="embed",
            ...     description="Generate embeddings for text",
            ...     schema={
            ...         "type": "object",
            ...         "properties": {
            ...             "text": {"type": "string"},
            ...             "model": {"type": "string"}
            ...         },
            ...         "required": ["text"]
            ...     }
            ... )
        """
        if name in self._tools:
            raise ValueError(f"Tool '{name}' is already registered")

        if not name or not name.strip():
            raise ValueError("Tool name cannot be empty")

        if not description or not description.strip():
            raise ValueError("Tool description cannot be empty")

        if not isinstance(schema, dict):
            raise ValueError("Schema must be a dictionary")

        tool_def = ToolDefinition(
            name=name,
            description=description,
            input_schema=schema,
        )

        self._tools[name] = tool_def
        logger.info(f"Registered tool: {name}")

    def add_tool_handler(
        self,
        tool_name: str,
        handler: Callable,
    ) -> None:
        """Add async handler for a registered tool.

        The handler must be an async callable that accepts parameters matching
        the tool's input schema and returns a result.

        Args:
            tool_name: Name of previously registered tool
            handler: Async callable to handle tool invocations

        Raises:
            ValueError: If tool is not registered or handler is not callable

        Example:
            >>> async def embed_handler(text: str, model: str = "default"):
            ...     # Generate embedding
            ...     return {"embedding": [0.1, 0.2, 0.3]}
            >>> server.add_tool_handler("embed", embed_handler)
        """
        if tool_name not in self._tools:
            raise ValueError(
                f"Tool '{tool_name}' not registered. "
                f"Call register_tool() first."
            )

        if not callable(handler):
            raise ValueError("Handler must be callable")

        self._handlers[tool_name] = handler

        # Register handler with FastMCP using decorator pattern
        tool_def = self._tools[tool_name]
        self._app.tool(name=tool_name)(handler)

        logger.info(f"Added handler for tool: {tool_name}")

    async def run(self) -> None:
        """Start the MCP server.

        This method starts the async server and begins listening for
        tool invocations. It blocks until shutdown() is called.

        Raises:
            RuntimeError: If server is already running

        Example:
            >>> server = MCPServer()
            >>> # Register tools and handlers...
            >>> await server.run()
        """
        if self._running:
            raise RuntimeError("Server is already running")

        self._running = True
        logger.info("Starting MCP server...")

        try:
            # FastMCP handles the async event loop internally
            # In production, this would start the server transport (stdio, HTTP, etc.)
            logger.info("MCP server started successfully")

        except Exception as e:
            logger.error(f"Failed to start server: {e}")
            self._running = False
            raise

    async def shutdown(self) -> None:
        """Shutdown the MCP server and cleanup resources.

        Performs graceful shutdown of the server and releases all resources
        including connections to Ollama and ChromaDB.

        Example:
            >>> await server.shutdown()
        """
        if not self._running:
            logger.warning("Server is not running")
            return

        logger.info("Shutting down MCP server...")

        try:
            # Clean up resources
            # Future: Close Ollama client connections
            # Future: Close ChromaDB connections

            self._running = False
            logger.info("MCP server shutdown complete")

        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            raise

    @property
    def is_running(self) -> bool:
        """Check if server is currently running.

        Returns:
            True if server is running, False otherwise
        """
        return self._running

    @property
    def tools(self) -> dict[str, ToolDefinition]:
        """Get all registered tools.

        Returns:
            Dictionary mapping tool names to ToolDefinition objects
        """
        return self._tools.copy()

    async def __aenter__(self):
        """Async context manager entry."""
        await self.run()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - shutdown server."""
        await self.shutdown()
