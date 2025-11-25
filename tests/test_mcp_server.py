"""Tests for MCP server skeleton and tool registration.

This module tests the MCP server infrastructure including:
- Server initialization
- Tool registration with schema validation
- Handler management
- Server lifecycle (run/shutdown)
- Error handling and validation
"""

import pytest

from docvec.mcp_server import MCPServer, ToolDefinition


class TestToolDefinition:
    """Test ToolDefinition dataclass."""

    def test_tool_definition_creation(self):
        """Test creating a tool definition."""
        schema = {
            "type": "object",
            "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}},
            "required": ["query"],
        }

        tool_def = ToolDefinition(
            name="search",
            description="Search documents",
            input_schema=schema,
        )

        assert tool_def.name == "search"
        assert tool_def.description == "Search documents"
        assert tool_def.input_schema == schema

    def test_tool_definition_immutability(self):
        """Test that tool definitions are frozen."""
        tool_def = ToolDefinition(
            name="test",
            description="Test tool",
            input_schema={"type": "object"},
        )

        with pytest.raises(AttributeError):
            tool_def.name = "modified"


class TestMCPServerInitialization:
    """Test MCP server initialization."""

    def test_init_with_default_name(self):
        """Test initialization with default server name."""
        server = MCPServer()

        assert server is not None
        assert not server.is_running
        assert len(server.tools) == 0

    def test_init_with_custom_name(self):
        """Test initialization with custom server name."""
        server = MCPServer(name="custom-server")

        assert server is not None
        assert not server.is_running

    def test_init_with_dependencies(self):
        """Test initialization with config, indexer, and storage."""
        config = {"key": "value"}
        indexer = object()
        storage = object()

        server = MCPServer(
            name="test-server",
            config=config,
            indexer=indexer,
            storage=storage,
        )

        assert server is not None
        assert server._config == config
        assert server._indexer is indexer
        assert server._storage is storage

    def test_init_creates_empty_tools_registry(self):
        """Test that initialization creates empty tool registry."""
        server = MCPServer()

        assert isinstance(server.tools, dict)
        assert len(server.tools) == 0


class TestToolRegistration:
    """Test tool registration functionality."""

    def test_register_tool_basic(self):
        """Test basic tool registration."""
        server = MCPServer()
        schema = {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        }

        server.register_tool(
            name="embed",
            description="Generate embeddings",
            schema=schema,
        )

        assert "embed" in server.tools
        assert server.tools["embed"].name == "embed"
        assert server.tools["embed"].description == "Generate embeddings"
        assert server.tools["embed"].input_schema == schema

    def test_register_multiple_tools(self):
        """Test registering multiple tools."""
        server = MCPServer()

        server.register_tool(
            name="tool1",
            description="First tool",
            schema={"type": "object"},
        )
        server.register_tool(
            name="tool2",
            description="Second tool",
            schema={"type": "object"},
        )

        assert len(server.tools) == 2
        assert "tool1" in server.tools
        assert "tool2" in server.tools

    def test_register_duplicate_tool_raises_error(self):
        """Test that registering duplicate tool name raises error."""
        server = MCPServer()

        server.register_tool(
            name="duplicate",
            description="First registration",
            schema={"type": "object"},
        )

        with pytest.raises(ValueError, match="already registered"):
            server.register_tool(
                name="duplicate",
                description="Second registration",
                schema={"type": "object"},
            )

    def test_register_tool_empty_name_raises_error(self):
        """Test that empty tool name raises error."""
        server = MCPServer()

        with pytest.raises(ValueError, match="name cannot be empty"):
            server.register_tool(
                name="",
                description="Valid description",
                schema={"type": "object"},
            )

    def test_register_tool_whitespace_name_raises_error(self):
        """Test that whitespace-only name raises error."""
        server = MCPServer()

        with pytest.raises(ValueError, match="name cannot be empty"):
            server.register_tool(
                name="   ",
                description="Valid description",
                schema={"type": "object"},
            )

    def test_register_tool_empty_description_raises_error(self):
        """Test that empty description raises error."""
        server = MCPServer()

        with pytest.raises(ValueError, match="description cannot be empty"):
            server.register_tool(
                name="valid_name",
                description="",
                schema={"type": "object"},
            )

    def test_register_tool_invalid_schema_raises_error(self):
        """Test that non-dict schema raises error."""
        server = MCPServer()

        with pytest.raises(ValueError, match="Schema must be a dictionary"):
            server.register_tool(
                name="valid_name",
                description="Valid description",
                schema="not a dict",
            )

    def test_register_tool_complex_schema(self):
        """Test registering tool with complex nested schema."""
        server = MCPServer()
        schema = {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "filters": {
                    "type": "object",
                    "properties": {
                        "source": {"type": "string"},
                        "date_range": {
                            "type": "object",
                            "properties": {
                                "start": {"type": "string"},
                                "end": {"type": "string"},
                            },
                        },
                    },
                },
                "limit": {"type": "integer", "minimum": 1, "maximum": 100},
            },
            "required": ["query"],
        }

        server.register_tool(
            name="advanced_search",
            description="Advanced search with filters",
            schema=schema,
        )

        assert "advanced_search" in server.tools
        assert server.tools["advanced_search"].input_schema == schema


class TestToolHandlers:
    """Test tool handler management."""

    @pytest.mark.asyncio
    async def test_add_tool_handler_basic(self):
        """Test adding basic async handler."""
        server = MCPServer()

        server.register_tool(
            name="test_tool",
            description="Test tool",
            schema={"type": "object"},
        )

        async def test_handler():
            return {"result": "success"}

        server.add_tool_handler("test_tool", test_handler)

        assert "test_tool" in server._handlers
        assert server._handlers["test_tool"] == test_handler

    @pytest.mark.asyncio
    async def test_add_handler_for_unregistered_tool_raises_error(self):
        """Test that adding handler for unregistered tool raises error."""
        server = MCPServer()

        async def test_handler():
            return {}

        with pytest.raises(ValueError, match="not registered"):
            server.add_tool_handler("unregistered_tool", test_handler)

    @pytest.mark.asyncio
    async def test_add_non_callable_handler_raises_error(self):
        """Test that non-callable handler raises error."""
        server = MCPServer()

        server.register_tool(
            name="test_tool",
            description="Test tool",
            schema={"type": "object"},
        )

        with pytest.raises(ValueError, match="must be callable"):
            server.add_tool_handler("test_tool", "not a callable")

    @pytest.mark.asyncio
    async def test_add_handler_with_parameters(self):
        """Test adding handler with parameters."""
        server = MCPServer()

        schema = {
            "type": "object",
            "properties": {"text": {"type": "string"}, "model": {"type": "string"}},
            "required": ["text"],
        }

        server.register_tool(
            name="embed",
            description="Generate embedding",
            schema=schema,
        )

        async def embed_handler(text: str, model: str = "default"):
            return {"text": text, "model": model}

        server.add_tool_handler("embed", embed_handler)

        assert "embed" in server._handlers

    @pytest.mark.asyncio
    async def test_add_multiple_handlers(self):
        """Test adding handlers for multiple tools."""
        server = MCPServer()

        for i in range(3):
            server.register_tool(
                name=f"tool{i}",
                description=f"Tool {i}",
                schema={"type": "object"},
            )

            async def handler(index=i):
                return {"index": index}

            server.add_tool_handler(f"tool{i}", handler)

        assert len(server._handlers) == 3


class TestServerLifecycle:
    """Test server lifecycle management."""

    @pytest.mark.asyncio
    async def test_run_server(self):
        """Test starting the server."""
        server = MCPServer()

        await server.run()

        assert server.is_running

    @pytest.mark.asyncio
    async def test_run_already_running_raises_error(self):
        """Test that running already-running server raises error."""
        server = MCPServer()

        await server.run()

        with pytest.raises(RuntimeError, match="already running"):
            await server.run()

    @pytest.mark.asyncio
    async def test_shutdown_server(self):
        """Test shutting down the server."""
        server = MCPServer()

        await server.run()
        assert server.is_running

        await server.shutdown()
        assert not server.is_running

    @pytest.mark.asyncio
    async def test_shutdown_not_running_server(self):
        """Test shutting down server that is not running."""
        server = MCPServer()

        # Should not raise error
        await server.shutdown()
        assert not server.is_running

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test server as async context manager."""
        server = MCPServer()

        async with server:
            assert server.is_running

        assert not server.is_running

    @pytest.mark.asyncio
    async def test_is_running_property(self):
        """Test is_running property accuracy."""
        server = MCPServer()

        assert not server.is_running

        await server.run()
        assert server.is_running

        await server.shutdown()
        assert not server.is_running


class TestServerProperties:
    """Test server property methods."""

    def test_tools_property_returns_copy(self):
        """Test that tools property returns a copy."""
        server = MCPServer()

        server.register_tool(
            name="test",
            description="Test",
            schema={"type": "object"},
        )

        tools1 = server.tools
        tools2 = server.tools

        # Should be equal but not the same object
        assert tools1 == tools2
        assert tools1 is not tools2

    def test_tools_property_immutable(self):
        """Test that modifying returned tools doesn't affect server."""
        server = MCPServer()

        server.register_tool(
            name="test",
            description="Test",
            schema={"type": "object"},
        )

        tools = server.tools
        tools["new_tool"] = ToolDefinition(
            name="new",
            description="New",
            input_schema={},
        )

        # Original server tools should be unchanged
        assert "new_tool" not in server.tools
        assert len(server.tools) == 1


class TestIntegration:
    """Integration tests for complete workflows."""

    @pytest.mark.asyncio
    async def test_full_workflow(self):
        """Test complete server workflow."""
        server = MCPServer(name="integration-test")

        # Register tool
        schema = {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        }
        server.register_tool(
            name="search",
            description="Search documents",
            schema=schema,
        )

        # Add handler
        async def search_handler(query: str):
            return {"results": [f"Result for: {query}"]}

        server.add_tool_handler("search", search_handler)

        # Run server
        await server.run()
        assert server.is_running
        assert "search" in server.tools
        assert "search" in server._handlers

        # Shutdown
        await server.shutdown()
        assert not server.is_running

    @pytest.mark.asyncio
    async def test_multiple_tools_workflow(self):
        """Test workflow with multiple tools."""
        server = MCPServer()

        # Register multiple tools
        tools_config = [
            (
                "embed",
                "Generate embeddings",
                {"properties": {"text": {"type": "string"}}},
            ),
            (
                "search",
                "Search documents",
                {"properties": {"query": {"type": "string"}}},
            ),
            ("index", "Index documents", {"properties": {"path": {"type": "string"}}}),
        ]

        for name, desc, props in tools_config:
            schema = {"type": "object", **props}
            server.register_tool(name, desc, schema)

            async def handler(name=name, **kwargs):
                return {"tool": name, "params": kwargs}

            server.add_tool_handler(name, handler)

        assert len(server.tools) == 3
        assert len(server._handlers) == 3

        await server.run()
        assert server.is_running

        await server.shutdown()
        assert not server.is_running
