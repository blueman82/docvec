"""Tests for MCP server entry point.

This module tests __main__.py functionality including:
- CLI argument parsing with defaults
- Component initialization in dependency order
- Signal handling for graceful shutdown
- Logging configuration to stderr
- Tool handler functionality
"""

import argparse
import logging
import sys
from unittest.mock import AsyncMock, Mock, patch

import pytest

from docvec.__main__ import (
    initialize_components,
    parse_arguments,
    setup_logging,
)


class TestArgumentParsing:
    """Test CLI argument parsing."""

    def test_parse_arguments_defaults(self):
        """Test that default arguments are set correctly."""
        with patch("sys.argv", ["vector-mcp"]):
            args = parse_arguments()

            # Ollama defaults
            assert args.host == "http://localhost:11434"
            assert args.model == "nomic-embed-text"
            assert args.timeout == 30

            # ChromaDB defaults
            assert args.db_path == "./chroma_db"
            assert args.collection == "documents"

            # Indexing defaults
            assert args.chunk_size == 512
            assert args.batch_size == 32
            assert args.max_tokens == 512

            # Logging defaults
            assert args.log_level == "INFO"

    def test_parse_arguments_custom_values(self):
        """Test parsing custom argument values."""
        with patch(
            "sys.argv",
            [
                "vector-mcp",
                "--host",
                "http://custom:8080",
                "--model",
                "custom-model",
                "--timeout",
                "60",
                "--db-path",
                "/custom/path",
                "--collection",
                "custom_collection",
                "--chunk-size",
                "1024",
                "--batch-size",
                "64",
                "--max-tokens",
                "256",
                "--log-level",
                "DEBUG",
            ],
        ):
            args = parse_arguments()

            assert args.host == "http://custom:8080"
            assert args.model == "custom-model"
            assert args.timeout == 60
            assert args.db_path == "/custom/path"
            assert args.collection == "custom_collection"
            assert args.chunk_size == 1024
            assert args.batch_size == 64
            assert args.max_tokens == 256
            assert args.log_level == "DEBUG"

    def test_parse_arguments_log_level_choices(self):
        """Test that only valid log levels are accepted."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

        for level in valid_levels:
            with patch("sys.argv", ["vector-mcp", "--log-level", level]):
                args = parse_arguments()
                assert args.log_level == level

    def test_parse_arguments_invalid_log_level_raises_error(self):
        """Test that invalid log level raises SystemExit."""
        with patch("sys.argv", ["vector-mcp", "--log-level", "INVALID"]):
            with pytest.raises(SystemExit):
                parse_arguments()


class TestLoggingSetup:
    """Test logging configuration."""

    def test_setup_logging_configures_stderr(self):
        """Test that logging is configured to write to stderr."""
        with patch("logging.basicConfig") as mock_config:
            setup_logging("INFO")

            # Verify basicConfig was called
            mock_config.assert_called_once()

            # Verify it configured stderr
            call_kwargs = mock_config.call_args[1]
            assert call_kwargs["stream"] == sys.stderr
            assert call_kwargs["level"] == logging.INFO

    def test_setup_logging_levels(self):
        """Test that different log levels are set correctly."""
        levels = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }

        for level_name, level_value in levels.items():
            with patch("logging.basicConfig") as mock_config:
                setup_logging(level_name)

                call_kwargs = mock_config.call_args[1]
                assert call_kwargs["level"] == level_value

    def test_setup_logging_invalid_level_defaults_to_info(self):
        """Test that invalid log level defaults to INFO."""
        with patch("logging.basicConfig") as mock_config:
            setup_logging("INVALID_LEVEL")

            call_kwargs = mock_config.call_args[1]
            assert call_kwargs["level"] == logging.INFO


class TestComponentInitialization:
    """Test component initialization."""

    @patch("docvec.__main__.QueryTools")
    @patch("docvec.__main__.IndexingTools")
    @patch("docvec.__main__.BatchProcessor")
    @patch("docvec.__main__.Indexer")
    @patch("docvec.__main__.DocumentHasher")
    @patch("docvec.__main__.ChromaStore")
    @patch("docvec.__main__.OllamaClient")
    def test_initialize_components_dependency_order(
        self,
        mock_ollama,
        mock_chroma,
        mock_hasher,
        mock_indexer,
        mock_batch,
        mock_indexing_tools,
        mock_query_tools,
    ):
        """Test that components are initialized in correct dependency order."""
        # Setup mocks
        mock_ollama_instance = Mock()
        mock_ollama_instance.ensure_model.return_value = True
        mock_ollama.return_value = mock_ollama_instance

        mock_chroma_instance = Mock()
        mock_chroma.return_value = mock_chroma_instance

        mock_hasher_instance = Mock()
        mock_hasher.return_value = mock_hasher_instance

        mock_indexer_instance = Mock()
        mock_indexer.return_value = mock_indexer_instance

        mock_batch_instance = Mock()
        mock_batch.return_value = mock_batch_instance

        mock_indexing_instance = Mock()
        mock_indexing_tools.return_value = mock_indexing_instance

        mock_query_instance = Mock()
        mock_query_tools.return_value = mock_query_instance

        # Create args
        args = argparse.Namespace(
            host="http://localhost:11434",
            model="nomic-embed-text",
            timeout=30,
            db_path="./test_db",
            collection="test_collection",
            chunk_size=512,
            batch_size=32,
            max_tokens=512,
        )

        # Initialize components
        components = initialize_components(args)

        # Verify all components were created
        assert "embedder" in components
        assert "storage" in components
        assert "hasher" in components
        assert "indexer" in components
        assert "batch_processor" in components
        assert "indexing_tools" in components
        assert "query_tools" in components

        # Verify OllamaClient was initialized correctly
        mock_ollama.assert_called_once_with(
            host="http://localhost:11434",
            model="nomic-embed-text",
            timeout=30,
        )

        # Verify ChromaStore was initialized correctly
        mock_chroma.assert_called_once()
        call_kwargs = mock_chroma.call_args[1]
        assert call_kwargs["collection_name"] == "test_collection"

        # Verify Indexer was initialized with correct dependencies
        mock_indexer.assert_called_once_with(
            embedder=mock_ollama_instance,
            storage=mock_chroma_instance,
            chunk_size=512,
            batch_size=32,
            max_tokens=512,
        )

        # Verify BatchProcessor dependencies
        mock_batch.assert_called_once_with(
            indexer=mock_indexer_instance,
            hasher=mock_hasher_instance,
            storage=mock_chroma_instance,
        )

        # Verify IndexingTools dependencies
        mock_indexing_tools.assert_called_once_with(
            batch_processor=mock_batch_instance,
            indexer=mock_indexer_instance,
        )

        # Verify QueryTools dependencies
        mock_query_tools.assert_called_once_with(
            embedder=mock_ollama_instance,
            storage=mock_chroma_instance,
        )

    @patch("docvec.__main__.OllamaClient")
    def test_initialize_components_ensure_model_failure_raises(self, mock_ollama):
        """Test that ensure_model failure raises RuntimeError."""
        # Setup mock to fail ensure_model (model not available and can't pull)
        mock_ollama_instance = Mock()
        mock_ollama_instance.ensure_model.return_value = False
        mock_ollama.return_value = mock_ollama_instance

        args = argparse.Namespace(
            host="http://localhost:11434",
            model="nonexistent-model",
            timeout=30,
            db_path="./test_db",
            collection="test_collection",
            chunk_size=512,
            batch_size=32,
            max_tokens=512,
        )

        # Should raise RuntimeError when model can't be ensured
        with pytest.raises(RuntimeError, match="could not be loaded or pulled"):
            initialize_components(args)

    @patch("docvec.__main__.OllamaClient")
    def test_initialize_components_failure_raises_exception(self, mock_ollama):
        """Test that initialization failure raises exception."""
        # Make OllamaClient raise an exception
        mock_ollama.side_effect = Exception("Initialization failed")

        args = argparse.Namespace(
            host="http://localhost:11434",
            model="nomic-embed-text",
            timeout=30,
            db_path="./test_db",
            collection="test_collection",
            chunk_size=512,
            batch_size=32,
            max_tokens=512,
        )

        with pytest.raises(Exception, match="Initialization failed"):
            initialize_components(args)


class TestToolHandlers:
    """Test MCP tool handler functions."""

    @pytest.mark.asyncio
    async def test_index_file_handler_calls_indexing_tools(self):
        """Test that index_file handler delegates to IndexingTools."""
        from docvec import __main__

        # Mock the indexing_tools global
        mock_tools = AsyncMock()
        mock_tools.index_file.return_value = {
            "success": True,
            "data": {"file": "test.txt", "chunks": 5},
        }

        __main__.indexing_tools = mock_tools

        result = await __main__.index_file("test.txt")

        mock_tools.index_file.assert_called_once_with("test.txt")
        assert result["success"] is True
        assert result["data"]["chunks"] == 5

    @pytest.mark.asyncio
    async def test_index_file_handler_not_initialized(self):
        """Test that index_file returns error when not initialized."""
        from docvec import __main__

        __main__.indexing_tools = None

        result = await __main__.index_file("test.txt")

        assert result["success"] is False
        assert "not initialized" in result["error"]

    @pytest.mark.asyncio
    async def test_index_directory_handler_calls_indexing_tools(self):
        """Test that index_directory handler delegates to IndexingTools."""
        from docvec import __main__

        mock_tools = AsyncMock()
        mock_tools.index_directory.return_value = {
            "success": True,
            "data": {"directory": "/path", "new_documents": 10},
        }

        __main__.indexing_tools = mock_tools

        result = await __main__.index_directory("/path", recursive=True)

        mock_tools.index_directory.assert_called_once_with("/path", True)
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_search_handler_calls_query_tools(self):
        """Test that search handler delegates to QueryTools."""
        from docvec import __main__

        mock_tools = AsyncMock()
        mock_tools.search.return_value = {
            "results": [],
            "total_results": 0,
        }

        __main__.query_tools = mock_tools

        result = await __main__.search("test query", n_results=5)

        mock_tools.search.assert_called_once_with("test query", 5)
        assert "results" in result

    @pytest.mark.asyncio
    async def test_search_handler_handles_exception(self):
        """Test that search handler catches and formats exceptions."""
        from docvec import __main__

        mock_tools = AsyncMock()
        mock_tools.search.side_effect = Exception("Search failed")

        __main__.query_tools = mock_tools

        result = await __main__.search("test query")

        assert "error" in result
        assert "Search failed" in result["error"]

    @pytest.mark.asyncio
    async def test_search_with_filters_handler(self):
        """Test search_with_filters handler."""
        from docvec import __main__

        mock_tools = AsyncMock()
        mock_tools.search_with_filters.return_value = {
            "results": [],
            "total_results": 0,
            "filters": {"source_file": "test.md"},
        }

        __main__.query_tools = mock_tools

        filters = {"source_file": "test.md"}
        result = await __main__.search_with_filters("query", filters, n_results=3)

        mock_tools.search_with_filters.assert_called_once_with("query", filters, 3)
        assert result["filters"] == filters

    @pytest.mark.asyncio
    async def test_search_with_budget_handler(self):
        """Test search_with_budget handler."""
        from docvec import __main__

        mock_tools = AsyncMock()
        mock_tools.search_with_budget.return_value = {
            "results": [],
            "total_tokens": 0,
            "max_tokens": 1000,
        }

        __main__.query_tools = mock_tools

        result = await __main__.search_with_budget("query", max_tokens=1000)

        mock_tools.search_with_budget.assert_called_once_with("query", 1000)
        assert result["max_tokens"] == 1000


class TestSignalHandling:
    """Test signal handling for graceful shutdown."""

    @patch("sys.exit")
    def test_handle_shutdown_calls_exit(self, mock_exit):
        """Test that handle_shutdown calls sys.exit."""
        from docvec.__main__ import handle_shutdown

        handle_shutdown(2, None)

        mock_exit.assert_called_once_with(0)


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_main_initialization_flow(self):
        """Test main function initialization flow."""
        from docvec import __main__

        # Mock parse_arguments
        mock_args = argparse.Namespace(
            host="http://localhost:11434",
            model="nomic-embed-text",
            timeout=30,
            db_path="./test_db",
            collection="test_collection",
            chunk_size=512,
            batch_size=32,
            max_tokens=512,
            log_level="INFO",
        )

        with (
            patch("docvec.__main__.parse_arguments", return_value=mock_args),
            patch("docvec.__main__.setup_logging"),
            patch("docvec.__main__.initialize_components") as mock_init,
            patch("docvec.__main__.mcp") as mock_mcp,
            patch("signal.signal"),
        ):

            # Setup mock components
            mock_indexing = Mock()
            mock_management = Mock()
            mock_query = Mock()
            mock_init.return_value = {
                "indexing_tools": mock_indexing,
                "management_tools": mock_management,
                "query_tools": mock_query,
            }

            # Mock mcp.run to avoid blocking
            mock_mcp.run = Mock()

            # Run main
            __main__.main()

            # Verify initialization was called
            mock_init.assert_called_once()

            # Verify globals were set
            assert __main__.indexing_tools == mock_indexing
            assert __main__.management_tools == mock_management
            assert __main__.query_tools == mock_query

            # Verify mcp.run was called with stdio transport
            mock_mcp.run.assert_called_once_with(transport="stdio")

    def test_main_handles_initialization_failure(self):
        """Test that main handles initialization failures gracefully."""
        from docvec import __main__

        mock_args = argparse.Namespace(
            host="http://localhost:11434",
            model="nomic-embed-text",
            timeout=30,
            db_path="./test_db",
            collection="test_collection",
            chunk_size=512,
            batch_size=32,
            max_tokens=512,
            log_level="INFO",
        )

        with (
            patch("docvec.__main__.parse_arguments", return_value=mock_args),
            patch("docvec.__main__.setup_logging"),
            patch("docvec.__main__.initialize_components") as mock_init,
            patch("sys.exit") as mock_exit,
        ):

            # Make initialization fail
            mock_init.side_effect = Exception("Init failed")

            # Run main
            __main__.main()

            # Verify exit was called with error code
            mock_exit.assert_called_once_with(1)
