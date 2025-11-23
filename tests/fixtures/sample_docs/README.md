# Vector MCP Server Documentation

## Overview

The Vector MCP Server provides semantic search capabilities for document indexing and retrieval.

## Features

### Document Indexing

The server supports multiple document formats:
- Markdown files (.md)
- PDF documents (.pdf)
- Python code (.py)
- Plain text (.txt)

### Semantic Search

Powered by Ollama embeddings and ChromaDB vector storage:
- Fast similarity search
- Metadata filtering
- Token budget management

## Installation

Install dependencies using uv:

```bash
uv pip install -e .
```

## Usage

Start the MCP server:

```bash
python -m vector_mcp --host http://localhost:11434 --model nomic-embed-text
```

### Indexing Documents

Use the MCP tools to index files:
- `index_file`: Index a single document
- `index_directory`: Index all files in a directory

### Querying

Perform semantic searches:
- `search`: Basic semantic search
- `search_with_filters`: Filter by metadata
- `search_with_budget`: Limit results by token count

## Architecture

The system consists of several components:
1. **Chunkers**: Split documents into manageable pieces
2. **Embedder**: Generate vector embeddings using Ollama
3. **Storage**: Persist embeddings in ChromaDB
4. **MCP Server**: Expose functionality via Model Context Protocol

## Contributing

Contributions are welcome! Please follow the coding standards and include tests.
