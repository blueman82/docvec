#!/usr/bin/env python3
"""MCP server entry point for vector database document indexing.

This is the main entry point for running the DocVec server.
It imports and runs the main function from the docvec package.
"""

if __name__ == "__main__":
    from docvec.__main__ import main

    main()
