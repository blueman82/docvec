"""Tests for AST-based code chunker."""

import pytest

from docvec.chunking.base import Chunk
from docvec.chunking.code_chunker import CodeChunker


class TestCodeChunker:
    """Test suite for CodeChunker."""

    def test_chunk_simple_function(self):
        """Test chunking file with single function."""
        chunker = CodeChunker()
        content = '''def hello():
    """Say hello."""
    print("Hello, world!")
'''
        chunks = chunker.chunk(content, "test.py")

        assert len(chunks) == 1
        assert chunks[0].content.strip() == content.strip()
        assert chunks[0].chunk_index == 0
        assert chunks[0].metadata["type"] == "function"

    def test_chunk_multiple_functions(self):
        """Test chunking file with multiple functions."""
        chunker = CodeChunker()
        content = '''def func1():
    return 1

def func2():
    return 2

def func3():
    return 3
'''
        chunks = chunker.chunk(content, "test.py")

        assert len(chunks) == 3
        assert "def func1():" in chunks[0].content
        assert "def func2():" in chunks[1].content
        assert "def func3():" in chunks[2].content

        for idx, chunk in enumerate(chunks):
            assert chunk.chunk_index == idx
            assert chunk.metadata["type"] == "function"

    def test_chunk_class_definition(self):
        """Test chunking file with class definition."""
        chunker = CodeChunker()
        content = '''class MyClass:
    """A simple class."""

    def __init__(self):
        self.value = 42

    def method(self):
        return self.value
'''
        chunks = chunker.chunk(content, "test.py")

        assert len(chunks) == 1
        assert "class MyClass:" in chunks[0].content
        assert chunks[0].metadata["type"] == "class"
        assert chunks[0].metadata["class_name"] == "MyClass"

    def test_chunk_with_imports(self):
        """Test that imports are preserved in first chunk."""
        chunker = CodeChunker()
        content = '''import os
import sys
from typing import Optional

def func():
    return os.path.join("a", "b")
'''
        chunks = chunker.chunk(content, "test.py")

        assert len(chunks) == 2
        assert chunks[0].metadata["type"] == "imports"
        assert "import os" in chunks[0].content
        assert "import sys" in chunks[0].content
        assert "from typing import Optional" in chunks[0].content

        assert chunks[1].metadata["type"] == "function"
        assert "def func():" in chunks[1].content

    def test_chunk_with_module_docstring(self):
        """Test that module docstring is included with imports."""
        chunker = CodeChunker()
        content = '''"""Module docstring."""
import os

def func():
    pass
'''
        chunks = chunker.chunk(content, "test.py")

        assert len(chunks) == 2
        assert chunks[0].metadata["type"] == "imports"
        assert chunks[0].metadata["has_docstring"] is True
        assert "Module docstring" in chunks[0].content
        assert "import os" in chunks[0].content

    def test_chunk_async_function(self):
        """Test chunking async function."""
        chunker = CodeChunker()
        content = '''async def async_func():
    """An async function."""
    await something()
    return 42
'''
        chunks = chunker.chunk(content, "test.py")

        assert len(chunks) == 1
        assert chunks[0].metadata["type"] == "async_function"
        assert "async def async_func():" in chunks[0].content

    def test_chunk_mixed_definitions(self):
        """Test chunking file with mixed functions and classes."""
        chunker = CodeChunker()
        content = '''def standalone():
    return 1

class MyClass:
    def method(self):
        pass

async def async_func():
    pass
'''
        chunks = chunker.chunk(content, "test.py")

        assert len(chunks) == 3
        assert chunks[0].metadata["type"] == "function"
        assert chunks[1].metadata["type"] == "class"
        assert chunks[1].metadata["class_name"] == "MyClass"
        assert chunks[2].metadata["type"] == "async_function"

    def test_chunk_start_line_metadata(self):
        """Test that start_line metadata is correct."""
        chunker = CodeChunker()
        content = '''import os

def func1():
    pass

def func2():
    pass
'''
        chunks = chunker.chunk(content, "test.py")

        # First chunk is imports (line 1)
        # Second chunk is func1 (line 3)
        # Third chunk is func2 (line 6)
        assert chunks[1].metadata["start_line"] == 3
        assert chunks[2].metadata["start_line"] == 6

    def test_fallback_chunking_syntax_error(self):
        """Test that syntax errors trigger fallback chunking."""
        chunker = CodeChunker(chunk_size=3)
        content = '''def broken(
    this is not valid python
    syntax error here
'''
        chunks = chunker.chunk(content, "test.py")

        # Should fall back to line-based chunking
        assert len(chunks) == 1
        assert chunks[0].metadata["type"] == "fallback"

    def test_fallback_chunking_line_limit(self):
        """Test fallback chunking respects chunk_size."""
        chunker = CodeChunker(chunk_size=2)

        # Force fallback by using invalid Python syntax
        content = "def broken(\n" + "invalid syntax here\n" * 10

        chunks = chunker.chunk(content, "test.txt")

        # Should create multiple chunks based on chunk_size
        assert len(chunks) > 1
        for chunk in chunks:
            assert chunk.metadata["type"] == "fallback"

    def test_chunk_empty_content_raises_error(self):
        """Test that empty content raises ValueError."""
        chunker = CodeChunker()
        with pytest.raises(ValueError, match="Content cannot be empty"):
            chunker.chunk("", "test.py")

    def test_chunk_whitespace_only_raises_error(self):
        """Test that whitespace-only content raises ValueError."""
        chunker = CodeChunker()
        with pytest.raises(ValueError, match="Content cannot be empty"):
            chunker.chunk("   \n\t  ", "test.py")

    def test_chunk_preserves_source_file(self):
        """Test that source file is preserved in all chunks."""
        chunker = CodeChunker()
        content = '''def func1():
    pass

def func2():
    pass
'''
        source_file = "/path/to/module.py"
        chunks = chunker.chunk(content, source_file)

        for chunk in chunks:
            assert chunk.source_file == source_file

    def test_chunk_sequential_indexing(self):
        """Test that chunks have sequential zero-based indexing."""
        chunker = CodeChunker()
        content = '''import os

def func1():
    pass

def func2():
    pass

class MyClass:
    pass
'''
        chunks = chunker.chunk(content, "test.py")

        assert len(chunks) == 4
        for idx, chunk in enumerate(chunks):
            assert chunk.chunk_index == idx

    def test_chunk_empty_file_with_only_comments(self):
        """Test chunking file with only comments."""
        chunker = CodeChunker()
        content = '''# This is a comment
# Another comment
'''
        chunks = chunker.chunk(content, "test.py")

        # Should create one chunk with the content
        assert len(chunks) == 1
        assert chunks[0].metadata["type"] == "other"

    def test_chunk_complex_class_with_methods(self):
        """Test chunking complex class keeps it together."""
        chunker = CodeChunker()
        content = '''class Complex:
    """A complex class."""

    def __init__(self, value):
        self.value = value

    def method1(self):
        return self.value

    def method2(self):
        return self.value * 2

    @property
    def prop(self):
        return self.value
'''
        chunks = chunker.chunk(content, "test.py")

        # Should keep entire class as one chunk
        assert len(chunks) == 1
        assert chunks[0].metadata["type"] == "class"
        assert chunks[0].metadata["class_name"] == "Complex"
        assert "def __init__" in chunks[0].content
        assert "def method1" in chunks[0].content
        assert "def method2" in chunks[0].content
        assert "@property" in chunks[0].content

    def test_chunk_with_multiline_imports(self):
        """Test chunking with multiline imports."""
        chunker = CodeChunker()
        content = '''from typing import (
    Optional,
    List,
    Dict,
)

def func():
    pass
'''
        chunks = chunker.chunk(content, "test.py")

        assert len(chunks) == 2
        assert chunks[0].metadata["type"] == "imports"
        # All import lines should be captured
        assert "from typing import" in chunks[0].content

    def test_chunk_decorator_with_function(self):
        """Test that decorators are included with function."""
        chunker = CodeChunker()
        content = '''@decorator
@another_decorator(arg=True)
def decorated_func():
    pass
'''
        chunks = chunker.chunk(content, "test.py")

        assert len(chunks) == 1
        assert chunks[0].metadata["type"] == "function"
        assert "@decorator" in chunks[0].content
        assert "@another_decorator" in chunks[0].content

    def test_chunk_nested_functions_stay_together(self):
        """Test that nested functions stay with parent."""
        chunker = CodeChunker()
        content = '''def outer():
    def inner():
        return 42
    return inner()
'''
        chunks = chunker.chunk(content, "test.py")

        # Outer function should include inner function
        assert len(chunks) == 1
        assert "def outer():" in chunks[0].content
        assert "def inner():" in chunks[0].content

    def test_custom_chunk_size(self):
        """Test that custom chunk_size is used for fallback."""
        chunker = CodeChunker(chunk_size=5)
        assert chunker.chunk_size == 5

    def test_chunk_only_imports(self):
        """Test file with only imports."""
        chunker = CodeChunker()
        content = '''import os
import sys
from pathlib import Path
'''
        chunks = chunker.chunk(content, "test.py")

        assert len(chunks) == 1
        assert chunks[0].metadata["type"] == "imports"

    def test_chunk_preserves_function_signature(self):
        """Test that full function signature is preserved."""
        chunker = CodeChunker()
        content = '''def complex_func(
    arg1: str,
    arg2: int = 42,
    *args,
    **kwargs
) -> Optional[str]:
    """Complex signature."""
    return arg1
'''
        chunks = chunker.chunk(content, "test.py")

        assert len(chunks) == 1
        full_content = chunks[0].content
        assert "arg1: str" in full_content
        assert "arg2: int = 42" in full_content
        assert "*args" in full_content
        assert "**kwargs" in full_content
        assert "-> Optional[str]" in full_content
