"""Tests for AST-based code chunker."""

import pytest

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
        content = """def func1():
    return 1

def func2():
    return 2

def func3():
    return 3
"""
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
        content = """import os
import sys
from typing import Optional

def func():
    return os.path.join("a", "b")
"""
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
        content = """def standalone():
    return 1

class MyClass:
    def method(self):
        pass

async def async_func():
    pass
"""
        chunks = chunker.chunk(content, "test.py")

        assert len(chunks) == 3
        assert chunks[0].metadata["type"] == "function"
        assert chunks[1].metadata["type"] == "class"
        assert chunks[1].metadata["class_name"] == "MyClass"
        assert chunks[2].metadata["type"] == "async_function"

    def test_chunk_start_line_metadata(self):
        """Test that start_line metadata is correct."""
        chunker = CodeChunker()
        content = """import os

def func1():
    pass

def func2():
    pass
"""
        chunks = chunker.chunk(content, "test.py")

        # First chunk is imports (line 1)
        # Second chunk is func1 (line 3)
        # Third chunk is func2 (line 6)
        assert chunks[1].metadata["start_line"] == 3
        assert chunks[2].metadata["start_line"] == 6

    def test_fallback_chunking_syntax_error(self):
        """Test that syntax errors trigger fallback chunking."""
        chunker = CodeChunker(chunk_size=3)
        content = """def broken(
    this is not valid python
    syntax error here
"""
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
        content = """def func1():
    pass

def func2():
    pass
"""
        source_file = "/path/to/module.py"
        chunks = chunker.chunk(content, source_file)

        for chunk in chunks:
            assert chunk.source_file == source_file

    def test_chunk_sequential_indexing(self):
        """Test that chunks have sequential zero-based indexing."""
        chunker = CodeChunker()
        content = """import os

def func1():
    pass

def func2():
    pass

class MyClass:
    pass
"""
        chunks = chunker.chunk(content, "test.py")

        assert len(chunks) == 4
        for idx, chunk in enumerate(chunks):
            assert chunk.chunk_index == idx

    def test_chunk_empty_file_with_only_comments(self):
        """Test chunking file with only comments."""
        chunker = CodeChunker()
        content = """# This is a comment
# Another comment
"""
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
        content = """from typing import (
    Optional,
    List,
    Dict,
)

def func():
    pass
"""
        chunks = chunker.chunk(content, "test.py")

        assert len(chunks) == 2
        assert chunks[0].metadata["type"] == "imports"
        # All import lines should be captured
        assert "from typing import" in chunks[0].content

    def test_chunk_decorator_with_function(self):
        """Test that decorators are included with function."""
        chunker = CodeChunker()
        content = """@decorator
@another_decorator(arg=True)
def decorated_func():
    pass
"""
        chunks = chunker.chunk(content, "test.py")

        assert len(chunks) == 1
        assert chunks[0].metadata["type"] == "function"
        assert "@decorator" in chunks[0].content
        assert "@another_decorator" in chunks[0].content

    def test_chunk_nested_functions_stay_together(self):
        """Test that nested functions stay with parent."""
        chunker = CodeChunker()
        content = """def outer():
    def inner():
        return 42
    return inner()
"""
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
        content = """import os
import sys
from pathlib import Path
"""
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

    def test_max_tokens_parameter_default(self):
        """Test that max_tokens defaults to 512."""
        chunker = CodeChunker()
        assert chunker.max_tokens == 512

    def test_max_tokens_parameter_custom(self):
        """Test custom max_tokens parameter."""
        chunker = CodeChunker(max_tokens=256)
        assert chunker.max_tokens == 256

    def test_split_oversized_function(self):
        """Test that oversized functions are split."""
        # Use very small max_tokens to force splitting
        chunker = CodeChunker(max_tokens=30)  # ~120 chars max

        # Create a function that significantly exceeds the limit
        content = '''def large_func():
    """A function with lots of code that needs to be chunked."""
    variable_one = "This is a long string value"
    variable_two = "Another long string value here"
    variable_three = "Yet another string for testing"
    calculation_result = variable_one + variable_two
    final_result = calculation_result + variable_three
    return final_result
'''
        chunks = chunker.chunk(content, "test.py")

        # Should be split into multiple chunks
        assert len(chunks) > 1
        # All chunks should have split_part in metadata
        for chunk in chunks:
            assert "split_part" in chunk.metadata
        # Chunk indices should be sequential
        for idx, chunk in enumerate(chunks):
            assert chunk.chunk_index == idx

    def test_split_oversized_class_by_methods(self):
        """Test that oversized classes are split by methods first."""
        # Use small max_tokens to force splitting
        chunker = CodeChunker(max_tokens=50)  # ~200 chars max

        content = '''class LargeClass:
    """A class with many methods that need to be split across chunks."""

    def method_one(self):
        """First method with a longer docstring."""
        result = "method one result"
        return result

    def method_two(self):
        """Second method with a longer docstring."""
        result = "method two result"
        return result

    def method_three(self):
        """Third method with a longer docstring."""
        result = "method three result"
        return result

    def method_four(self):
        """Fourth method with a longer docstring."""
        result = "method four result"
        return result
'''
        chunks = chunker.chunk(content, "test.py")

        # Should be split - class is larger than ~200 chars
        assert len(chunks) > 1
        # First chunk should still have class_name metadata
        assert any("class_name" in chunk.metadata for chunk in chunks)

    def test_split_preserves_chunk_index_sequence(self):
        """Test that chunk indices remain sequential after splitting."""
        chunker = CodeChunker(max_tokens=50)

        content = '''import os

def func1():
    """Short function."""
    return 1

def large_func():
    """A function that will be split."""
    x = 1
    y = 2
    z = 3
    a = x + y + z
    b = a * 2
    c = b * 3
    return c

def func2():
    """Another short function."""
    return 2
'''
        chunks = chunker.chunk(content, "test.py")

        # Verify sequential indexing
        for idx, chunk in enumerate(chunks):
            assert chunk.chunk_index == idx

    def test_small_chunks_not_split(self):
        """Test that chunks within limit are not split."""
        chunker = CodeChunker(max_tokens=512)  # Default, generous limit

        content = '''def small_func():
    """A small function."""
    return 42
'''
        chunks = chunker.chunk(content, "test.py")

        assert len(chunks) == 1
        # Should NOT have split_part metadata
        assert "split_part" not in chunks[0].metadata

    def test_split_metadata_preserved(self):
        """Test that original metadata is preserved after splitting."""
        chunker = CodeChunker(max_tokens=50)

        content = '''def large_func():
    """Function docstring."""
    line1 = "value1"
    line2 = "value2"
    line3 = "value3"
    line4 = "value4"
    return line1 + line2 + line3 + line4
'''
        chunks = chunker.chunk(content, "test.py")

        # All chunks should have type metadata preserved
        for chunk in chunks:
            assert chunk.metadata["type"] == "function"
            assert chunk.source_file == "test.py"

    def test_split_class_preserves_class_name(self):
        """Test that split class chunks preserve class_name metadata."""
        chunker = CodeChunker(max_tokens=40)  # ~160 chars max

        content = '''class BigClass:
    """A big class with many methods to ensure splitting."""

    def __init__(self):
        self.value = 42
        self.other = "test string value"
        self.third = "another value"

    def method_one(self):
        """Return the main value."""
        return self.value

    def method_two(self):
        """Return the other value."""
        return self.other

    def method_three(self):
        """Do some calculation."""
        return self.value + len(self.other)
'''
        chunks = chunker.chunk(content, "test.py")

        # Should be split
        assert len(chunks) > 1
        # All class-derived chunks should preserve class_name
        class_chunks = [c for c in chunks if c.metadata.get("class_name")]
        assert len(class_chunks) > 0
        for chunk in class_chunks:
            assert chunk.metadata["class_name"] == "BigClass"
