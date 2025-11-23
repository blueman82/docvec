"""Document hash calculator for deduplication.

This module provides SHA-256 based hashing of file content to enable
efficient deduplication during batch indexing.
"""

import hashlib
from pathlib import Path


class DocumentHasher:
    """Calculates SHA-256 hashes for document deduplication.

    Uses file content hashing to identify duplicate documents before indexing.
    Consistent hash values enable efficient deduplication checks.

    Example:
        >>> hasher = DocumentHasher()
        >>> hash1 = hasher.hash_document(Path("file.txt"))
        >>> hash2 = hasher.hash_document(Path("file.txt"))
        >>> assert hash1 == hash2
    """

    def hash_document(self, file_path: Path) -> str:
        """Compute SHA-256 hash of file content.

        Reads entire file content and computes SHA-256 hash. Uses binary
        mode to handle all file types uniformly.

        Args:
            file_path: Path to file to hash

        Returns:
            Hexadecimal SHA-256 hash string (64 characters)

        Raises:
            FileNotFoundError: If file does not exist
            IOError: If file cannot be read

        Example:
            >>> hasher = DocumentHasher()
            >>> file_hash = hasher.hash_document(Path("document.txt"))
            >>> len(file_hash)
            64
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if not file_path.is_file():
            raise IOError(f"Path is not a file: {file_path}")

        try:
            # Read file in binary mode for consistent hashing
            content = file_path.read_bytes()

            # Compute SHA-256 hash
            hash_obj = hashlib.sha256(content)
            return hash_obj.hexdigest()

        except Exception as e:
            raise IOError(f"Failed to read file {file_path}: {e}") from e

    def hash_content(self, content: str) -> str:
        """Compute SHA-256 hash of string content.

        Useful for hashing in-memory content without file I/O.

        Args:
            content: String content to hash

        Returns:
            Hexadecimal SHA-256 hash string (64 characters)

        Example:
            >>> hasher = DocumentHasher()
            >>> hash_val = hasher.hash_content("Hello, world!")
            >>> len(hash_val)
            64
        """
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
