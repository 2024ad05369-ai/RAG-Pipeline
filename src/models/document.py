"""
Document Model

Represents a source document loaded into the RAG pipeline.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class Document:
    """
    Represents a source document.
    """

    # -------------------------------------------------------------------------
    # Document Identification
    # -------------------------------------------------------------------------

    document_id: int

    # -------------------------------------------------------------------------
    # File Information
    # -------------------------------------------------------------------------

    file_name: str
    file_path: Path

    # -------------------------------------------------------------------------
    # Document Content
    # -------------------------------------------------------------------------

    text: str

    # -------------------------------------------------------------------------
    # Document Statistics
    # -------------------------------------------------------------------------

    token_count: int

    @property
    def word_count(self) -> int:
        """
        Returns the number of words in the document.
        """
        return len(self.text.split())

    @property
    def character_count(self) -> int:
        """
        Returns the number of characters in the document.
        """
        return len(self.text)

    def __str__(self) -> str:
        return (
            f"Document("
            f"id={self.document_id}, "
            f"name='{self.file_name}', "
            f"tokens={self.token_count}, "
            f"words={self.word_count}"
            f")"
        )