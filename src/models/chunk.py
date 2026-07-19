"""
Chunk Model

Author:
    Mrityunjay Dubey

Description
-----------
Represents a chunk generated from a source document.

Each chunk contains both the extracted text and the metadata
required for downstream stages of the RAG pipeline.
"""

from dataclasses import dataclass


@dataclass(slots=True)
class Chunk:
    """
    Represents a single chunk in the RAG pipeline.
    """

    # -------------------------------------------------------------------------
    # Chunk Identification
    # -------------------------------------------------------------------------

    chunk_id: int

    # -------------------------------------------------------------------------
    # Source Document Information
    # -------------------------------------------------------------------------

    document_id: int
    document_name: str

    # -------------------------------------------------------------------------
    # Chunk Position
    # -------------------------------------------------------------------------

    start_token: int
    end_token: int

    # -------------------------------------------------------------------------
    # Chunk Content
    # -------------------------------------------------------------------------

    text: str

    # -------------------------------------------------------------------------
    # Chunk Statistics
    # -------------------------------------------------------------------------

    token_count: int

    # -------------------------------------------------------------------------
    # Chunking Strategy
    # -------------------------------------------------------------------------

    strategy: str

    @property
    def word_count(self) -> int:
        """
        Returns the number of words in the chunk.
        """

        return len(self.text.split())

    @property
    def character_count(self) -> int:
        """
        Returns the number of characters.
        """

        return len(self.text)

    def __str__(self) -> str:

        return (
            f"Chunk("
            f"id={self.chunk_id}, "
            f"document='{self.document_name}', "
            f"tokens={self.token_count}, "
            f"range=[{self.start_token}:{self.end_token}], "
            f"strategy='{self.strategy}'"
            f")"
        )