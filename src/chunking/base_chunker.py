"""
Base Chunker

Author:
    Mrityunjay Dubey

Description:
    Defines the abstract interface for all chunking strategies.

    Every chunking strategy in the RAG pipeline must inherit from
    BaseChunker and implement the create_chunks() method.
"""

from abc import ABC, abstractmethod

from src.models.document import Document
from src.models.chunk import Chunk


class BaseChunker(ABC):
    """
    Abstract base class for all document chunkers.
    """

    @abstractmethod
    def create_chunks(
        self,
        documents: list[Document]
    ) -> list[Chunk]:
        """
        Generate chunks from the supplied documents.

        Parameters
        ----------
        documents : list[Document]

        Returns
        -------
        list[Chunk]
        """

        pass