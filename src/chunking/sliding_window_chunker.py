"""
Sliding Window Chunker

Author:
    Mrityunjay Dubey, Hari Sharma

Description
-----------
Creates overlapping chunks from documents using a fixed-size
sliding window approach.

Each consecutive chunk shares a configurable number of tokens
with the previous chunk. This helps preserve context across
chunk boundaries and often improves retrieval quality in RAG
systems.
"""

from config.config import MAX_CHUNK_TOKENS, CHUNK_OVERLAP_TOKEN_COUNT
from config.logging_config import logger
from src.chunking.base_chunker import BaseChunker
from src.models.chunk import Chunk
from src.models.document import Document
from src.utils.tokenizer import Tokenizer


class SlidingWindowChunker(BaseChunker):
    """
    Generates overlapping chunks using a sliding window.
    """

    def __init__(self,chunk_size: int = MAX_CHUNK_TOKENS, overlap: int = CHUNK_OVERLAP_TOKEN_COUNT) -> None:
        """
        Initialize the sliding window chunker.

        Parameters
        ----------
        chunk_size : int
            Maximum number of tokens per chunk.

        overlap : int
            Number of overlapping tokens between
            consecutive chunks.
        """

        if chunk_size <= 0:
            raise ValueError(
                "Chunk size must be greater than zero."
            )

        if overlap < 0:
            raise ValueError(
                "Overlap cannot be negative."
            )

        if overlap >= chunk_size:
            raise ValueError(
                "Overlap must be smaller than chunk size."
            )

        self.chunk_size = chunk_size
        self.overlap = overlap

    def create_chunks(self, documents: list[Document]) -> list[Chunk]:
        """
        Generate overlapping chunks.

        Parameters
        ----------
        documents : list[Document]

        Returns
        -------
        list[Chunk]
        """

        logger.info(
            "Starting Sliding Window Chunking..."
        )

        chunks: list[Chunk] = []

        chunk_id = 1

        step = self.chunk_size - self.overlap

        for document in documents:
            token_ids = Tokenizer.tokenize(document.text)
            total_tokens = len(token_ids)
            for start in range(0, total_tokens, step):
                end = min(start + self.chunk_size, total_tokens)
                chunk_token_ids = token_ids[start:end]
                if not chunk_token_ids:
                    continue
                chunk_text = Tokenizer.detokenize(chunk_token_ids)
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        document_id=document.document_id,
                        document_name=document.file_name,
                        start_token=start,
                        end_token=end - 1,
                        text=chunk_text,
                        token_count=len(chunk_token_ids),
                        strategy="Sliding Window",
                    )
                )
                chunk_id += 1

                #
                # Stop once the last token of the document
                # has been included.
                #
                if end == total_tokens:
                    break

        logger.info("Sliding Window Chunking completed. Generated %d chunks.", len(chunks))
        return chunks