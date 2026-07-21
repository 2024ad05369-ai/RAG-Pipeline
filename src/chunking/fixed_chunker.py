"""
Fixed Size Chunker

Author:
    Mrityunjay Dubey, Hari Sharma

Description
-----------
Implements fixed-size token-based chunking.

Characteristics
---------------
- Fixed chunk size
- No overlap
- Deterministic chunk generation
- Uses Hugging Face tokenizer for accurate token counting
"""

from config.config import MAX_CHUNK_TOKENS
from config.logging_config import logger

from src.chunking.base_chunker import BaseChunker
from src.models.chunk import Chunk
from src.models.document import Document
from src.utils.tokenizer import Tokenizer


class FixedChunker(BaseChunker):
    """
    Creates fixed-size chunks without overlap.

    Each chunk contains at most MAX_CHUNK_TOKENS tokens.
    """

    def __init__(
        self,
        chunk_size: int = MAX_CHUNK_TOKENS,
    ):
        """
        Initialize the FixedChunker.

        Parameters
        ----------
        chunk_size : int
            Maximum number of tokens per chunk.
        """

        self.chunk_size = chunk_size

    def create_chunks(
        self,
        documents: list[Document],
    ) -> list[Chunk]:
        """
        Create fixed-size chunks from the supplied documents.

        Parameters
        ----------
        documents : list[Document]
            Documents to be chunked.

        Returns
        -------
        list[Chunk]
            Generated chunks.
        """

        logger.info("Starting Fixed Size Chunking...")

        chunks: list[Chunk] = []
        chunk_id = 1

        for document in documents:

            logger.info(
                "Processing document: %s",
                document.file_name,
            )

            token_ids = Tokenizer.tokenize(document.text)

            for start in range(0, len(token_ids), self.chunk_size):

                end = start + self.chunk_size

                chunk_token_ids = token_ids[start:end]

                if not chunk_token_ids:
                    continue

                chunk_text = Tokenizer.detokenize(
                    chunk_token_ids
                )

                chunk = Chunk(
                    chunk_id=chunk_id,
                    document_id=document.document_id,
                    document_name=document.file_name,
                    start_token=start,
                    end_token=min(end,len(token_ids)) - 1,
                    text=chunk_text,
                    token_count=len(chunk_token_ids),
                    strategy="Fixed",
                )

                chunks.append(chunk)

                chunk_id += 1

        logger.info(
            "Fixed chunking completed. Generated %d chunks.",
            len(chunks),
        )

        return chunks