"""
Semantic Chunker

Author:
    Mrityunjay Dubey, Hari Sharma

Description
-----------
Implements semantic chunking using sentence embeddings.

Characteristics
---------------
- Splits text based on semantic similarity
- Uses sentence-transformers for embeddings
- Detects topic shifts using cosine similarity threshold
"""

import re

import numpy as np
from sentence_transformers import SentenceTransformer
from config.config import MAX_CHUNK_TOKENS
from src.chunking.base_chunker import BaseChunker
from src.models.chunk import Chunk
from src.models.document import Document
from src.utils.embeddings import SentenceEmbedding

class SemanticChunker(BaseChunker):
    """
    Creates fixed-size chunks without overlap.

    Each chunk contains at most MAX_CHUNK_TOKENS tokens.
    """

    def __init__(self, chunk_size: int = MAX_CHUNK_TOKENS):
        """
        Initialize the FixedChunker.

        Parameters
        ----------
        chunk_size : int
            Maximum number of tokens per chunk.
        """

        self.chunk_size = chunk_size

    def create_chunks(self, documents: list[Document],) -> list[Chunk]:
        """
        Create semantic chunks from the supplied documents.

        Parameters
        ----------
        documents : list[Document]
            Documents to be chunked.

        Returns
        -------
        list[Chunk]
            Generated semantic chunks.
        """
        chunk_id = 1
        start = 0
        end = 0
        all_chunks = []
        for document in documents:
            text = document.text
            semantic_chunks = self.get_semantic_chunks(text)
            for chunk_text in semantic_chunks:
                start = start + end
                total_tokens = len(chunk_text.split())
                end = end + total_tokens
                chunk = Chunk(
                    chunk_id=chunk_id,
                    document_id=document.document_id,
                    document_name=document.file_name,
                    start_token=start,
                    end_token=end - 1,
                    text=chunk_text,
                    token_count=len(chunk_text.split()),
                    strategy="Semantic",
                )
                all_chunks.append(chunk)
                chunk_id += 1

        return all_chunks
    
    def cosine_similarity(self, v1, v2):
        """
        Compute the cosine similarity between two vectors.
        """
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    def get_semantic_chunks(self, text, percentile_threshold=70):
        """
        Create semantic chunks from the supplied text.

        Parameters
        ----------
        text : str
            Input text to be segmented into semantically coherent chunks.
        percentile_threshold : int
            Percentile used to determine the similarity threshold for splitting.

        Returns
        -------
        list[str]
            Generated semantic chunks.
        """

        # Split the input text into sentences for semantic analysis.
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return []

        embeddings = SentenceEmbedding.get_SentenceEmbedding(sentences)
        print(f"Embedding generated successfully! Vector shape: {embeddings.shape}")

        # Calculate cosine similarity between adjacent sentences.
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = self.cosine_similarity(embeddings[i], embeddings[i+1])
            similarities.append(sim)

        # Determine a threshold for identifying topic shifts.
        threshold = np.percentile(similarities, percentile_threshold)

        # Segment the text into chunks based on semantic similarity.
        chunks = []
        current_chunk = [sentences[0]]

        for i, sim in enumerate(similarities):
            if sim < threshold:
                # Topic shift detected; finalize the current chunk and start a new one.
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentences[i + 1]]
            else:
                current_chunk.append(sentences[i + 1])

        # Add the final chunk if any text remains.
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

