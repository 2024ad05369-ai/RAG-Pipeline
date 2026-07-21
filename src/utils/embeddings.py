"""
It is a utility module for loading and using sentence embeddings in the RAG pipeline.

Author:
    Mrityunjay Dubey, Hari Sharma

Description:
    This module provides a singleton wrapper for loading the locally stored
    sentence-transformer model and generating embeddings for text inputs.
"""

from sentence_transformers import SentenceTransformer
from torch.fft import Tensor

from config.config import EMBEDDINGS_DIRECTORY


class SentenceEmbedding:
    """
    Singleton wrapper around
    """
    _sentence_transformer = None

    @classmethod
    def get_Sentence_Transformer(cls): 
        """
        Returns a shared embedding instance.
        """
        # 1. Point directly to the folder containing the saved files
        # 2. Load the model directly from the disk path
        # local_files_only=True ensures the script fails immediately 
        # if files are missing, rather than attempting a web download
        if cls._sentence_transformer is None:
            cls._sentence_transformer = SentenceTransformer(EMBEDDINGS_DIRECTORY.as_posix(), local_files_only=True)

        return cls._sentence_transformer

    @classmethod
    def get_SentenceEmbedding(cls, sentences: list[str]) -> Tensor: 
        """
        Generate sentence embeddings for the given list of sentences.
        """

        return cls.get_Sentence_Transformer().encode(sentences) 