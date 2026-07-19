"""
Tokenizer Utility

Provides a singleton tokenizer instance for the
entire RAG pipeline.
"""

from transformers import AutoTokenizer

from config.config import TOKENIZER_DIRECTORY


class Tokenizer:
    """
    Singleton wrapper around Hugging Face tokenizer.
    """

    _tokenizer = None

    @classmethod
    def get_tokenizer(cls):
        """
        Returns a shared tokenizer instance.
        """

        if cls._tokenizer is None:

            cls._tokenizer = AutoTokenizer.from_pretrained(
                TOKENIZER_DIRECTORY,
                local_files_only=True
            )

        return cls._tokenizer

    @classmethod
    def tokenize(cls, text: str) -> list[int]:
        """
        Convert text into token IDs.
        """

        tokenizer = cls.get_tokenizer()

        return tokenizer.encode(
            text,
            add_special_tokens=False
        )

    @classmethod
    def detokenize(cls, token_ids: list[int]) -> str:
        """
        Convert token IDs back to text.
        """

        tokenizer = cls.get_tokenizer()

        return tokenizer.decode(
            token_ids,
            skip_special_tokens=True
        )

    @classmethod
    def count_tokens(cls, text: str) -> int:
        """
        Returns the number of tokens.
        """

        return len(cls.tokenize(text))