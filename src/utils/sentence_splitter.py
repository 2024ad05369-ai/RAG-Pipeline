"""
Sentence Splitter Utility.
"""

import nltk
from nltk.tokenize import sent_tokenize


class SentenceSplitter:
    """
    Utility class for sentence segmentation.
    """

    @staticmethod
    def split(text: str) -> list[str]:
        """
        Split text into sentences.

        Parameters
        ----------
        text : str

        Returns
        -------
        list[str]
        """

        return sent_tokenize(text)