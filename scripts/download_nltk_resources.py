"""
Script for downloading the NLTK resource "punkt" for text processing.

Author:
    Mrityunjay Dubey, Hari Sharma

Description:
    This script downloads the required NLTK resource once during project setup
    so the text processing pipeline can use tokenization without repeated setup.
"""

import nltk

def main() -> None:
    nltk.download("punkt")


if __name__ == "__main__":
    main()