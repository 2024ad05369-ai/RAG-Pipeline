"""
Downloads the required NLTK resources.
Run once during project setup.
"""

import nltk

def main() -> None:
    nltk.download("punkt")


if __name__ == "__main__":
    main()