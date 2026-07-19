"""
Downloads the tokenizer once and stores it inside the project.
"""

from pathlib import Path

from transformers import AutoTokenizer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

SAVE_DIRECTORY = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "tokenizer"
    / "all-MiniLM-L6-v2"
)


def main():

    print("Downloading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    SAVE_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True
    )

    tokenizer.save_pretrained(
        SAVE_DIRECTORY
    )

    print()

    print("Tokenizer downloaded successfully.")

    print(f"Saved to:\n{SAVE_DIRECTORY}")


if __name__ == "__main__":
    main()