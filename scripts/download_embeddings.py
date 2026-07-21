"""
Script for downloading the Sentence Transformer model "all-MiniLM-L6-v2".

Author:
    Mrityunjay Dubey, Hari Sharma

Description:
    This script downloads the specified sentence-transformer model once and stores
    it in the project's local models directory so it can be reused without needing
    to download it repeatedly.
"""
from pathlib import Path
from sentence_transformers import SentenceTransformer

EMBEDDINGS_MODEL_NAME = "all-MiniLM-L6-v2"

EMBEDDINGS_DIRECTORY = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "embeddings"
    / EMBEDDINGS_MODEL_NAME
)

def main():

    print("Downloading sentence transformer...")

    # Download from the web
    model = SentenceTransformer(EMBEDDINGS_MODEL_NAME)

    EMBEDDINGS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save the model to the specified directory
    model.save(EMBEDDINGS_DIRECTORY)

    print("Sentence transformer downloaded successfully.")
    print(f"Model successfully isolated and saved to: {EMBEDDINGS_DIRECTORY}")

if __name__ == "__main__":
    main()