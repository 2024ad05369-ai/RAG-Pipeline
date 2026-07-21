"""
It is a simple test script for verifying sentence embeddings loading.

Author:
    Mrityunjay Dubey, Hari Sharma

Description:
    This script loads the local sentence-transformer model from disk and generates
    a sample embedding to verify the embedding pipeline.
"""

from pathlib import Path

from sentence_transformers import SentenceTransformer

# Define your exact target directory
EMBEDDINGS_MODEL_NAME = "all-MiniLM-L6-v2"

EMBEDDINGS_DIRECTORY = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "embeddings"
    / EMBEDDINGS_MODEL_NAME
)

# Point directly to the folder containing the saved files
fixed_location = EMBEDDINGS_DIRECTORY.as_posix()

# Load the model directly from the disk path
# local_files_only=True ensures the script fails immediately
# if files are missing, rather than attempting a web download
model = SentenceTransformer(fixed_location, local_files_only=True)

# Generate your embeddings normally
sentences = ["This model loaded completely from a fixed local path."]
embeddings = model.encode(sentences)

print(f"Embedding generated successfully! Vector shape: {embeddings.shape}")


