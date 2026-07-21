"""
It is a performance test script for comparing chunking strategies.

Author:
    Mrityunjay Dubey, Hari Sharma

Description:
    This test script evaluates fixed-size, sliding-window, and semantic chunking
    approaches on a selected document to compare chunking behavior.
"""

import re
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.chunking.fixed_chunker import FixedChunker
from src.chunking.sliding_window_chunker import SlidingWindowChunker
from src.chunking.semantic_chunker import SemanticChunker

from src.loaders.document_loader import DocumentLoader

# 1. Setup local model
MODEL_PATH = "./models/embeddings/all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_PATH, local_files_only=True)

# Step 1: Load documents
loader = DocumentLoader()
documents = loader.load_documents()

# Filters out odd numbers, keeping only even ones
my_selected_doc = [document for document in documents if document.file_name == "travel-expenses-policy-nov-251.txt"]

for document in my_selected_doc:
    print(f"Document: {document.file_name}, Words: {document.word_count}, Characters: {document.character_count}")

# --- STRATEGY 1: FIXED-SIZE (By Character for simplicity) ---
def fixed_chunking(documents):
    chunker = FixedChunker()
    chunks = chunker.create_chunks(documents)
    return chunks


# --- STRATEGY 2: SLIDING WINDOW ---
def sliding_window_chunking(documents):
    chunker = SlidingWindowChunker(chunk_size=200, overlap=20)
    chunks = chunker.create_chunks(documents)
    return chunks

# --- STRATEGY 3: SEMANTIC CHUNKING (Boundary Preserving) ---
def semantic_chunking(documents):
    chunker = SemanticChunker()
    chunks = chunker.create_chunks(documents)
    return chunks

# Run Analysis
results = {
    "Fixed-Size": fixed_chunking(my_selected_doc),
    "Sliding Window": sliding_window_chunking(my_selected_doc),
    "Semantic": semantic_chunking(my_selected_doc)
}

# Compile Metrics
analysis_data = []
for strategy, chunks in results.items():
    # Extract word counts for each chunk using a list comprehension
    word_counts = [len(chunk.text.split()) for chunk in chunks]

    # Calculate standard deviation and average using numpy
    std_dev_size = np.std(word_counts)
    average_words = np.mean(word_counts)

    # Track broken sentences
    broken_count = 0
    terminal_punctuation = (".", "!", "?", '"', "'")

    for chunk in chunks:
        clean_chunk = chunk.text.strip()
        if not clean_chunk:
            continue

        # If the chunk does NOT end with a standard sentence finisher, it is broken
        if not clean_chunk.endswith(terminal_punctuation):
            broken_count += 1

    # Calculate percentage
    broken_sentences_pct = (broken_count / len(chunks)) * 100

    analysis_data.append({
        "Strategy": strategy,
        "Total Chunks": len(chunks),
        "Avg Chunk Length (Chars)": average_words,
        "Std Dev Chunk Length": std_dev_size,
        "Broken Sentences (%)": broken_sentences_pct
    })

df = pd.DataFrame(analysis_data)
print(df.to_string(index=False))
