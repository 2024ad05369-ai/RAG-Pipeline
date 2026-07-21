"""
It is a test script for validating semantic chunking behavior.

Author:
    Mrityunjay Dubey, Hari Sharma

Description:
    This script loads documents, creates semantic chunks, and prints a sample of
    the generated chunks to verify the semantic chunking pipeline.
"""

from src.chunking.semantic_chunker import SemanticChunker
from src.loaders.document_loader import DocumentLoader


def main():
    """
    Test semantic chunking on documents in the same style as the
    fixed chunker smoke test.
    """

    # Step 1: Load documents
    loader = DocumentLoader()
    documents = loader.load_documents()

    # Step 2: Create semantic chunks
    chunker = SemanticChunker()
    chunks = chunker.create_chunks(documents)

    # Step 3: Print summary
    print("\n" + "=" * 80)
    print(f"Total Documents : {len(documents)}")
    print(f"Total Chunks    : {len(chunks)}")
    print("=" * 80)

    # Step 4: Display a few chunks
    print("\nSample Chunks\n")
    print("=" * 80)

    for chunk in chunks[:5]:
        print(chunk)
        print(f"Document : {chunk.document_name}")
        print(f"Tokens    : {chunk.token_count}")
        print(f"Words     : {chunk.word_count}")
        print(f"Characters: {chunk.character_count}")
        print("-" * 80)
        print(chunk.text[:250])
        print("-" * 80)
        print()


if __name__ == "__main__":
    main()