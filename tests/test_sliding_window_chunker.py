"""
Test Sliding Window Chunker
"""

from src.chunking.sliding_window_chunker import (
    SlidingWindowChunker,
)
from src.loaders.document_loader import (
    DocumentLoader,
)


def main():

    loader = DocumentLoader()

    documents = loader.load_documents()

    chunker = SlidingWindowChunker(
        chunk_size=200,
        overlap=20,
    )

    chunks = chunker.create_chunks(
        documents
    )

    print()

    print("=" * 80)
    print("Sliding Window Chunk Validation")
    print("=" * 80)

    all_valid = True

    #
    # Validate chunk size
    #
    for chunk in chunks:

        if chunk.token_count > chunker.chunk_size:

            print(
                f"[FAILED] Chunk "
                f"{chunk.chunk_id}"
            )

            all_valid = False

    #
    # Validate overlap
    #
    previous_chunk = None

    for chunk in chunks:

        if (
            previous_chunk is not None
            and
            previous_chunk.document_id
            == chunk.document_id
        ):

            expected_start = (
                previous_chunk.end_token
                - chunker.overlap
                + 1
            )

            if chunk.start_token != expected_start:

                print(
                    f"[FAILED] "
                    f"Overlap mismatch "
                    f"between chunks "
                    f"{previous_chunk.chunk_id}"
                    f" and "
                    f"{chunk.chunk_id}"
                )

                all_valid = False

        previous_chunk = chunk

    if all_valid:

        print()

        print(
            "All validations passed."
        )

    print()

    print("=" * 80)
    print("Sample Chunks")
    print("=" * 80)

    for chunk in chunks[:5]:

        print(chunk)

        print(
            f"Token Range : "
            f"{chunk.start_token}"
            f" - "
            f"{chunk.end_token}"
        )

        print(
            f"Token Count : "
            f"{chunk.token_count}"
        )

        print(
            f"Words       : "
            f"{chunk.word_count}"
        )

        print()


if __name__ == "__main__":
    main()