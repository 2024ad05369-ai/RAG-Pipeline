from src.chunking.fixed_chunker import FixedChunker
from src.loaders.document_loader import DocumentLoader
from src.models import chunk
from src.utils.tokenizer import Tokenizer


def main():

    # Step 1: Load documents
    loader = DocumentLoader()
    documents = loader.load_documents()

    # Step 2: Create chunks
    chunker = FixedChunker()
    chunks = chunker.create_chunks(documents)

    # Step 3: Print summary
    print("\n" + "=" * 80)
    print(f"Total Documents : {len(documents)}")
    print(f"Total Chunks    : {len(chunks)}")
    print("=" * 80)

    # Step 4: Validate every chunk
    print("\nValidating chunks...\n")

    all_valid = True

    for chunk in chunks:

        if chunk.token_count > chunker.chunk_size:

            print(
                f"[FAILED] Chunk {chunk.chunk_id} "
                f"contains {chunk.token_count} tokens."
            )

            all_valid = False

        if chunk.start_token > chunk.end_token:

            print(
                f"[FAILED] Invalid token range "
                f"for chunk {chunk.chunk_id}"
            )

            all_valid = False

    if all_valid:

        print()

        print("All chunks validated successfully.")

    # Step 5: Display first few chunks
    print("\nSample Chunks\n")
    print("=" * 80)

    for chunk in chunks[:5]:


        print(chunk)
        print(f"Start Token : {chunk.start_token}")
        print(f"End Token   : {chunk.end_token}")
        print(f"Tokens      : {chunk.token_count}")
        print(f"Words       : {chunk.word_count}")
        print(f"Characters  : {chunk.character_count}")
        print(f"Token Count : {chunk.token_count}")
        print("-" * 80)
        print(chunk.text[:250])
        print("-" * 80)
        print()


if __name__ == "__main__":
    main()