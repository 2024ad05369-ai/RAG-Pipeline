from src.loaders.document_loader import DocumentLoader


def main():

    loader = DocumentLoader()

    documents = loader.load_documents()

    print("\n")

    print("=" * 60)
    print(f"Documents Loaded : {len(documents)}")
    print("=" * 60)

    for document in documents:

        print(document)


if __name__ == "__main__":
    main()