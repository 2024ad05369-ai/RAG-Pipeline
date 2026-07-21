"""
Document Loader

Author:
    Mrityunjay Dubey, Hari Sharma

Description:
    Loads all cleaned text documents from the corpus directory and
    converts them into Document objects.

Responsibilities
----------------
1. Discover all .txt files.
2. Validate each file.
3. Read document contents.
4. Create Document objects.
5. Return a list of loaded documents.
"""

from pathlib import Path

from config.config import CORPUS_DIR
from config.logging_config import logger
from src.models.document import Document
from src.utils.tokenizer import Tokenizer

logger.info("Loading HR Policy documents...")
class DocumentLoader:
    """
    Loads HR Policy documents from the configured corpus directory.
    """

    def __init__(self, corpus_directory: Path = CORPUS_DIR):
        """
        Initialize the loader.

        Parameters
        ----------
        corpus_directory : Path
            Directory containing cleaned text files.
        """

        self.corpus_directory = corpus_directory

    def load_documents(self) -> list[Document]:
        """
        Load all text documents from the corpus directory.

        Returns
        -------
        list[Document]
            List of loaded Document objects.

        Raises
        ------
        FileNotFoundError
            If the corpus directory does not exist.
        """

        if not self.corpus_directory.exists():
            raise FileNotFoundError(
                f"Corpus directory not found: {self.corpus_directory}"
            )

        text_files = sorted(self.corpus_directory.glob("*.txt"))

        if not text_files:
            logger.warning("No text files found in the corpus directory.")
            return []

        logger.info("Found %d text files.", len(text_files))

        documents: list[Document] = []

        for document_id, file_path in enumerate(text_files, start=1):

            try:

                text = file_path.read_text(
                    encoding="utf-8"
                ).strip()

                if not text:
                    logger.warning(
                        "Skipping empty document: %s",
                        file_path.name
                    )
                    continue

                document = Document(
                    document_id=document_id,
                    file_name=file_path.name,
                    file_path=file_path,
                    text=text,
                    token_count=Tokenizer.count_tokens(text),
                )

                documents.append(document)

                logger.info(
                    "Loaded %-30s (%d words)",
                    file_path.name,
                    document.word_count
                )

            except Exception as error:

                logger.exception(
                    "Failed to load %s",
                    file_path.name
                )

                raise error

        logger.info(
            "Successfully loaded %d documents.",
            len(documents)
        )

        return documents