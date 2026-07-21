"""
It is a simple test script for verifying the Document model.

Author:
    Mrityunjay Dubey, Hari Sharma

Description:
    This script creates a sample Document object and prints its basic properties
    to verify the document model behavior.
"""

from pathlib import Path

from src.models.document import Document

doc = Document(
    document_id=1,
    file_name="Leave-and-Holiday-Policy.txt",
    file_path=Path("data/corpus/Leave-and-Holiday-Policy.txt"),
    text=(
        "This Leave and Holiday Policy outlines the guidelines and procedures "
        "for employees regarding leave and holiday entitlements."
    )
)

print(doc)
print(doc.word_count)
print(doc.character_count)