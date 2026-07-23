"""
Performance test for Dense Retrieval.

Author:
    Hari Sharma

Description:
    This test script evaluates Dense Retrieval using FAISS
    on the best chunking strategy from Assignment Part A.
"""

import pandas as pd

from src.loaders.document_loader import DocumentLoader
from src.chunking.semantic_chunker import SemanticChunker
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.sparse_retriever import SparseRetriever


# -------------------------------------------------------
# Step 1 : Load Documents
# -------------------------------------------------------

loader = DocumentLoader()
documents = loader.load_documents()

print(f"\nTotal Documents Loaded : {len(documents)}")


# -------------------------------------------------------
# Step 2 : Best Chunking Strategy
# -------------------------------------------------------

chunker = SemanticChunker()
chunks = chunker.create_chunks(documents)

print(f"Total Chunks Created : {len(chunks)}")


# -------------------------------------------------------
# Step 3 : Build Dense Index
# -------------------------------------------------------

retriever = SparseRetriever()

build_time = retriever.build_index(chunks)

print(f"\nIndex Build Time : {build_time:.2f} ms")


# -------------------------------------------------------
# Step 4 : Assignment-1 Evaluation Queries
# -------------------------------------------------------

queries = [

    # Replace with your Assignment-1 evaluation queries

    "Explain the company's annual leave policy and carry-forward rules.",
    "What employee behavior is expected under the code of conduct?",
    "Who is eligible for remote work and what conditions apply?",
    "Describe the employee grievance process.",
    "What are the rules regarding workplace harassment?",
    "Explain probation period expectations.",
    "How are employee performance reviews conducted?",
    "What is the company's attendance policy?",
    "What training requirements apply to new employees?",
    "Explain resignation and notice period requirements."

]


# -------------------------------------------------------
# Step 5 : Run Retrieval
# -------------------------------------------------------

results_data = []

for query_no, query in enumerate(queries, start=1):

    results, latency = retriever.retrieve(
        query=query,
        top_k=5
    )

    print("=" * 80)
    print(f"Query : {query}")
    print(f"Query {query_no}")
    print(f"Latency : {latency:.2f} ms")
    print("=" * 80)

    for rank, result in enumerate(results, start=1):

        print("-" * 60)
        print(f"Rank : {rank}")
        print(f"Document : {result['chunk'].document_name}")
        print(f"Similarity Score : {result['score']:.4f}")

        print("-" * 80)
        print(result["chunk"].text[:300])
        print("-" * 80)

    results_data.append({

        "Query": query,
        "Latency (ms)": round(latency, 2),
        "Top Score": round(results[0]["score"], 4)

    })


# -------------------------------------------------------
# Step 6 : Summary
# -------------------------------------------------------

df = pd.DataFrame(results_data)

print("\n")
print("=" * 70)
print("Sparse Retrieval Summary")
print("=" * 70)

print(df.to_string(index=False))