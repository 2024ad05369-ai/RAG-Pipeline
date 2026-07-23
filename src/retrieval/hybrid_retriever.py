"""
Hybrid Retriever using Reciprocal Rank Fusion (RRF)

Author:
    Hari Sharma
"""

import time

from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.sparse_retriever import SparseRetriever


class HybridRetriever:

    def __init__(self):

        self.dense = DenseRetriever()
        self.sparse = SparseRetriever()

    def build_index(self, chunks):

        dense_time = self.dense.build_index(chunks)

        sparse_time = self.sparse.build_index(chunks)

        return dense_time + sparse_time

    def retrieve(self,
                 query,
                 top_k=5,
                 k=60):

        start = time.perf_counter()

        dense_results, _ = self.dense.retrieve(query, top_k)

        sparse_results, _ = self.sparse.retrieve(query, top_k)

        fused_scores = {}
        chunk_lookup = {}

        # Dense contribution
        for rank, result in enumerate(dense_results, start=1):

            chunk = result["chunk"]
            chunk_id = chunk.chunk_id

            chunk_lookup[chunk_id] = chunk

            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0) + (1 / (k + rank))

        # Sparse contribution
        for rank, result in enumerate(sparse_results, start=1):

            chunk = result["chunk"]
            chunk_id = chunk.chunk_id

            chunk_lookup[chunk_id] = chunk
            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0) + (1 / (k + rank))

        ranked_results = sorted(
            fused_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        end = time.perf_counter()

        latency = (end - start) * 1000

        results = []

        for chunk_id, score in ranked_results[:top_k]:

            results.append({

                "score": score,

                "chunk": chunk_lookup[chunk_id]

            })

        return results, latency