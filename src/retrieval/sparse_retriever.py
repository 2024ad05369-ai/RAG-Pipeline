"""
Sparse Retriever using BM25

Author:
    Hari Sharma

Description:
    Implements sparse retrieval using the BM25 ranking algorithm.
"""

import time

from rank_bm25 import BM25Okapi


class SparseRetriever:

    def __init__(self):

        self.chunks = []
        self.bm25 = None

    def build_index(self, chunks):

        self.chunks = chunks

        tokenized_corpus = []

        start = time.perf_counter()

        for chunk in chunks:
            tokenized_corpus.append(
                chunk.text.lower().split()
            )

        self.bm25 = BM25Okapi(tokenized_corpus)

        end = time.perf_counter()

        build_time = (end - start) * 1000

        return build_time

    def retrieve(self,
                 query,
                 top_k=5):

        start = time.perf_counter()

        tokenized_query = query.lower().split()

        scores = self.bm25.get_scores(tokenized_query)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]

        end = time.perf_counter()

        latency = (end - start) * 1000

        results = []

        for idx in ranked_indices:

            results.append({

                "score": float(scores[idx]),
                "chunk": self.chunks[idx]

            })

        return results, latency