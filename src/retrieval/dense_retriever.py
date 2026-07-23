"""
Dense Retriever using SentenceTransformer + FAISS

Author:
    Hari Sharma

Description:
    Builds a dense vector index over document chunks and performs
    semantic retrieval using cosine similarity.
"""

import time
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer

class DenseRetriever:

    def __init__(self,
                 model_path="./models/embeddings/all-MiniLM-L6-v2"):

        self.model = SentenceTransformer(
            model_path,
            local_files_only=True
        )

        self.index = None
        self.chunks = []
        self.embeddings = None
        self.model_name = "all-MiniLM-L6-v2"
        self.embedding_dimension = 384
        self.similarity_metric = "Cosine Similarity"

    def build_index(self, chunks):

        self.chunks = chunks

        texts = [chunk.text for chunk in chunks]

        start = time.perf_counter()

        embeddings = self.model.encode(
            texts,
            batch_size=32,
            convert_to_numpy=True,
            show_progress_bar=True
        )

        embeddings = embeddings.astype("float32")

        # Cosine Similarity
        faiss.normalize_L2(embeddings)

        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(dimension)

        self.index.add(embeddings)

        self.embeddings = embeddings

        end = time.perf_counter()

        build_time = (end - start) * 1000

        return build_time

    def retrieve(self,
                 query,
                 top_k=5):

        start = time.perf_counter()

        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True
        )

        query_embedding = query_embedding.astype("float32")

        faiss.normalize_L2(query_embedding)

        scores, indices = self.index.search(
            query_embedding,
            top_k
        )

        end = time.perf_counter()

        latency = (end - start) * 1000

        results = []

        for score, idx in zip(scores[0], indices[0]):

            results.append({
                "score": float(score),
                "chunk": self.chunks[idx]
            })

        return results, latency