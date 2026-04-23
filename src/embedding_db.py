"""
Embedding Database Module for RAG System
Handles generating embeddings and performing similarity-based retrieval
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Dict


class EmbeddingDB:
    def __init__(self, model_name: str = "all-mpnet-base-v2"):
        self.model = SentenceTransformer(model_name)
        self.chunks: List[str] = []
        self.embeddings: np.ndarray = np.array([])
        self.model_name = model_name

    def add_documents(self, chunks: List[str]) -> None:
        if not chunks:
            print("Warning: No chunks provided")
            return
        
        self.chunks.extend(chunks)
        print(f"✓ Added {len(chunks)} documents (total: {len(self.chunks)})")

    def create_embeddings(self) -> None:
        if not self.chunks:
            print("Error: No documents to embed")
            return
        
        print(f"Generating embeddings for {len(self.chunks)} documents...")
        self.embeddings = self.model.encode(self.chunks, convert_to_numpy=True)
        print(f"✓ Embeddings created. Shape: {self.embeddings.shape}")

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        if self.embeddings.size == 0:
            raise ValueError("Call create_embeddings() first")

        if top_k > len(self.chunks):
            top_k = len(self.chunks)

        query_embedding = self.model.encode(query, convert_to_numpy=True)

        similarities = cosine_similarity([query_embedding], self.embeddings)[0]

        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = [
            (self.chunks[idx], float(similarities[idx]))
            for idx in top_indices
        ]

        return results

    def get_stats(self) -> Dict:
        return {
            "num_documents": len(self.chunks),
            "embedding_dim": self.embeddings.shape[1] if self.embeddings.size > 0 else 0,
            "model_name": self.model_name,
            "embeddings_created": self.embeddings.size > 0
        }

    def clear(self) -> None:
        self.chunks = []
        self.embeddings = np.array([])
        print("✓ Database cleared")