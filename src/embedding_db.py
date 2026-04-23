"""
Embedding Database Module using FAISS (Optimized Vector Search)
"""

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Dict


class EmbeddingDB:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.chunks: List[str] = []
        self.index = None
        self.embedding_dim = None

    def add_documents(self, chunks: List[str]) -> None:
        if not chunks:
            print("Warning: No chunks provided")
            return

        self.chunks.extend(chunks)
        print(f"✓ Added {len(chunks)} documents (Total: {len(self.chunks)})")

    def create_embeddings(self) -> None:
        if not self.chunks:
            print("Error: No documents to embed")
            return

        print("Generating embeddings...")
        embeddings = self.model.encode(self.chunks, convert_to_numpy=True)

        # Normalize (cosine similarity)
        faiss.normalize_L2(embeddings)

        self.embedding_dim = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(embeddings)

        print(f"✓ FAISS index created with {len(self.chunks)} vectors")

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        if self.index is None:
            raise ValueError("Index not created. Call create_embeddings() first.")

        query_embedding = self.model.encode([query], convert_to_numpy=True)

        faiss.normalize_L2(query_embedding)

        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunks):
                results.append((self.chunks[idx], float(score)))

        return results

    def get_stats(self) -> Dict:
        return {
            "num_documents": len(self.chunks),
            "embedding_dim": self.embedding_dim,
            "index_created": self.index is not None
        }

    def clear(self) -> None:
        self.chunks = []
        self.index = None
        print("✓ Database cleared")