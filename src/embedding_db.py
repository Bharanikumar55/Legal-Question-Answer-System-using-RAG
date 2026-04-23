from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Dict


class EmbeddingDB:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.chunks: List[str] = []
        self.embeddings = None

    def add_documents(self, chunks: List[str]) -> None:
        self.chunks.extend(chunks)

    def create_embeddings(self) -> None:
        self.embeddings = self.vectorizer.fit_transform(self.chunks)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.embeddings)[0]

        top_indices = similarities.argsort()[::-1][:top_k]

        return [(self.chunks[i], float(similarities[i])) for i in top_indices]