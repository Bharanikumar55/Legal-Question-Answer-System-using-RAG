"""
Embedding Database Module for RAG System
Handles generating embeddings and performing similarity-based retrieval
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Dict


class EmbeddingDB:
    """
    A vector database for storing text chunks and their embeddings,
    with efficient similarity search capabilities.
    """

    def __init__(self, model_name: str = "all-mpnet-base-v2"):
        """
        Initialize the EmbeddingDB with a sentence-transformer model.
        
        Args:
            model_name (str): Name of the sentence-transformer model to use.
                            Default is "all-MiniLM-L6-v2" (fast and efficient).
                            Other options: "all-mpnet-base-v2" (more accurate but slower),
                                         "all-roberta-large-v1" (higher quality)
        """
        self.model = SentenceTransformer(model_name)
        self.chunks: List[str] = []
        self.embeddings: np.ndarray = np.array([])
        self.model_name = model_name

    def add_documents(self, chunks: List[str]) -> None:
        """
        Add text chunks to the database.
        
        Args:
            chunks (List[str]): List of text chunks to add.
        
        Example:
            db.add_documents(["Fraud is punishable...", "Contract law states..."])
        """
        if not chunks:
            print("Warning: No chunks provided to add_documents()")
            return
        
        self.chunks.extend(chunks)
        print(f"✓ Added {len(chunks)} documents to database (total: {len(self.chunks)})")

    def create_embeddings(self) -> None:
        """
        Generate embeddings for all stored text chunks using sentence-transformers.
        
        This must be called after add_documents() and before search().
        
        Example:
            db.create_embeddings()
        """
        if not self.chunks:
            print("Error: No documents in database. Use add_documents() first.")
            return
        
        print(f"Generating embeddings for {len(self.chunks)} documents...")
        self.embeddings = self.model.encode(self.chunks, convert_to_numpy=True)
        print(f"✓ Embeddings generated successfully. Shape: {self.embeddings.shape}")

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Search for the most relevant documents based on query using cosine similarity.
        
        Args:
            query (str): The search query.
            top_k (int): Number of top results to return (default: 5).
        
        Returns:
            List[Tuple[str, float]]: List of tuples containing (chunk_text, similarity_score).
                                     Results are sorted by similarity score (highest first).
        
        Raises:
            ValueError: If no embeddings exist or invalid top_k value.
        
        Example:
            results = db.search("What is fraud?", top_k=3)
            for text, score in results:
                print(f"Score: {score:.4f}\nText: {text}\n")
        """
        if self.embeddings.size == 0:
            raise ValueError("No embeddings found. Call create_embeddings() first.")
        
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer.")
        
        if top_k > len(self.chunks):
            print(f"Warning: top_k ({top_k}) exceeds available documents ({len(self.chunks)}). Returning all.")
            top_k = len(self.chunks)
        
        # Encode the query
        query_embedding = self.model.encode(query, convert_to_numpy=True)
        
        # Compute cosine similarity between query and all documents
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        
        # Get top-k indices sorted by similarity (descending)
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Return top-k results as (chunk, similarity_score) tuples
        results = [
            (self.chunks[idx], float(similarities[idx]))
            for idx in top_indices
        ]
        
        return results

    def get_stats(self) -> Dict:
        """
        Get statistics about the current database state.
        
        Returns:
            Dict: Contains number of documents, embedding dimension, model name.
        
        Example:
            stats = db.get_stats()
            print(f"Database has {stats['num_documents']} documents")
        """
        return {
            "num_documents": len(self.chunks),
            "embedding_dim": self.embeddings.shape[1] if self.embeddings.size > 0 else 0,
            "model_name": self.model_name,
            "embeddings_created": self.embeddings.size > 0
        }

    def clear(self) -> None:
        """
        Clear all stored documents and embeddings.
        
        Example:
            db.clear()
        """
        self.chunks = []
        self.embeddings = np.array([])
        print("✓ Database cleared.")

""" The below code was used to test the working of the EmbeddingDB class.
 It demonstrates adding documents, creating embeddings, and performing a search query. Uncomment to run the demonstration. """
# # Example usage / demonstration
# if __name__ == "__main__":
#     # Initialize database
#     db = EmbeddingDB()
    
#     # Add sample legal documents
#     sample_chunks = [
# """Section 302 of the Indian Penal Code provides that whoever commits murder shall be punished with death, or imprisonment for life, and shall also be liable to fine. The determination of punishment shall take into account the gravity of the offense, the manner of commission, and any mitigating or aggravating circumstances presented before the court.""",
# """Notwithstanding anything contained in any other law for the time being in force, any agreement made without free consent of the parties thereto shall be voidable at the option of the party whose consent was so caused by coercion, undue influence, fraud, misrepresentation, or mistake, as defined under Sections 13 to 22 of the Indian Contract Act, 1872.""",
# """Where a person is accused of an offense under this Code, the burden of proving the existence of circumstances bringing the case within any of the general exceptions shall lie upon him, and the court shall presume the absence of such circumstances unless the contrary is proved, in accordance with the provisions of the Indian Evidence Act, 1872.""",
# """Any person who dishonestly induces the person deceived to deliver any property to any person, or to consent that any person shall retain any property, shall be punishable under Section 420 of the Indian Penal Code, which deals with cheating and dishonestly inducing delivery of property, and may extend to imprisonment of either description for a term which may extend to seven years, and shall also be liable to fine.""",
# """A contract, as defined under Section 2(h) of the Indian Contract Act, 1872, is an agreement enforceable by law. An agreement becomes a contract when it is made by the free consent of parties competent to contract, for a lawful consideration and with a lawful object, and is not hereby expressly declared to be void.""",
# """In cases where the terms of a contract are reduced to the form of a document, no evidence shall be given in proof of the terms of such contract except the document itself, or secondary evidence of its contents in cases in which secondary evidence is admissible under the provisions hereinbefore contained in the Indian Evidence Act, 1872.""",
# """Whoever voluntarily causes grievous hurt by means of any instrument for shooting, stabbing or cutting, or any instrument which, used as a weapon of offense, is likely to cause death, shall be punished under Section 326 of the Indian Penal Code with imprisonment for life, or with imprisonment of either description for a term which may extend to ten years, and shall also be liable to fine.""",
# """No suit shall be instituted against the Government or against a public officer in respect of any act purporting to be done by such public officer in his official capacity until the expiration of two months next after notice in writing has been delivered, in accordance with Section 80 of the Code of Civil Procedure, 1908.""",
# """The doctrine of res judicata, as embodied under Section 11 of the Code of Civil Procedure, 1908, prevents courts from adjudicating upon matters that have already been directly and substantially in issue in a former suit between the same parties, and have been finally decided by a competent court.""",
# """Subject to the provisions of this Act, any person aggrieved by an order passed by a lower court may prefer an appeal to a higher court, provided that such appeal is filed within the prescribed limitation period and complies with procedural requirements as laid down under the Limitation Act, 1963."""
# ]
    
#     db.add_documents(sample_chunks)
#     db.create_embeddings()
    
#     # Perform search
#     query = "Can you sue a public officer immediately?"
#     results = db.search(query, top_k=3)
    
#     print(f"\n🔍 Query: {query}\n")
#     print("🎯 Top Results:\n")
#     for i, (text, score) in enumerate(results, 1):
#         print(f"{i}. [Similarity: {score:.4f}]")
#         print(f"   {text}\n")
    
#     # Print stats
#     print(f"📊 Database Stats: {db.get_stats()}")
