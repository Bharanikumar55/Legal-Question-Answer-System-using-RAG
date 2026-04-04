"""
Query Retrieval Module for RAG System
Handles: query cleaning, embedding, similarity search, and ranking
"""

from sklearn.metrics.pairwise import cosine_similarity


# -------------------------------
# Step 1: Clean Query
# -------------------------------
def clean_query(query: str) -> str:
    """Clean user query for better retrieval"""
    return query.lower().strip()


# -------------------------------
# Step 2: Get Query Embedding
# -------------------------------
def get_query_embedding(query: str, model):
    """Convert query to embedding vector"""
    return model.encode(query)


# -------------------------------
# Step 3: Filter + Rank Results
# -------------------------------
def rank_and_filter(results, top_k=3, threshold=0.3):
    """
    Sort results and filter low-quality matches.
    Expects results as list of (text, score) tuples from EmbeddingDB.search()
    """
    # Filter low similarity results
    filtered = [(text, score) for text, score in results if score >= threshold]

    # Sort by highest similarity
    filtered.sort(key=lambda x: x[1], reverse=True)

    # Remove duplicate texts
    seen = set()
    unique_results = []
    for text, score in filtered:
        if text not in seen:
            unique_results.append((text, score))
            seen.add(text)

    return unique_results[:top_k]


# -------------------------------
# Step 4: Final Retrieval Pipeline
# -------------------------------
def retrieve_relevant_chunks(query, db, top_k=3):
    """
    Full pipeline: query → embedding → similarity search → top chunks
    
    Args:
        query (str): User's question
        db (EmbeddingDB): The embedding database with stored chunks
        top_k (int): Number of top chunks to return
    
    Returns:
        List[str]: Top matching text chunks
    """
    # Clean query
    cleaned_query = clean_query(query)

    # Use EmbeddingDB's built-in search (returns [(text, score), ...])
    results = db.search(cleaned_query, top_k=top_k * 2)  # fetch extra, then filter

    # Rank + filter by threshold
    top_results = rank_and_filter(results, top_k=top_k, threshold=0.3)

    # Return only the text
    return [text for text, score in top_results]