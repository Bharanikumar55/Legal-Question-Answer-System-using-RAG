# TODO: Implement this module
# query_retrieval.py

from sklearn.metrics.pairwise import cosine_similarity


# -------------------------------
# Step 1: Clean Query
# -------------------------------
def clean_query(query: str) -> str:
    """
    Clean user query for better retrieval
    """
    return query.lower().strip()


# -------------------------------
# Step 2: Get Query Embedding
# -------------------------------
def get_query_embedding(query: str, model):
    """
    Convert query to embedding vector
    """
    return model.encode(query)


# -------------------------------
# Step 3: Similarity Search
# -------------------------------
def compute_similarity(query_embedding, vector_db):
    """
    Compute cosine similarity between query and stored embeddings
    """
    results = []

    for item in vector_db:
        score = cosine_similarity(
            [query_embedding], [item["embedding"]]
        )[0][0]

        results.append({
            "text": item["text"],
            "score": score
        })

    return results


# -------------------------------
# Step 4: Filter + Rank Results
# -------------------------------
def rank_and_filter(results, top_k=3, threshold=0.3):
    """
    Sort results and filter low-quality matches
    """
    # Remove low similarity results
    filtered = [r for r in results if r["score"] >= threshold]

    # Sort by highest similarity
    filtered.sort(key=lambda x: x["score"], reverse=True)

    # Remove duplicate texts
    seen = set()
    unique_results = []
    for r in filtered:
        if r["text"] not in seen:
            unique_results.append(r)
            seen.add(r["text"])

    return unique_results[:top_k]


# -------------------------------
# Step 5: Final Retrieval Pipeline
# -------------------------------
def retrieve_relevant_chunks(query, vector_db, model, top_k=3):
    """
    Full pipeline: query → embedding → similarity → top chunks
    """
    # Clean query
    query = clean_query(query)

    # Convert to embedding
    query_embedding = get_query_embedding(query, model)

    # Compute similarity
    results = compute_similarity(query_embedding, vector_db)

    # Rank + filter
    top_results = rank_and_filter(results, top_k=top_k)

    # Extract only text
    top_chunks = [r["text"] for r in top_results]

    return top_chunks