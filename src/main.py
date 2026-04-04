# TODO: Implement this module
"""
Main Entry Point - Legal RAG System
Connects: data_processing → embedding_db → query_retrieval → llm_module
"""

import os
from pathlib import Path
from data_processing import process_pdf
from embedding_db import EmbeddingDB
from query_retrieval import retrieve_relevant_chunks
from llm_module import generate_answer


def build_index(data_dir: str, db: EmbeddingDB):
    """
    Load all PDFs from data_dir, chunk them, and store in vector DB.
    This runs once at startup.
    """
    data_path = Path(data_dir)
    pdf_files = list(data_path.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in {data_dir}")
        return

    all_texts = []

    for pdf_file in pdf_files:
        try:
            print(f"Processing: {pdf_file.name}")
            chunks = process_pdf(str(pdf_file))
            texts = [c["content"] for c in chunks]
            all_texts.extend(texts)
            print(f"  ✓ {len(texts)} chunks extracted")
        except Exception as e:
            print(f"  ✗ Skipping {pdf_file.name} — {e}")

    db.add_documents(all_texts)
    db.create_embeddings()
    print(f"\n✓ Index built. Total chunks: {len(all_texts)}\n")


def main():
    # ── 1. Setup paths ──────────────────────────────────────────────
    project_root = Path(__file__).parent
    data_dir = project_root.parent / "data"

    # ── 2. Build the vector index from PDFs ─────────────────────────
    print("=" * 50)
    print("  Legal Question-Answering System (RAG)")
    print("=" * 50)

    db = EmbeddingDB()
    build_index(str(data_dir), db)

    # ── 3. Question-answering loop ───────────────────────────────────
    print("Ask questions about the legal documents.")
    print("Type 'exit' to quit.\n")

    while True:
        query = input("Your question: ").strip()

        if query.lower() == "exit":
            print("Goodbye!")
            break

        if not query:
            continue

        # Retrieve relevant chunks
        top_chunks = retrieve_relevant_chunks(query, db, top_k=3)

        if not top_chunks:
            print("No relevant information found in the documents.\n")
            continue

        # Generate answer using LLM
        answer = generate_answer(query, top_chunks)
        print(f"\nAnswer: {answer}\n")
        print("-" * 40)


if __name__ == "__main__":
    main()