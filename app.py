"""
LexAI — Legal Q&A System (Premium UI + LLM)
"""
import os
os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"
os.environ["DISABLE_TRANSFORMERS_AV"] = "1"
import streamlit as st
import sys
import tempfile
from pathlib import Path

# ── Page config ─────────────────────────────────────
st.set_page_config(
    page_title="LexAI — Legal Q&A System",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom UI Styling ───────────────────────────────
st.markdown("""
<style>
body {
    background-color: #0F0F0F;
    color: #E8E3DA;
}

h1, h2, h3 {
    color: #C9A84C;
}

.stButton button {
    background: #C9A84C;
    color: black;
    border-radius: 8px;
    padding: 0.5rem 1.5rem;
}

.answer-box {
    background: #1A1A1A;
    border-left: 4px solid #C9A84C;
    padding: 1.5rem;
    border-radius: 10px;
    margin-top: 1rem;
}

.chunk-box {
    background: #1A1A1A;
    padding: 1rem;
    border-radius: 8px;
    margin-top: 0.5rem;
    font-size: 0.9rem;
    color: #B0AA9F;
}
</style>
""", unsafe_allow_html=True)


# ── Add src to path ─────────────────────────────────
sys.path.append(str(Path(__file__).parent / "src"))

from data_processing import process_pdf
from embedding_db import EmbeddingDB
from query_retrieval import retrieve_relevant_chunks
from llm_module import generate_answer


# ── Session State ───────────────────────────────────
if "db" not in st.session_state:
    st.session_state.db = None


# ── Sidebar ─────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚖️ LexAI")
    st.markdown("AI-powered Legal Intelligence")

    uploaded_files = st.file_uploader(
        "Upload Legal PDFs",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:
        if st.button("Index Documents"):
            with st.spinner("Processing documents..."):
                db = EmbeddingDB()
                all_texts = []

                for file in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(file.read())
                        tmp_path = tmp.name

                    chunks = process_pdf(tmp_path)
                    texts = [c["content"] for c in chunks]
                    all_texts.extend(texts)

                db.add_documents(all_texts)
                db.create_embeddings()
                st.session_state.db = db
                st.success("Documents indexed successfully")


# ── Main UI ─────────────────────────────────────────
st.markdown("# ⚖️ LexAI")
st.markdown("### Ask questions from legal documents and get precise answers")

query = st.text_area("Enter your legal question:")

if st.button("🔍 Get Answer"):

    if not query.strip():
        st.warning("Please enter a question")

    elif st.session_state.db is None:
        st.error("Upload and index documents first")

    else:
        with st.spinner("Analyzing documents..."):

            chunks = retrieve_relevant_chunks(query, st.session_state.db, top_k=3)

            if not chunks:
                st.warning("No relevant information found")

            else:
                answer = generate_answer(query, chunks)

                # Answer Box
                st.markdown(f"""
                <div class="answer-box">
                    <h4>📌 Answer</h4>
                    <p>{answer}</p>
                </div>
                """, unsafe_allow_html=True)

                # Sources
                st.markdown("### 📎 Sources")
                for i, chunk in enumerate(chunks, 1):
                    st.markdown(f"""
                    <div class="chunk-box">
                        <b>Source {i}:</b><br>{chunk}
                    </div>
                    """, unsafe_allow_html=True)


# ── Professional Footer ──────────────────────────────
st.markdown("""
<hr style="border: 0.5px solid #333; margin-top: 3rem;">
<div style="text-align:center; color:#666; font-size:0.8rem;">
LexAI · Legal Question Answering System · Powered by RAG Architecture
</div>
""", unsafe_allow_html=True)