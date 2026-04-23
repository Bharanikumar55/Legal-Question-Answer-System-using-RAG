"""
LLM Module - Gemini (Stable Working Version)
"""

import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load env
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found")

genai.configure(api_key=GEMINI_API_KEY)


def format_context(chunks, max_chars=4000):
    formatted = []
    total_length = 0

    for i, chunk in enumerate(chunks):
        text = f"[Source {i+1}]: {chunk}"

        if total_length + len(text) > max_chars:
            break

        formatted.append(text)
        total_length += len(text)

    return "\n\n".join(formatted)


def generate_answer(query, chunks):
    if not chunks:
        return "No relevant context found."

    context = format_context(chunks)

    prompt = f"""
You are a legal assistant AI.

Rules:
- Answer ONLY from the context
- Do NOT use outside knowledge
- If not found, say: "Answer not found in context"
- Mention sources like (Source 1)

Context:
{context}

Question:
{query}

Answer:
"""

    try:
        model = genai.GenerativeModel("models/gemini-2.5-flash")
        response = model.generate_content(prompt)

        return response.text.strip()

    except Exception as e:
        return f"LLM Error: {str(e)}"