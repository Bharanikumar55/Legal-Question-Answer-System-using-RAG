# 📌 Legal Question-Answering System using RAG

---

## 🧠 Overview

This project is an AI-powered system that answers user questions from large legal documents using **Retrieval-Augmented Generation (RAG)**.

Instead of manually reading lengthy documents, users can simply ask questions, and the system retrieves relevant information and generates accurate answers.

---

## 🎯 Objectives

* Extract meaningful answers from legal documents
* Reduce time spent on manual document reading
* Provide accurate and context-based responses
* Demonstrate the use of NLP and LLMs in real-world applications

---

## ⚙️ System Workflow

```
PDF → Text → Chunking → Embeddings → Vector DB → Query → Retrieval → LLM → Answer
```

---

## 🧱 Project Structure

```
legal-rag-project/
│
├── data/
│   └── sample.pdf
│
├── src/
│   ├── data_processing.py
│   ├── embedding_db.py
│   ├── query_retrieval.py
│   ├── llm_module.py
│   └── main.py
│
├── requirements.txt
├── README.md
└── .env
```

---

## 👥 Team Members & Roles

| Name     | Role                                 |
| -------- | ------------------------------------ |
| Bharani  | Query Processing & Retrieval         |
| Pranay   | Embeddings & Vector Database         |
| Sai Teja | Data Processing (PDF → Chunks)       |
| Parin    | LLM Integration & Prompt Engineering |

---

## 🧠 Key Concepts Used

* **Natural Language Processing (NLP)** – Understanding human language
* **Embeddings** – Converting text into numerical vectors
* **Vector Database** – Storing embeddings for similarity search
* **Retrieval** – Finding relevant information
* **LLM (Large Language Model)** – Generating human-like answers
* **RAG (Retrieval-Augmented Generation)** – Combining retrieval + generation

---

## 🛠️ Technologies & Libraries

* Python
* pdfplumber / PyPDF2
* sentence-transformers
* scikit-learn
* google-generativeai (Gemini API)
* (Optional) FAISS / ChromaDB

---

## 🚀 Features

* Handles large legal documents
* Retrieves relevant information efficiently
* Generates accurate answers using LLM
* Modular and scalable architecture
* Easy to extend into web applications

---

## 🧪 How It Works (Example)

**Document contains:**

> Fraud is punishable with imprisonment up to 3 years.

**User Query:**

> What is the punishment for fraud?

**System Output:**

> Fraud is punishable with imprisonment up to 3 years.

---

## ⚠️ Challenges

* Understanding complex legal language
* Avoiding incorrect (hallucinated) answers
* Handling very large documents efficiently
* Maintaining high retrieval accuracy

---

## 🔮 Future Enhancements

* Use FAISS for faster vector search
* Add Streamlit-based UI
* Highlight answers in document
* Add multilingual support
* Build real-time chatbot interface

---

## 🔧 Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Bharanikumar55/Legal-Question-Answer-System-using-RAG.git
cd Legal-Question-Answer-System-using-RAG
```

---

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 3. Add API Key

Create a `.env` file:

```
API_KEY=your_api_key_here
```

---

### 4. Run the Project

```bash
python src/main.py
```

---

## 📌 Usage

* Place your PDF file in the `data/` folder
* Run the program
* Enter your question
* Get accurate answers instantly

---

## 🧠 Key Insight

This project demonstrates how combining **retrieval mechanisms with language models** can significantly improve answer accuracy compared to standalone LLMs.

---

## 📌 One-Line Summary

An AI system that extracts accurate answers from legal documents using Retrieval-Augmented Generation (RAG).

---

## 🙌 Acknowledgements

* Hugging Face
* Google Generative AI (Gemini)
* Open-source NLP community

---
