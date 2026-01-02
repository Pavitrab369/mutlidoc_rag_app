# Multi-Doc RAG App (Streamlit + Groq + Chroma)

A lightweight Streamlit frontend for RAG over uploaded PDFs. Upload one or more PDFs, index them into a local Chroma vector store with HuggingFace embeddings, and ask questions answered by Groq’s `llama-3.3-70b-versatile` model.

## Features

- Multi-PDF upload (drag/drop)
- Chunking via RecursiveCharacterTextSplitter
- Local persistent Chroma vector store (`docs_vectorstore/`)
- HuggingFace embeddings
- Groq Llama 3.3 70B QA via RetrievalQA
- Simple Streamlit UI

## Requirements

- Python 3.9+
- Dependencies: `streamlit`, `langchain_community`, `langchain_text_splitters`, `langchain_huggingface`, `langchain_chroma`, `langchain_groq`, `python-dotenv`, `unstructured` (PDF), and their system deps (e.g., Poppler/Tesseract if needed by Unstructured).

## Setup

1. Create and activate a venv

```bash
python -m venv .venv
.venv\Scripts\activate
```

2. Install deps

```bash
pip install -r requirements.txt
```

(If `requirements.txt` is absent, install the packages listed above.)

3. Environment variables (`.env`)

```bash
GROQ_API_KEY=your_groq_api_key
```

## Run

```bash
streamlit run app.py
```

## Usage

1. Open the Streamlit UI.
2. Upload one or more PDFs; they are saved to `data/` and indexed into `docs_vectorstore/`.
3. Ask a question in the text box and click **answer** to get the model’s response.

## Notes

- The vector store persists on disk; delete `docs_vectorstore/` to reindex from scratch.
- Ensure Unstructured’s optional system dependencies are installed for robust PDF parsing.
