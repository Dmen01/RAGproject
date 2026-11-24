# LocalDocs

A local document retrieval and question-answering tool built with FastAPI and ChromaDB. It processes PDF and text files, breaks them into overlapping text chunks, generates vector embeddings, and performs grounded semantic search.

---

## Architecture Overview

1. **Ingestion & Text Extraction**
   - Extracts plain text from `.pdf` (using `pypdf`) and `.txt` files.
   - Rejects blank, unreadable, or scanned non-OCR files.

2. **Chunking**
   - Splits extracted text into ~700-character windows with a 100-character overlap to retain context across boundaries.

3. **Vector Storage (ChromaDB)**
   - Computes dense vector embeddings in batches of up to 50 items.
   - Stores chunk embeddings with metadata (`doc_id`, `filename`) in a local `./chroma_db` database.
   - Maintains an `uploads/registry.json` manifest for quick file listing without querying Chroma collection tables.

4. **Grounded Querying**
   - Queries ChromaDB using vector similarity for the top relevant chunks.
   - Prompts the model strictly with the retrieved passages to answer queries accurately or state that the information is absent.

---

## Directory Layout

```
rag-app/
├── main.py            # FastAPI routes and server setup
├── rag_engine.py      # Core processing (parsing, chunking, embeddings, ChromaDB)
├── requirements.txt   # Python dependencies
├── static/
│   └── index.html     # Single-page interface
├── uploads/           # Raw uploaded files and document registry
│   └── registry.json
├── chroma_db/         # Local persistent vector database
└── .env               # Environment variables (GEMINI_API_KEY)
```

---

## Installation & Setup

1. **Set up the virtual environment:**
   ```bash
   cd rag-app
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Configure your API key:**
   Create or edit `.env` in the `rag-app` directory:
   ```env
   GEMINI_API_KEY="your_api_key_here"
   ```

3. **Start the local server:**
   ```bash
   uvicorn main:app --reload --port 8000
   ```

4. **Access the interface:**
   Open `http://localhost:8000` in your browser.

---

## API Endpoints

- `GET /` — Serves the frontend single-page application.
- `POST /api/upload` — Accepts a `.pdf` or `.txt` multipart upload, extracts text, chunks, embeds, and indexes it.
- `GET /api/documents` — Returns the current list of indexed documents.
- `DELETE /api/documents/{doc_id}` — Deletes the file and purges all its vectors from ChromaDB.
- `POST /api/ask` — Accepts `{"question": "..."}` and returns `{"answer": "...", "sources": [...]}`.
