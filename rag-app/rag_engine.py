import os
import json
import uuid
from pathlib import Path
from dotenv import load_dotenv
import pypdf
import chromadb
from google import genai
from google.genai import types

# -----------------------------------------------------------------------------
# Configuration and Initialization
# -----------------------------------------------------------------------------

# Load environment configuration (.env)
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=BASE_DIR / ".env")
load_dotenv()

# Client setup for embeddings and text generation
api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    client = genai.Client(api_key=api_key)
else:
    client = genai.Client()

# Local persistent vector store
# Chunks are stored in a persistent SQLite/duckdb-backed Chroma instance
CHROMA_PATH = BASE_DIR / "chroma_db"
chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))
collection = chroma_client.get_or_create_collection(name="documents")

# File storage and fast metadata registry
# A lightweight JSON registry tracks doc_id -> metadata so listing documents
# doesn't require scanning all vectors in ChromaDB on every page load.
UPLOADS_DIR = BASE_DIR / "uploads"
REGISTRY_PATH = UPLOADS_DIR / "registry.json"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Metadata Registry Helpers
# -----------------------------------------------------------------------------

def get_registry() -> dict:
    """Read the local document catalog from disk."""
    if not REGISTRY_PATH.exists():
        return {}
    try:
        with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_registry(registry: dict):
    """Persist updated document catalog to disk."""
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)


# -----------------------------------------------------------------------------
# Text Extraction and Chunking
# -----------------------------------------------------------------------------

def extract_text(file_path: Path, filename: str) -> str:
    """
    Extract readable text content from supported file formats (.pdf, .txt).
    Raises ValueError if unsupported or if the extracted content is blank
    (such as scanned images without an OCR text layer).
    """
    ext = Path(filename).suffix.lower()
    text = ""

    if ext == ".txt":
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
    elif ext == ".pdf":
        reader = pypdf.PdfReader(str(file_path))
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    else:
        raise ValueError(f"Unsupported file extension: {ext}. Only .pdf and .txt are allowed.")

    text = text.strip()
    if not text:
        raise ValueError("File appears empty or contains non-extractable content.")

    return text


def chunk_text(text: str, chunk_size: int = 700, chunk_overlap: int = 100) -> list[str]:
    """
    Slice text into overlapping character windows.
    The overlap ensures context across chunk boundaries isn't lost during retrieval.
    """
    chunks = []
    if not text:
        return chunks

    step = chunk_size - chunk_overlap
    if step <= 0:
        step = chunk_size

    for i in range(0, len(text), step):
        chunk = text[i:i + chunk_size].strip()
        if chunk:
            chunks.append(chunk)

    return chunks


# -----------------------------------------------------------------------------
# Embeddings
# -----------------------------------------------------------------------------

def embed_texts(texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
    """
    Generate dense vector representations for text chunks.
    Processes in batches of up to 50 items to respect API payload limits.
    """
    all_embeddings = []
    batch_size = 50

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = client.models.embed_content(
            model="gemini-embedding-001",
            contents=batch,
            config=types.EmbedContentConfig(task_type=task_type)
        )
        for emb in response.embeddings:
            all_embeddings.append(emb.values)

    return all_embeddings


def embed_query(query: str) -> list[float]:
    """
    Generate a query vector for similarity search in the vector database.
    Uses task_type RETRIEVAL_QUERY for symmetric similarity matching against document chunks.
    """
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=query,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
    )
    if response.embeddings:
        return response.embeddings[0].values
    raise ValueError("Failed to obtain query embedding.")


# -----------------------------------------------------------------------------
# Indexing and Deletion Pipeline
# -----------------------------------------------------------------------------

def index_document(filename: str, file_bytes: bytes) -> dict:
    """
    Complete document ingestion pipeline:
    1. Saves original uploaded file to disk.
    2. Extracts raw text.
    3. Chunks text into overlapping spans.
    4. Computes vector embeddings.
    5. Inserts vectors + chunk text + metadata into ChromaDB.
    6. Updates local registry.json catalog.
    """
    doc_id = str(uuid.uuid4())
    safe_filename = Path(filename).name
    saved_path = UPLOADS_DIR / f"{doc_id}_{safe_filename}"

    with open(saved_path, "wb") as f:
        f.write(file_bytes)

    try:
        text = extract_text(saved_path, safe_filename)
        chunks = chunk_text(text, chunk_size=700, chunk_overlap=100)
        if not chunks:
            raise ValueError("Document contains no readable text chunks.")

        embeddings = embed_texts(chunks, task_type="RETRIEVAL_DOCUMENT")
        chunk_ids = [f"{doc_id}_{idx}" for idx in range(len(chunks))]
        metadatas = [{"doc_id": doc_id, "filename": safe_filename} for _ in range(len(chunks))]

        collection.add(
            ids=chunk_ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas
        )

        registry = get_registry()
        registry[doc_id] = {"filename": safe_filename, "chunks": len(chunks)}
        save_registry(registry)

        return {"doc_id": doc_id, "filename": safe_filename, "chunks": len(chunks)}
    except Exception:
        # Clean up disk file if indexing fails midway
        if saved_path.exists():
            saved_path.unlink()
        raise


def list_documents() -> list[dict]:
    """Retrieve all indexed documents currently stored in the system."""
    registry = get_registry()
    docs = []
    for doc_id, info in registry.items():
        docs.append({
            "doc_id": doc_id,
            "filename": info.get("filename", "unknown"),
            "chunks": info.get("chunks", 0)
        })
    return docs


def delete_document(doc_id: str) -> bool:
    """
    Purge a document from both the filesystem and ChromaDB:
    - Removes all vectors matching doc_id.
    - Deletes stored raw file from uploads/.
    - Removes entry from registry.json.
    """
    registry = get_registry()
    if doc_id not in registry:
        return False

    info = registry[doc_id]
    filename = info.get("filename", "")

    # Purge vectors from Chroma collection
    collection.delete(where={"doc_id": doc_id})

    # Purge original file
    saved_path = UPLOADS_DIR / f"{doc_id}_{filename}"
    if saved_path.exists():
        saved_path.unlink()

    # Update catalog
    del registry[doc_id]
    save_registry(registry)
    return True


# -----------------------------------------------------------------------------
# Retrieval-Augmented Generation 
# -----------------------------------------------------------------------------

def query_rag(question: str) -> dict:
    """
    RAG Query Flow:
    1. Embed user question.
    2. Query ChromaDB for top 5 most similar document chunks.
    3. Format retrieved passages with source citations.
    4. Send strictly grounded prompt to the language model.
    5. Return factual answer + list of cited document names.
    """
    count = collection.count()
    if count == 0:
        return {
            "answer": "Please upload a document first before asking questions.",
            "sources": []
        }

    # Retrieve top 5 closest chunks (or fewer if total chunks < 5)
    n_results = min(5, count)
    query_vector = embed_query(question)

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=n_results
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    if not documents:
        return {
            "answer": "I don't know based on the uploaded documents.",
            "sources": []
        }

    # Assemble context passages and track unique sources
    sources = []
    context_blocks = []
    for doc_text, meta in zip(documents, metadatas):
        src_name = meta.get("filename", "Unknown document")
        if src_name not in sources:
            sources.append(src_name)
        context_blocks.append(f"--- Document: {src_name} ---\n{doc_text}")

    context_str = "\n\n".join(context_blocks)

    # Grounded prompt: enforce strict context adherence to prevent hallucinations
    strict_prompt = (
        "You are a factual assistant. Answer the question ONLY using the provided context.\n"
        "If the answer cannot be found in the provided context, respond with EXACTLY:\n"
        "\"I don't know based on the uploaded documents.\"\n\n"
        f"Context:\n{context_str}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )

    # Use gemini-3.1-flash-lite as the default for fast low-latency responses,
    # with an automatic fallback if temporarily rate-limited.
    model_name = os.getenv("LLM_MODEL", "gemini-3.1-flash-lite")
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=strict_prompt
        )
    except Exception:
        fallback_model = "gemini-3.5-flash-lite" if model_name != "gemini-3.5-flash-lite" else "gemini-3.5-flash"
        response = client.models.generate_content(
            model=fallback_model,
            contents=strict_prompt
        )

    answer_text = response.text.strip() if response.text else "I don't know based on the uploaded documents."
    return {
        "answer": answer_text,
        "sources": sources
    }
