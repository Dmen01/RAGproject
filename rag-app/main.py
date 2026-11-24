from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from rag_engine import index_document, list_documents, delete_document, query_rag

# Initialize directory paths
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="Document Search & Knowledge Base API",
    description="Local document retrieval and grounded Q&A service"
)

# Serve frontend assets from static directory
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class AskRequest(BaseModel):
    question: str


@app.get("/")
def serve_index():
    """Serve the single-page application UI."""
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Frontend file static/index.html not found.")
    return FileResponse(index_file)


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload and index a new document (.pdf or .txt).
    Extracts text, splits into vector chunks, and registers document in ChromaDB.
    """
    filename = file.filename or ""
    ext = Path(filename).suffix.lower()

    if ext not in [".pdf", ".txt"]:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file format. Please upload a .pdf or .txt file."
        )

    content = await file.read()
    if not content:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty."
        )

    try:
        result = index_document(filename, content)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to index document: {str(e)}")


@app.get("/api/documents")
def get_documents():
    """List all currently indexed documents with chunk statistics."""
    return list_documents()


@app.delete("/api/documents/{doc_id}")
def delete_doc(doc_id: str):
    """
    Delete a document and purge all associated vector embeddings from ChromaDB.
    """
    success = delete_document(doc_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found.")
    return {"status": "deleted", "doc_id": doc_id}


@app.post("/api/ask")
def ask(req: AskRequest):
    """
    Query the document knowledge base.
    Embeds question, performs vector similarity search, and synthesizes a grounded answer.
    """
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Query prompt cannot be empty.")

    try:
        return query_rag(question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search pipeline error: {str(e)}")
