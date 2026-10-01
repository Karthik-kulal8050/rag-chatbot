from fastapi import FastAPI,UploadFile, File
from pydantic import BaseModel
from .pipeline import get_collection,ingest_folder
from .hybrid import hybrid_search
from .rerank import rerank
from .generate import generate_answer
import hashlib
import shutil
from pathlib import Path
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse



app = FastAPI(title="Document RAG Chatbot")

_cache = {}

@app.on_event("startup")
def startup():
    _cache["default"] = get_collection("default", "data/default")
    print("Default collection loaded.")

class AskRequest(BaseModel):
    question: str
    collection_id: str = "default"

class AskResponse(BaseModel):
    answer: str
    sources: list

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    if req.collection_id not in _cache:
        return {"anser": "collection not found. Please upload a document first.","sources": []}
    chunks, index, bm25 = _cache[req.collection_id]

    candidates = hybrid_search(req.question, index, bm25, chunks, k=20)
    top_chunks = rerank(req.question, candidates, k=5)
    answer = generate_answer(req.question, top_chunks)

    sources = [{"file": c["source"], "page": c["page"]} for c in top_chunks]
    return {"answer": answer, "sources": sources}

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        return {"error": "Only PDF files are supported."}

    contents = await file.read()

    if len(contents) > 20 * 1024 * 1024:
        return {"error": "File too large. Max 20MB."}

    file_hash = hashlib.md5(contents).hexdigest()[:12]
    collection_id = f"upload_{file_hash}"

    collection_folder = UPLOAD_DIR / collection_id
    collection_folder.mkdir(exist_ok=True)
    file_path = collection_folder / file.filename

    with open(file_path, "wb") as f:
        f.write(contents)

    try:
        chunks, index, bm25 = ingest_folder(str(collection_folder), collection_id)
    except Exception as e:
        shutil.rmtree(collection_folder, ignore_eroors = True)
        return {"error": f"Failed to process PDF: {str(e)}"}

    _cache[collection_id] = (chunks, index, bm25)

    return {
        "collection_id": collection_id,
        "filename": file.filename,
        "chunks_created": len(chunks),
    }

app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.get("/")
def serve_frontend():
    return FileResponse("app/static/index.html")
