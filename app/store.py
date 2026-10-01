import pickle
import faiss
from pathlib import Path

STORAGE_DIR = Path("storage")
STORAGE_DIR.mkdir(exist_ok=True)

def save_collection(collection_id, chunks, index, bm25):
    folder = STORAGE_DIR / collection_id
    folder.mkdir(exist_ok=True)

    faiss.write_index(index, str(folder / "faiss.index"))
    with open(folder / "chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
    with open(folder / "bm25.pkl", "wb") as f:
        pickle.dump(bm25, f)

def load_collection(collection_id):
    folder = STORAGE_DIR / collection_id
    if not folder.exists():
        return None

    index = faiss.read_index(str(folder / "faiss.index"))
    with open(folder / "chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    with open(folder / "bm25.pkl", "rb") as f:
        bm25 = pickle.load(f)
    return chunks, index, bm25

def collection_exists(collection_id):
    return (STORAGE_DIR / collection_id).exists()