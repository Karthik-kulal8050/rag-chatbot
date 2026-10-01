import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer
from app.parse import parse_pdf
from .chunk import chunk_pages


_model = None
def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model

def build_index(chunks):
    texts = [c["text"] for c in chunks]
    embeddings = get_model().encode(texts, normalize_embeddings=True, show_progress_bar=True)
    embeddings = np.array(embeddings, dtype="float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index

def search(query, index, chunks, k=5):
    q = get_model.encode([query], normalize_embeddings=True)
    q = np.array(q, dtype = "float32")
    scores, ids = index.search(q, k)
    results = []
    for score, i in zip(scores[0], ids[0]):
        results.append({**chunks[i], "score": float(score)})
    return results

if __name__ == "__main__":
    all_chunks = []
    for pdf in Path("data").glob("*pdf"):
        all_chunks.extend(chunk_pages(parse_pdf(pdf)))
    index = build_index(all_chunks)
    query = "discuss the shared memory and distributed memory architectures"
    for r in search(query, index, all_chunks):
        print(f"[{r['score']:.2f}] {r['source']} p.{r['page']}")
        print(r["text"][:200], "\n")

    
