import numpy as np
from rank_bm25 import BM25Okapi
from pathlib import Path
from app.parse import parse_pdf
from .chunk import chunk_pages
from app.index import build_index, get_model 
import faiss
from app.rerank import rerank


def build_bm25(chunks):
    tokenized = [c["text"].lower().split() for c in chunks]
    return BM25Okapi(tokenized)

def normalize(scores):
    scores = np.array(scores, dtype="float32")
    if scores.max() - scores.min() < 1e-9:
        return np.zeros_like(scores)
    return (scores - scores.min()) / (scores.max() - scores.min())

def hybrid_search(query,index, bm25, chunks, k=5, alpha = 0.6):
    q_emb = get_model.encode([query], normalize_embeddings=True).astype("float32")
    sem_scores, sem_ids = index.search(q_emb,len(chunks))    
    sem_full = np.zeros(len(chunks), dtype="float32")
    for score, i in zip(sem_scores[0], sem_ids[0]):
        sem_full[i] = score

    bm25_scores = bm25.get_scores(query.lower().split())
    combined = alpha * normalize(sem_full) + (1 - alpha) * normalize(bm25_scores)
    top_ids = np.argsort(combined)[::-1][:k]
    return [{**chunks[i], "score" : float(combined[i])} for i in top_ids]


if __name__ == "__main__":
    all_chunks = []
    for pdf in Path("data").glob("*.pdf"):
        all_chunks.extend(chunk_pages(parse_pdf(pdf)))

    index = build_index(all_chunks)
    bm25 = build_bm25(all_chunks)

    query = "What is data quality and its dimensions?"
    candidates = hybrid_search(query, index, bm25, all_chunks, k=20)   # wider net
    final = rerank(query, candidates, k=5)

    for r in final:
        print(f"[{r['rerank_score']:.2f}] {r['source']} p.{r['page']}")
        print(r["text"], "\n")