from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(query, candidates, k=5):
    pairs = [[query, c["text"]] for c in candidates]
    scores = reranker.predict(pairs)
    for c, s in zip(candidates,scores):
        c["rerank_score"] = float(s)
    ranked = sorted(candidates, key = lambda c: c["rerank_score"], reverse=True)
    return ranked[:k]
