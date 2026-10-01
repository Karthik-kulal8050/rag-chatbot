from sentence_transformers import CrossEncoder

_reranker = None

def get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker


def rerank(query, candidates, k=5):
    reranker = get_reranker()
    pairs = [[query, c["text"]] for c in candidates]
    scores = reranker.predict(pairs)
    for c, s in zip(candidates,scores):
        c["rerank_score"] = float(s)
    ranked = sorted(candidates, key = lambda c: c["rerank_score"], reverse=True)
    return ranked[:k]
