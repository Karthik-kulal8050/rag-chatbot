# 📄 Document Q&A RAG Chatbot

An end-to-end Retrieval-Augmented Generation (RAG) system that answers questions from PDF documents with cited sources. Built and evaluated on VTU engineering notes (Big Data Analytics, IoT, Parallel Computing), with support for uploading and querying any PDF.

**🔗 Live demo:** https://vtu-rag-chatbot-ke0y.onrender.com/
*(Note: on Render's free tier, PDF upload is memory-constrained — see Known Limitations. Default document Q&A works fully.)*

## Demo for Non-vtu pdf's

https://github.com/Karthik-kulal8050/rag-chatbot/tree/main/demo

## Architecture
PDF → Parse (PyMuPDF) → Clean → Chunk (overlap) → Filter low-value chunks
↓
Embed (MiniLM) → FAISS index
↓
Question → Embed → Hybrid Search (FAISS + BM25) → Rerank (Cross-Encoder)
↓
Top-5 chunks → Prompt → Groq (Llama 3.3) → Answer + Citations


## Key engineering decisions

- **Hybrid retrieval (BM25 + semantic search)**: pure semantic search alone retrieved topically-similar but incorrect content (e.g., confusing IoT and Parallel Computing chunks due to shared vocabulary). Hybrid search combines keyword precision with semantic understanding.
- **Cross-encoder reranking**: hybrid search still let low-value content (question banks, figure captions) rank highly. Reranking the top-20 candidates with a cross-encoder improved **Hit Rate@5 from 72.2% to 100%**.
- **Faithfulness-tuned prompting**: initial prompts allowed the LLM to supplement retrieved context with outside knowledge (a subtle hallucination). Tightening the system prompt improved **answer faithfulness from 70% to 90%**, measured via LLM-as-judge evaluation against an 18-question test set.
- **Chunk filtering**: question-bank pages and figure captions were being embedded and retrieved as if they were explanatory content. Added pattern-based filtering to exclude low-value chunks before indexing.

## Evaluation

Built a custom 18-question test set spanning all 3 subjects, with known correct source pages, to measure retrieval and generation quality quantitatively rather than by inspection.

| Metric | Before | After | Change |
|---|---|---|---|
| Hit Rate@5 (retrieval) | 72.2% (hybrid only) | **100%** (+ reranking) | +27.8 pts |
| Faithfulness (generation) | 70% | **90%** (tuned prompt) | +20 pts |

## Features

- Multi-document support: preloaded VTU notes, or upload any PDF to create an isolated, queryable collection
- Grounded answers with source file + page citations
- Hybrid search (FAISS + BM25) with cross-encoder reranking
- Cached indexing (precomputed index loads instantly instead of re-embedding on every restart)
- REST API (FastAPI) + custom web UI
- Dockerized, deployable

## Known limitations

- **PDF upload is memory-intensive** (live embedding of new documents) and isn't reliable on Render's free 512MB tier. It works fully in the local/Docker environment — see demo video above. The default-document chat, which uses a precomputed index, works reliably on the free-tier deployment.
- **No conversation memory**: each question is treated independently; follow-up questions like "tell me more about that" aren't yet supported. Planned enhancement: query rewriting using chat history.
- **Faithfulness is 90%, not 100%**: the remaining failure case involves the LLM slightly over-elaborating on list-style content (e.g., definitions not explicitly in the source). Documented via LLM-as-judge evaluation rather than assumed.
- Retrieval quality depends on PDF text quality; scanned/image-only pages are currently skipped (no OCR yet).

## Tech stack

**Backend:** Python, FastAPI, PyMuPDF, FAISS, rank-bm25, sentence-transformers, Groq (Llama 3.3)
**Frontend:** Vanilla HTML/CSS/JS
**Infra:** Docker, Render

## Run locally

```bash
git clone <your-repo-url>
cd vtu-rag-chatbot
pip install -r requirements.txt
# add GROQ_API_KEY to .env
python -m app.pipeline   # build the default index
uvicorn app.main:app --reload
```
Or with Docker:
```bash
docker build -t vtu-rag-chatbot .
docker run -p 8000:8000 --env-file .env vtu-rag-chatbot
```

## Evaluation methodology

See `eval/evaluate.py`. Retrieval is measured via Hit Rate@5 against a hand-built test set with known source pages. Faithfulness is measured via LLM-as-judge: a separate prompt checks whether every claim in the generated answer is supported by the retrieved context.