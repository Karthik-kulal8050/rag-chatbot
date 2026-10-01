import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def build_prompt(question, chunks):
    context = ""
    for c in chunks:
        context += f"[Source: {c['source']}, Page {c['page']}]\n{c['text']}\n\n"

    system = (
        "You are a helpful study assistant. Answer using the information in the context below. "
        "Base your answer on the context, and avoid adding significant outside facts or definitions "
        "that aren't supported by it. "
        "If the context contains relevant information, answer using it, even if some rephrasing is needed. "
        "Only say \"I don't know based on the provided documents\" if the context truly has nothing relevant to the question. "
        "Always cite the source file and page number for your answer."
    )
    user = f"Context:\n{context}\nQuestion: {question}"
    return system, user

def generate_answer(question, chunks):
    system, user = build_prompt(question, chunks)
    response = client.chat.completions.create(
        model = "openai/gpt-oss-120b",
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature = 0.1,
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    from pathlib import Path
    from app.parse import parse_pdf
    from chunk import chunk_pages
    from app.index import build_index
    from app.hybrid import build_bm25, hybrid_search
    from app.rerank import rerank

    all_chunks = []
    for pdf in Path("data").glob("*.pdf"):
        all_chunks.extend(chunk_pages(parse_pdf(pdf)))

    index = build_index(all_chunks)
    bm25 = build_bm25(all_chunks)

    question = "What is the capital of France"
    candidates = hybrid_search(question, index, bm25, all_chunks, k=20)
    top_chunks = rerank(question, candidates, k=5)

    answer = generate_answer(question, top_chunks)
    print(answer)