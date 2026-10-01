from pathlib import Path
from .parse import parse_pdf
from .chunk import chunk_pages
from .index import build_index
from .hybrid import build_bm25, hybrid_search
from .rerank import rerank

test_set = [
    {"q": "What is Big Data Analytics? Explain the different classifications of analytics.", "file": "bda_module1.pdf", "pages": [25]},
    {"q": "Differentiate between Descriptive, Predictive and Prescriptive Analytics.", "file": "bda_module1.pdf", "pages": [31]},
    {"q": "Explain the importance of Big Data Analytics in business and decision making.", "file": "bda_module1.pdf", "pages": [15]},
    {"q": "Explain the architecture of HDFS and describe the roles of NameNode and DataNodes.", "file": "bda_module2.pdf", "pages": [14]},
    {"q": "Explain how the MapReduce framework works, including the Map and Reduce phases.", "file": "bda_module2.pdf", "pages": [25,26,27,28,29,30]},
    {"q": "What is YARN? Explain the role of ResourceManager and ApplicationManager in Hadoop YARN.", "file": "bda_module2.pdf", "pages": [37]},

    {"q": "Explain the logical design of IoT and describe the functional blocks of an IoT system.", "file": "iot_module1.pdf", "pages": [7]},
    {"q": "What are the major IoT enabling technologies? Explain the role of Wireless Sensor Networks in IoT.", "file": "iot_module1.pdf", "pages": [13]},
    {"q": "Explain IoT Level 1 and IoT Level 2 with suitable examples.", "file": "iot_module1.pdf", "pages": [17, 18]},
    {"q": "Differentiate between IoT and M2M based on communication protocols, hardware/software emphasis, data storage and applications.", "file": "iot_module2.pdf", "pages": [2, 3]},
    {"q": "Explain Software Defined Networking (SDN) and its separation of the control plane and data plane.", "file": "iot_module2.pdf", "pages": [4]},
    {"q": "What is NETCONF? Explain the role of YANG in NETCONF-based IoT system management.", "file": "iot_module2.pdf", "pages": list(range(12, 25))},

    {"q": "What is parallel programming? Explain data parallelism and task parallelism with examples.", "file": "pc_module1.pdf", "pages": [1, 2]},
    {"q": "Explain SIMD architecture and how it performs vector addition in parallel.", "file": "pc_module1.pdf", "pages": [3]},
    {"q": "Explain the cache coherence problem and describe snooping cache coherence.", "file": "pc_module1.pdf", "pages": [18]},
    {"q": "Explain GPU programming and the role of CPU host, GPU device and GPU threads.", "file": "pc_module2.pdf", "pages": [1]},
    {"q": "Define speedup and efficiency in MIMD systems. Explain their relationship with an example.", "file": "pc_module2.pdf", "pages": list(range(3,11))},
    {"q": "State and explain Amdahl's Law. How does the serial portion of a program limit the maximum speedup?", "file": "pc_module2.pdf", "pages": [7]},
]

def build_pipeline():
    all_chunks = []
    for pdf in Path("data/default").glob("*pdf"):
        all_chunks.extend(chunk_pages(parse_pdf(pdf)))
    index = build_index(all_chunks)
    bm25 = build_bm25(all_chunks)
    return all_chunks, index, bm25

def is_hit(retrieved_chunks, expected_file, expected_pages):
    for c in retrieved_chunks:
        if c["source"] == expected_file and c["page"] in expected_pages:
            return True
        return False


def evaluate(all_chunks, index, bm25, use_rerank=True, k=5):
    hits = 0
    results = []
    for item in test_set:
        candidates = hybrid_search(item["q"], index, bm25, all_chunks, k=20 if use_rerank else k)
        final = rerank(item["q"], candidates, k=k)if use_rerank else candidates[:k]
        hit = is_hit(final, item["file"], item["pages"])
        hits += hit
        results.append((item["q"][:50], hit))
    hit_rate = hits /len(test_set)
    return hit_rate, results

from .generate import generate_answer, client
def check_faitfulness(question, chunks, answer):
    context = "\n\n".join(c["text"] for c in chunks)
    judge_prompt = f"""You are a strict fact-checker. Given the CONTEXT and an ANSWER, determine if every claim in the ANSWER is actually supported by the CONTEXT.
    Reply in this format:
    Verdict: FAITHFUL or UNFAITHFUL
    Reason: (one sentence explaining what, if anything, isn't supported by the context

{context}

Answer:
{answer}

Reply with ONLY one word: "FAITHFUL" if all claims are supported, or "UNFAITHFUL" if the answer contains information not found in the context."""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": judge_prompt}],
        temperature=0,
    )
    return response.choices[0].message.content.strip()

if __name__ == "__main__":
    all_chunks, index, bm25 = build_pipeline()
    faithful_count = 0

    for item in test_set[:10]:
        candidates = hybrid_search(item["q"], index, bm25, all_chunks, k=20)
        top_chunks = rerank(item["q"], candidates, k=5)
        answer = generate_answer(item["q"], top_chunks)
        verdict = check_faitfulness(item["q"], top_chunks, answer)
        is_faithful = "UNFAITHFUL" not in verdict.upper()
        faithful_count += is_faithful
        print(("HIT" if is_faithful else "NOT HIT"), item["q"][:50], "->", verdict)
    print(f"\nFaithfulness: {faithful_count}/10 = {faithful_count/10:.0%}")
# def debug_miss(all_chunks, index, bm25, question, k=5):
#     candidates = hybrid_search(question, index, bm25, all_chunks, k=20)
#     final = rerank(question, candidates, k=k)
#     for c in final:
#         print(f"{c['source']} p.{c['page']} | {c['text'][:150]}")
#     print("---")

# if __name__ == "__main__":
#     all_chunks, index, bm25 = build_pipeline()
#     debug_miss(all_chunks, index, bm25, "Explain how the MapReduce framework works, including the Map and Reduce phases.")
#     debug_miss(all_chunks, index, bm25, "What is NETCONF? Explain the role of YANG in NETCONF-based IoT system management.")
#     debug_miss(all_chunks, index, bm25, "Define speedup and efficiency in MIMD systems. Explain their relationship with an example.")


    