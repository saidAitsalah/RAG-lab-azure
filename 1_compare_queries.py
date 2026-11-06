"""Lab 2: run keyword, vector and hybrid queries on the same question."""
import sys

from common import retrieve

question = " ".join(sys.argv[1:]) or "How do I disable API key authentication on a Foundry resource?"
print(f"Question: {question}\n")
for mode in ("keyword", "vector", "hybrid"):
    print(f"=== {mode.upper()} ===")
    for rank, c in enumerate(retrieve(question, mode=mode, k=3), start=1):
        preview = c["content"][:140].replace("\n", " ")
        print(f"{rank}. score={c['score']:.4f} | {c['title']}\n   {preview}...")
    print()
print("Note: scores are not comparable between modes (BM25, cosine similarity, RRF).")
