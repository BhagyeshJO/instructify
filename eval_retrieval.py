# eval_retrieval.py
import chromadb
from chromadb.utils import embedding_functions

# same setup as your app
client = chromadb.PersistentClient(path="./chroma")
ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

def get_collection(user_id: str):
    return client.get_or_create_collection(name=f"user_{user_id}", embedding_function=ef)

# 👉 Edit these to match your PDF. Use pages you verified from the document.
EVAL = [
    {"q": "How long is the warranty?", "expected_pages": [3]},
    {"q": "How to request an RMA?",   "expected_pages": [7]},
    # add more as you learn your doc
]

def evaluate(user_id="demo-user-v2", k=5):
    col = get_collection(user_id)
    total = len(EVAL)
    hits = 0
    results_detail = []
    for item in EVAL:
        res = col.query(query_texts=[item["q"]], n_results=k)
        pages = [m["page"] for m in (res["metadatas"][0] if res["metadatas"] else [])]
        hit = any(p in item["expected_pages"] for p in pages)
        hits += int(hit)
        results_detail.append({
            "q": item["q"],
            "expected": item["expected_pages"],
            "got_pages": pages,
            "hit@k": hit
        })
    print(f"\nHit@{k}: {hits}/{total} = {hits/total:.2f}")
    print("\nDetails:")
    for r in results_detail:
        print(f"- Q: {r['q']}\n  expected: {r['expected']}  got: {r['got_pages']}  hit@{k}: {r['hit@k']}")
    return hits, total

if __name__ == "__main__":
    evaluate()
