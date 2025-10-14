# chroma_check.py
import chromadb
from chromadb.utils import embedding_functions

USER = "demo-user-v2"  # ← use the exact user_id you indexed with

client = chromadb.PersistentClient(path="./chroma")
ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
col = client.get_or_create_collection(name=f"user_{USER}", embedding_function=ef)

# 1) Count how many vectors are stored
try:
    count = col.count()
except Exception:
    count = "unknown"
print("Collection:", f"user_{USER}")
print("Count:", count)

# 2) Peek at a few items (metadatas include pages)
peek = col.get(limit=3)
print("\nPeek (first 3):")
for d, m in zip(peek.get("documents", []), peek.get("metadatas", [])):
    print(f"- page={m.get('page')} title={m.get('title')} len={len(d)}")

# 3) Sample query → what pages do we get?
q = "How long is the warranty?"
res = col.query(query_texts=[q], n_results=5)
pages = [m["page"] for m in (res["metadatas"][0] if res.get("metadatas") else [])]
print("\nSample query pages:", pages)
