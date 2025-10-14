# test_chroma.py
import chromadb
from chromadb.utils import embedding_functions

# 1) start a local persistent DB (folder: ./chroma)
client = chromadb.PersistentClient(path="./chroma")

# 2) small, CPU-friendly embedding function
ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# 3) make/get a collection
col = client.get_or_create_collection(name="demo", embedding_function=ef)

# 4) add a couple of short texts
col.add(
    ids=["a1","a2","a3"],
    documents=[
        "The warranty period is 24 months from purchase.",
        "Contact support at support@example.com for RMA requests.",
        "The sky is blue and the sun is bright."
    ],
    metadatas=[
        {"doc_id":"policy.pdf","page":3},
        {"doc_id":"policy.pdf","page":7},
        {"doc_id":"notes.txt","page":1}
    ]
)

# 5) query by meaning
res = col.query(query_texts=["How long is the warranty?"], n_results=2)
print("\nTop results:\n")
for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
    print(f"[{meta['doc_id']}, p.{meta['page']}] dist={dist:.3f} → {doc}")
