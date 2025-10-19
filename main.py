# main.py
from fastapi import FastAPI, UploadFile, File
import os, shutil
import chromadb
from chromadb.utils import embedding_functions
from pydantic import BaseModel


import pdfplumber
from sentence_transformers import SentenceTransformer

BGE_NAME = "BAAI/bge-base-en-v1.5"
bge_model = SentenceTransformer(BGE_NAME, device="cpu")

app = FastAPI()
os.makedirs("storage", exist_ok=True)

@app.get("/health")
def health():
    return {"status": "ok"}

def analyze_pdf(path: str, min_chars: int = 80):
    per_page = []
    flagged = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = (page.extract_text() or "").strip()
            chars = len(text)
            if chars < min_chars:
                flagged.append(i)
            # we’ll also save a short snippet for quick sanity checks
            per_page.append({"page": i, "chars": chars, "snippet": text[:160].replace("\n"," ")})
    return {
        "pages_total": len(per_page),
        "pages_with_text": sum(1 for p in per_page if p["chars"] >= min_chars),
        "flagged_pages": flagged,
        "preview": per_page[:5]  # first 5-page preview
    }

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    dest_path = os.path.join("storage", file.filename)
    with open(dest_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    report = analyze_pdf(dest_path)

    # ---- NEW: index into Chroma ----
    user_id = "demo-user-bge-v2"          # later we’ll pass real user ids; for now fixed
    doc_id = os.path.splitext(file.filename)[0]
    title = file.filename

    # extract per-page text again (simple + clear)
    pages = []
    with pdfplumber.open(dest_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = (page.extract_text() or "").strip()
            if not text:
                continue
            # chunk within each page to keep page metadata
            for chunk in chunk_text(text):
                pages.append({
                    "doc_id": doc_id,
                    "title": title,
                    "page": i,
                    "text": chunk
                })

    # add to user collection
    col = get_collection(user_id)
    ids = [f"{doc_id}-p{p['page']}-{idx}" for idx, p in enumerate(pages)]
    docs = [p["text"] for p in pages]
    metas = [{"doc_id": p["doc_id"], "title": p["title"], "page": p["page"]} for p in pages]
    if docs:
        col.add(ids=ids, documents=docs, metadatas=metas)

    return {
        "ok": True,
        "saved_as": dest_path,
        **report,
        "indexed_chunks": len(docs)
    }


# Chroma client (persists in ./chroma folder)
CHROMA_DIR = "./chroma"
client = chromadb.PersistentClient(path=CHROMA_DIR)

# CPU-friendly embedding model
ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=BGE_NAME
)

def get_collection(user_id: str):
    # one collection per user keeps data isolated
    return client.get_or_create_collection(
        name=f"user_{user_id}",
        embedding_function=ef
    )

def chunk_text(text: str, max_words=220, overlap=60):
    words = text.split()
    step = max_words - overlap
    for i in range(0, len(words), step):
        yield " ".join(words[i:i+max_words])


class AskRequest(BaseModel):
    question: str
    # for later we could add: user_id, doc_id filter, etc.

def embed_query_bge(q: str):
    instr = "Represent this sentence for searching relevant passages: "
    return bge_model.encode(instr + q, convert_to_tensor=False, normalize_embeddings=True)

@app.post("/ask")
async def ask(req: AskRequest):
    user_id = "demo-user-bge-v2"   # new namespace (so we don’t mix with old vectors)
    col = get_collection(user_id)
    q_emb = embed_query_bge(req.question)
    res = col.query(query_embeddings=[q_emb], n_results=5)
    hits = []
    if res["documents"]:
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            hits.append({
                "snippet": doc[:300],
                "doc_id": meta["doc_id"],
                "title": meta["title"],
                "page": meta["page"],
                "distance": dist
            })
    return {"question": req.question, "results": hits}
@app.get("/")
def root():
    return {"message": "DocuQuery API. See /docs"}
