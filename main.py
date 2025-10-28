# -*- coding: utf-8 -*-
# main.py — DocuQuery with synthesized answers, de-duped hits, and a streamlined UI

import os
import re
import uuid
import shutil
import traceback
from typing import List, Dict, Any, Optional

import pdfplumber
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel

import chromadb
from chromadb.utils import embedding_functions
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# ----------------------------
# Config
# ----------------------------
BGE_NAME = "BAAI/bge-base-en-v1.5"
CHROMA_DIR = "./chroma"
STORAGE_DIR = "./storage"

os.makedirs(STORAGE_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)

# ----------------------------
# FastAPI setup
# ----------------------------
app = FastAPI(title="DocuQuery", version="0.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten for prod if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(Exception)
async def all_exception_handler(request, exc):
    print("UNHANDLED ERROR:\n", traceback.format_exc())
    return JSONResponse(status_code=500, content={"detail": f"{type(exc).__name__}: {exc}"})

# ----------------------------
# Model + Chroma setup
# ----------------------------
bge_model = SentenceTransformer(BGE_NAME, device="cpu")

client = chromadb.PersistentClient(
    path=CHROMA_DIR,
    settings=Settings(anonymized_telemetry=False, allow_reset=True),
)

ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=BGE_NAME)

def get_collection(user_id: str):
    """Get or create a per-user collection; reset if metadata is incompatible."""
    try:
        return client.get_or_create_collection(
            name=f"user_{user_id}",
            embedding_function=ef,
        )
    except KeyError:
        client.reset()
        return client.get_or_create_collection(
            name=f"user_{user_id}",
            embedding_function=ef,
        )

# ----------------------------
# Text cleaning & utils
# ----------------------------
_CID_RE = re.compile(r"\(cid:\d+\)")
_WS_RE = re.compile(r"\s+")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

def clean_text(t: str) -> str:
    """Remove PDF artifacts and normalize spaces/punctuation."""
    if not t:
        return ""
    t = _CID_RE.sub("", t)
    t = t.replace(" .", ".").replace(" ,", ",")
    t = _WS_RE.sub(" ", t).strip()
    return t

def split_sentences(t: str) -> List[str]:
    t = clean_text(t)
    parts = _SENT_SPLIT_RE.split(t)
    return [p.strip() for p in parts if p.strip()]

def analyze_pdf(path: str, min_chars: int = 80) -> Dict[str, Any]:
    """Basic sanity check for each page."""
    per_page, flagged = [], []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = clean_text(page.extract_text() or "")
            chars = len(text)
            if chars < min_chars:
                flagged.append(i)
            per_page.append({"page": i, "chars": chars, "snippet": text[:160]})
    return {
        "pages_total": len(per_page),
        "pages_with_text": sum(1 for p in per_page if p["chars"] >= min_chars),
        "flagged_pages": flagged,
        "preview": per_page[:5],
    }

def chunk_text(text: str, max_words: int = 220, overlap: int = 60):
    """Split text into overlapping word windows."""
    words = clean_text(text).split()
    if not words:
        return
    step = max(1, max_words - overlap)
    for i in range(0, len(words), step):
        yield " ".join(words[i:i + max_words])

def embed_query_bge(q: str) -> List[float]:
    instr = "Represent this sentence for searching relevant passages: "
    emb = bge_model.encode(instr + q, convert_to_tensor=False, normalize_embeddings=True)
    return emb.tolist() if hasattr(emb, "tolist") else list(emb)

def synthesize_answer(question: str, hits: List[Dict[str, Any]], max_sentences: int = 3) -> Dict[str, Any]:
    """
    Lightweight extractive synthesis:
    - score sentences by overlap with question tokens
    - pick top distinct sentences from highest-scoring hits
    - return answer + compact citations
    """
    if not hits:
        return {"answer": "", "citations": []}

    q_tokens = set(re.findall(r"\b\w+\b", question.lower()))
    candidates = []  # (score, sentence, page, title)

    for h in hits:
        snippet = clean_text(h.get("snippet", ""))
        for sent in split_sentences(snippet):
            toks = set(re.findall(r"\b\w+\b", sent.lower()))
            if not toks:
                continue
            overlap = len(q_tokens & toks)
            score = overlap / (1 + len(toks))  # simple overlap / length bias
            if overlap > 0 or len(candidates) < 4:
                candidates.append((score, sent, h.get("page"), h.get("title")))

    candidates.sort(key=lambda x: x[0], reverse=True)
    picked: List[tuple] = []
    seen = set()
    for score, sent, page, title in candidates:
        norm = sent.lower()
        if norm in seen:
            continue
        picked.append((sent, page, title))
        seen.add(norm)
        if len(picked) >= max_sentences:
            break

    if not picked:
        picked = [(clean_text(hits[0]["snippet"]), hits[0].get("page"), hits[0].get("title"))]

    answer = " ".join(s for s, _, _ in picked)
    citations = [{"page": p, "title": t} for _, p, t in picked]
    return {"answer": answer, "citations": citations}

def dedupe_hits(hits: List[Dict[str, Any]], head_len: int = 200) -> List[Dict[str, Any]]:
    """Drop near-duplicate hits (same page + same normalized snippet head)."""
    seen = set()
    out: List[Dict[str, Any]] = []
    for h in sorted(hits, key=lambda x: x["distance"]):  # best first
        key = (h.get("page"), clean_text(h.get("snippet", ""))[:head_len])
        if key in seen:
            continue
        seen.add(key)
        out.append(h)
    return out

# ----------------------------
# Routes
# ----------------------------
@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def root():
    return {"message": "DocuQuery API. See /docs"}

# Upload -> parse -> index
@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    try:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are supported.")

        dest_path = os.path.join(STORAGE_DIR, file.filename)
        with open(dest_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        report = analyze_pdf(dest_path)

        user_id = "demo-user-bge-v2"
        doc_id = os.path.splitext(file.filename)[0]
        title = file.filename

        pages = []
        with pdfplumber.open(dest_path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                text = clean_text(page.extract_text() or "")
                if not text:
                    continue
                for chunk in chunk_text(text):
                    pages.append({"doc_id": doc_id, "title": title, "page": i, "text": chunk})

        if not pages:
            raise HTTPException(status_code=400, detail="No extractable text found; PDF may be image-only (OCR required).")

        col = get_collection(user_id)
        ids = [f"{doc_id}-p{p['page']}-{idx}-{uuid.uuid4().hex[:6]}" for idx, p in enumerate(pages)]
        docs = [p["text"] for p in pages]
        metas = [{"doc_id": p["doc_id"], "title": p["title"], "page": p["page"]} for p in pages]
        col.add(ids=ids, documents=docs, metadatas=metas)

        return {"ok": True, "saved_as": dest_path, **report, "indexed_chunks": len(docs)}

    except HTTPException:
        raise
    except Exception as e:
        print("UPLOAD ERROR:\n", traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"upload failed: {type(e).__name__}: {e}")

# Ask -> retrieve (+ synthesis)
class AskRequest(BaseModel):
    question: str
    top_k: int = 5
    doc_id: Optional[str] = None
    max_distance: Optional[float] = None  # e.g., 0.85 to keep only strong matches

@app.post("/ask")
async def ask(req: AskRequest):
    try:
        user_id = "demo-user-bge-v2"
        col = get_collection(user_id)

        q_emb = embed_query_bge(req.question)
        where = {"doc_id": req.doc_id} if req.doc_id else None

        # fetch a few extra to allow de-duplication and distance filtering
        raw = col.query(query_embeddings=[q_emb], n_results=max(req.top_k, 10), where=where)

        hits: List[Dict[str, Any]] = []
        if raw.get("documents"):
            for doc, meta, dist in zip(raw["documents"][0], raw["metadatas"][0], raw["distances"][0]):
                hit = {
                    "snippet": clean_text(doc)[:800],
                    "doc_id": meta.get("doc_id"),
                    "title": meta.get("title"),
                    "page": meta.get("page"),
                    "distance": float(dist),
                }
                hits.append(hit)

        # optional distance threshold
        if req.max_distance is not None:
            hits = [h for h in hits if h["distance"] <= float(req.max_distance)]

        # de-duplicate and keep top_k
        hits = dedupe_hits(hits)[:req.top_k]

        syn = synthesize_answer(req.question, hits, max_sentences=3)
        return {
            "question": req.question,
            "answer": syn["answer"],
            "citations": syn["citations"],
            "results": hits,
        }

    except Exception as e:
        print("ASK ERROR:\n", traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"ask failed: {type(e).__name__}: {e}")

# List what’s indexed
@app.get("/docs")
def list_docs():
    user_id = "demo-user-bge-v2"
    col = get_collection(user_id)
    res = col.get(include=["metadatas"], limit=2000)
    counts = {}
    for meta in (res.get("metadatas") or []):
        if meta and meta.get("doc_id"):
            counts[meta["doc_id"]] = counts.get(meta["doc_id"], 0) + 1
    return {"documents": [{"doc_id": k, "chunks": v} for k, v in sorted(counts.items())]}

# Reset the vector store
@app.post("/reset")
def reset():
    try:
        client.reset()
        return {"ok": True, "message": "Chroma store cleared."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"reset failed: {type(e).__name__}: {e}")

# ----------------------------
# Minimal UI
# ----------------------------
from fastapi.responses import HTMLResponse

from fastapi.responses import HTMLResponse

@app.get("/ui", response_class=HTMLResponse)
def ui():
    return r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>DocuQuery</title>
<style>
  :root{
    --bg:#0b0f14; --panel:#0f141b; --panel-2:#121924;
    --border:#1b2330; --text:#eaf0f7; --muted:#9fb0c5;
    --brand:#6ea8fe; --brand-2:#3b82f6; --ok:#22c55e; --err:#ef4444;
  }
  [data-theme="light"]{
    --bg:#f5f7fb; --panel:#ffffff; --panel-2:#f7f9fc;
    --border:#dfe7f1; --text:#111827; --muted:#52627a;
    --brand:#2563eb; --brand-2:#1d4ed8;
  }

  *{box-sizing:border-box}
  html,body{height:100%}
  body{
    margin:0; font:16px/1.5 Inter,system-ui,-apple-system,Segoe UI,Roboto,Arial;
    color:var(--text); background:var(--bg);
    /* soft gradient backdrop */
    background-image:
      radial-gradient(60rem 40rem at 10% -20%, rgba(59,130,246,.22), transparent 60%),
      radial-gradient(80rem 60rem at 120% 10%, rgba(110,168,254,.18), transparent 60%);
  }
  .wrap{max-width:1100px; margin:0 auto; padding:28px 22px}
  .topbar{display:flex; align-items:center; justify-content:space-between; margin-bottom:18px}
  .logo{display:flex; gap:10px; align-items:center; font-weight:800}
  .logo .badge{width:36px; height:36px; display:grid; place-items:center;
    background:linear-gradient(135deg,var(--brand),var(--brand-2)); border-radius:10px; color:#fff; font-weight:900}
  .pill{color:var(--muted); font-size:13px}
  .actions{display:flex; gap:8px; align-items:center}
  .btn{
    background:linear-gradient(180deg,var(--brand),var(--brand-2)); color:#fff; border:0;
    border-radius:12px; padding:10px 14px; cursor:pointer; font-weight:600;
    box-shadow:0 6px 20px rgba(59,130,246,.25)
  }
  .btn:disabled{opacity:.6; cursor:not-allowed}
  .ghost{background:transparent; color:var(--muted); border:1px solid var(--border)}
  .grid{display:grid; grid-template-columns:1.05fr 1.5fr; gap:18px; align-items:start}

  .card{background:var(--panel); border:1px solid var(--border); border-radius:16px; padding:16px}
  .title{font-weight:700; margin-bottom:10px}
  .muted{color:var(--muted)}
  .row{display:flex; gap:10px}
  input[type="file"], input[type="text"]{
    width:100%; padding:12px 14px; border-radius:12px; border:1px solid var(--border);
    background:var(--panel-2); color:var(--text); outline:none
  }
  input[type="text"]::placeholder{color:var(--muted)}
  .hint{font-size:13px; margin-top:10px}

  /* Dropzone */
  .drop{border:2px dashed var(--border); border-radius:14px; padding:18px; text-align:center; background:var(--panel-2)}
  .drop.drag{border-color:var(--brand); background:rgba(110,168,254,.08)}

  /* Example chips */
  .chips{display:flex; flex-wrap:wrap; gap:8px; margin-top:10px}
  .chip{padding:6px 10px; border:1px solid var(--border); border-radius:999px; color:var(--muted); cursor:pointer; background:transparent}
  .chip:hover{border-color:var(--brand); color:var(--text)}

  /* Answer & hits */
  .answer{background:var(--panel-2); border:1px solid var(--border); border-radius:14px; padding:14px; white-space:pre-wrap}
  .cits{display:flex; gap:8px; flex-wrap:wrap; margin:10px 0 2px}
  .cit{font-size:12px; color:var(--muted); border:1px solid var(--border); border-radius:999px; padding:4px 8px; background:transparent}

  .hits{display:grid; gap:10px; margin-top:10px}
  .hit{border:1px solid var(--border); border-radius:14px; overflow:hidden; background:var(--panel-2)}
  .hit .hd{display:flex; align-items:center; justify-content:space-between; padding:10px 12px}
  .hit .pg{font-weight:700}
  .hit .bd{display:none; padding:0 12px 12px}
  .hit.open .bd{display:block}
  .link{color:var(--brand)}

  /* Footer */
  .footer{margin-top:20px; display:flex; gap:10px; align-items:center; color:var(--muted); font-size:13px}
  .toggle{background:transparent; border:1px solid var(--border); color:var(--muted); padding:8px 10px; border-radius:10px; cursor:pointer}
</style>
</head>
<body>
  <div class="wrap" id="app">
    <div class="topbar">
      <div class="logo">
        <div class="badge">DQ</div>
        <div>DocuQuery</div>
      </div>
      <div class="actions">
        <div class="pill">Local • Upload → Ask • Cited evidence</div>
        <button id="theme" class="toggle" title="Toggle theme">Toggle theme</button>
      </div>
    </div>

    <div class="grid">
      <!-- Upload -->
      <div class="card">
        <div class="title">Upload a PDF</div>
        <div id="drop" class="drop">
          <div class="muted">Drag & drop PDF here, or use the button below.</div>
        </div>
        <div style="height:10px"></div>
        <div class="row">
          <input id="file" type="file" accept="application/pdf"/>
          <button id="btnUpload" class="btn">Upload</button>
        </div>
        <div id="upStatus" class="hint muted"></div>
        <div class="hint muted">Tip: ask focused questions like <em>“Who is responsible for implementation?”</em></div>
      </div>

      <!-- Ask -->
      <div class="card">
        <div class="title">Ask a question</div>
        <div class="row">
          <input id="q" type="text" placeholder="e.g., What is the main purpose of this policy?"/>
          <button id="btnAsk" class="btn">Ask</button>
        </div>
        <div class="chips">
          <button class="chip" data-q="List the responsibilities defined.">Responsibilities?</button>
          <button class="chip" data-q="Who must form the EHS committee?">EHS committee?</button>
          <button class="chip" data-q="What are the emergency provisions mentioned?">Emergency?</button>
        </div>

        <div id="answerBox" style="margin-top:14px"></div>
        <div id="hitsBox" class="hits"></div>
      </div>
    </div>

    <div class="footer">
      <span>Need higher precision? Try a smaller max distance in the API (e.g. <code>0.85</code>).</span>
    </div>
  </div>

<script>
const $ = (id) => document.getElementById(id);
const api = (p) => new URL(p, location.origin).toString();

// theme toggle
(function initTheme(){
  const saved = localStorage.getItem("dq.theme") || "dark";
  if(saved === "light") document.documentElement.setAttribute("data-theme","light");
  $("theme").addEventListener("click", ()=>{
    const cur = document.documentElement.getAttribute("data-theme")==="light" ? "dark" : "light";
    document.documentElement.setAttribute("data-theme",cur==="light"?"light":"");
    localStorage.setItem("dq.theme", cur);
  });
})();

function cleanText(t){ return (t||"").replace(/\(cid:\d+\)/g,"").replace(/\s+/g," ").trim(); }

function render(data){
  const box = $("answerBox");
  const hitsBox = $("hitsBox");
  box.innerHTML = "";
  hitsBox.innerHTML = "";

  const ans = (data?.answer||"").trim();
  if(ans){
    const a = document.createElement("div");
    a.className = "answer";
    a.textContent = ans;
    box.appendChild(a);

    const cits = data?.citations || [];
    if(cits.length){
      const row = document.createElement("div");
      row.className = "cits";
      cits.slice(0,3).forEach(c=>{
        const s = document.createElement("span");
        s.className = "cit";
        s.textContent = `p. ${c.page ?? "?"} · ${c.title || "document"}`;
        row.appendChild(s);
      });
      box.appendChild(row);
    }
  }else{
    box.innerHTML = '<div class="muted">No answer found. Try rephrasing your question.</div>';
  }

  const hits = data?.results || [];
  hits.forEach((h,i)=>{
    const card = document.createElement("div");
    card.className = "hit" + (i===0?" open":"");
    const hd = document.createElement("div");
    hd.className = "hd";
    hd.innerHTML = `<div><span class="pg">Page ${h.page ?? "?"}</span> · <span class="muted">${h.title || h.doc_id || "document"}</span></div>
                    <a href="#" class="link">${i===0?'Hide':'Show'} details ▸</a>`;
    const bd = document.createElement("div");
    bd.className = "bd";
    bd.textContent = cleanText(h.snippet);
    hd.querySelector("a").addEventListener("click",(ev)=>{ ev.preventDefault(); card.classList.toggle("open"); });
    card.appendChild(hd); card.appendChild(bd); $("hitsBox").appendChild(card);
  });
}

// drag & drop
(function initDrop(){
  const drop = $("drop");
  drop.addEventListener("dragover", e=>{ e.preventDefault(); drop.classList.add("drag"); });
  drop.addEventListener("dragleave", ()=> drop.classList.remove("drag"));
  drop.addEventListener("drop", e=>{
    e.preventDefault(); drop.classList.remove("drag");
    if(e.dataTransfer.files && e.dataTransfer.files[0]){
      $("file").files = e.dataTransfer.files;
    }
  });
})();

async function doUpload(){
  const f = $("file").files[0];
  if(!f){ alert("Choose a PDF"); return; }
  $("btnUpload").disabled = true;
  $("upStatus").textContent = "Uploading…";
  const form = new FormData(); form.append("file", f, f.name);
  try{
    const r = await fetch(api("/upload"), { method:"POST", body:form });
    const txt = await r.text(); let data; try{ data = JSON.parse(txt); } catch{ data = {raw:txt}; }
    if(r.ok){
      $("upStatus").innerHTML = `<span style="color:var(--ok)">Uploaded.</span> Indexed chunks: <strong>${data?.indexed_chunks ?? "?"}</strong>`;
    }else{
      $("upStatus").innerHTML = `<span style="color:var(--err)">Upload failed:</span> ${data?.detail || txt}`;
    }
  }catch(e){
    $("upStatus").innerHTML = `<span style="color:var(--err)">${e}</span>`;
  }finally{ $("btnUpload").disabled = false; }
}

async function doAsk(qOverride){
  const q = (qOverride || $("q").value.trim());
  if(!q){ $("q").focus(); return; }
  $("btnAsk").disabled = true;
  $("answerBox").innerHTML = '<div class="muted">Thinking…</div>';
  $("hitsBox").innerHTML = '';
  try{
    const payload = { question:q, top_k:5 }; // adjust here if you want stricter by default
    const r = await fetch(api("/ask"), { method:"POST", headers:{ "content-type":"application/json" }, body:JSON.stringify(payload) });
    const data = await r.json();
    if(!r.ok) throw new Error(data?.detail || r.statusText);
    render(data);
  }catch(e){
    $("answerBox").innerHTML = `<div style="color:var(--err)">${e}</div>`;
  }finally{ $("btnAsk").disabled = false; }
}

$("btnUpload").addEventListener("click", doUpload);
$("btnAsk").addEventListener("click", ()=>doAsk());
$("q").addEventListener("keydown", e=>{ if(e.key==="Enter") doAsk(); });
document.querySelectorAll(".chip").forEach(ch=> ch.addEventListener("click", ()=>{ $("q").value = ch.dataset.q; doAsk(ch.dataset.q); }));
</script>
</body>
</html>
"""
