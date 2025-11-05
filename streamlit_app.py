"""Streamlit front-end for DocuQuery with a blue & white minimal theme."""
import os
from typing import Any, Dict, List, Optional

import requests
import streamlit as st

API_URL = os.getenv("DOCUQUERY_API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="DocuQuery",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded",
)

PRIMARY_BLUE = "#1E3A8A"
ACCENT_BLUE = "#2563EB"
LIGHT_BACKGROUND = "#F4F7FF"
CARD_BACKGROUND = "#FFFFFF"
MUTED_TEXT = "#5A6A85"

st.markdown(
    f"""
    <style>
        :root {{
            --primary-blue: {PRIMARY_BLUE};
            --accent-blue: {ACCENT_BLUE};
            --light-bg: {LIGHT_BACKGROUND};
            --card-bg: {CARD_BACKGROUND};
            --muted-text: {MUTED_TEXT};
            --border-radius: 18px;
        }}
        .main {{
            background: linear-gradient(180deg, rgba(30,58,138,0.08) 0%, rgba(37,99,235,0.12) 45%, rgba(244,247,255,1) 100%);
        }}
        .block-container {{
            padding-top: 2.2rem;
            padding-bottom: 2.8rem;
        }}
        .dq-card {{
            background: var(--card-bg);
            border-radius: var(--border-radius);
            padding: 1.6rem 1.8rem;
            box-shadow: 0 18px 45px rgba(30, 58, 138, 0.08);
            border: 1px solid rgba(30, 58, 138, 0.08);
        }}
        .dq-header {{
            font-weight: 700;
            margin-bottom: 0.25rem;
            color: var(--primary-blue);
        }}
        .dq-subtle {{
            color: var(--muted-text);
            font-size: 0.9rem;
        }}
        .dq-pill {{
            display: inline-flex;
            align-items: center;
            padding: 0.35rem 0.75rem;
            border-radius: 999px;
            background: rgba(37, 99, 235, 0.12);
            color: var(--accent-blue);
            font-weight: 600;
            font-size: 0.8rem;
        }}
        .dq-button > button {{
            border-radius: 999px !important;
            font-weight: 600 !important;
            letter-spacing: 0.01em;
        }}
        .dq-answer {{
            background: rgba(37, 99, 235, 0.08);
            border-radius: var(--border-radius);
            padding: 1.1rem 1.2rem;
            margin-top: 0.8rem;
            color: #0F172A;
        }}
        .dq-hit {{
            border-left: 4px solid var(--accent-blue);
            background: rgba(255,255,255,0.75);
            padding: 0.9rem 1rem;
            border-radius: 12px;
            margin-bottom: 0.7rem;
            box-shadow: 0 12px 30px rgba(15, 23, 42, 0.06);
        }}
        .dq-hit h4 {{
            margin: 0 0 0.35rem 0;
            color: var(--primary-blue);
        }}
        .dq-hit p {{
            margin: 0;
            color: var(--muted-text);
            font-size: 0.92rem;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## 📘 DocuQuery")
    st.markdown(
        "Upload PDF documents and ask focused questions to retrieve cited answers."
    )
    st.divider()
    st.markdown("### Connection")
    st.markdown(
        f"**API base URL**\n\n`{API_URL}`"
    )
    st.caption(
        "Set the `DOCUQUERY_API_URL` environment variable to point at a remote FastAPI deployment."
    )
    st.divider()
    st.markdown("### Tips")
    st.write(
        "• Upload clean, text-based PDFs for best results.\n"
        "• Ask direct questions (e.g., 'Who is responsible for compliance?').\n"
        "• Use the advanced options to adjust retrieval parameters."
    )

st.markdown(
    """
    <div style="display:flex; align-items:center; gap:1rem; margin-bottom:1.8rem;">
        <div class="dq-pill">Blue & white minimal workspace</div>
        <div>
            <h1 style="margin-bottom:0; color: var(--primary-blue);">DocuQuery Workspace</h1>
            <p style="margin-top:0.3rem; color: var(--muted-text);">Upload PDFs, ask questions, and review supporting evidence.</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

col_upload, col_query = st.columns((1, 1))

upload_status = st.empty()
ask_status = st.empty()
answer_placeholder = st.empty()
hits_placeholder = st.container()


def list_documents() -> List[Dict[str, Any]]:
    try:
        response = requests.get(f"{API_URL}/docs", timeout=10)
        response.raise_for_status()
    except requests.RequestException:
        return []
    data = response.json()
    return data.get("documents", [])


def upload_pdf(file) -> Optional[Dict[str, Any]]:
    try:
        response = requests.post(f"{API_URL}/upload", files=files, timeout=30)
        if response.headers.get("content-type", "").startswith("application/json"):
            payload = response.json()
        else:
            payload = {"detail": response.text}
        response.raise_for_status()
        return payload
    except requests.RequestException as exc:
    def ask_question(question: str, top_k: int = 5) -> Dict[str, Any]:
        try:
         response = requests.post(
            f"{API_URL}/ask",
            json={"question": question, "top_k": top_k},
            timeout=30,
        )
        payload = response.json()
        response.raise_for_status()
        return payload
    except requests.RequestException as exc:
        err_detail = ""
        if getattr(exc, "response", None) is not None:
            try:
                err_detail = exc.response.json().get("detail", exc.response.text)
            except Exception:  # noqa: BLE001 - streamlit surface only
                err_detail = exc.response.text
        return {"error": str(exc), "detail": err_detail}


with col_upload:
    st.markdown("<div class='dq-card'>", unsafe_allow_html=True)
    st.subheader("Upload a PDF", anchor=False)
    st.caption("Supported format: searchable PDF up to the platform's size limit.")

    uploaded_file = st.file_uploader(
        "Choose a PDF file", type=["pdf"], label_visibility="collapsed"
    )
    upload_btn = st.button("Upload document", key="upload_btn")

    if upload_btn and uploaded_file:
        with st.spinner("Uploading and analyzing..."):
            result = upload_pdf(uploaded_file)
        if result and "error" not in result:
            upload_status.success(
                f"✅ Uploaded `{uploaded_file.name}` — indexed pages: {result.get('page_count', '?')}"
            )
        else:
            error_message = result.get("detail") if result else "Unknown error"
            upload_status.error(f"Upload failed: {error_message}")
    elif upload_btn and not uploaded_file:
        upload_status.warning("Please choose a PDF file before uploading.")

    docs = list_documents()
    if docs:
        st.markdown("#### Indexed documents")
        for doc in docs:
            st.markdown(
                f"- **{doc.get('doc_id', 'unknown')}** · {doc.get('chunks', 0)} chunks"
            )
    else:
        st.caption("No indexed documents found yet.")

    st.markdown("</div>", unsafe_allow_html=True)

with col_query:
    st.markdown("<div class='dq-card'>", unsafe_allow_html=True)
    st.subheader("Ask DocuQuery", anchor=False)
    question = st.text_area(
        "What would you like to know?",
        placeholder="e.g., Who is responsible for implementation?",
        height=120,
        label_visibility="collapsed",
    )
    top_k = st.slider("Number of passages to retrieve", min_value=3, max_value=12, value=5)
    ask_btn = st.button("Get answer", key="ask_btn")

    if ask_btn:
        if not question.strip():
            ask_status.warning("Please enter a question before submitting.")
        else:
            with st.spinner("Contacting DocuQuery API..."):
                response = ask_question(question.strip(), top_k=top_k)
            if "error" in response:
                detail = response.get("detail") or response["error"]
                ask_status.error(f"Unable to retrieve answer: {detail}")
                answer_placeholder.empty()
                hits_placeholder.empty()
            else:
                ask_status.success("Answer ready")
                answer = response.get("answer") or response.get("answer", {}).get("answer")
                if isinstance(answer, dict):
                    answer_text = answer.get("answer", "")
                else:
                    answer_text = answer or ""

                if answer_text:
                    answer_placeholder.markdown(
                        f"<div class='dq-answer'>{answer_text}</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    answer_placeholder.info("No direct answer returned. Review the passages below.")

                citations = response.get("citations") or []
                hits = response.get("results") or response.get("hits") or []
                with hits_placeholder:
                    if hits:
                        st.markdown("#### Supporting passages")
                        for idx, hit in enumerate(hits, start=1):
                            title = hit.get("title") or f"Passage {idx}"
                            page = hit.get("page")
                            snippet = hit.get("snippet") or hit.get("text") or ""
                            meta_line = f"Page {page}" if page is not None else ""
                            if citations and idx <= len(citations):
                                citation = citations[idx - 1]
                                if isinstance(citation, dict):
                                    cite_page = citation.get("page")
                                    cite_title = citation.get("title")
                                    parts = []
                                    if cite_title:
                                        parts.append(str(cite_title))
                                    if cite_page is not None:
                                        parts.append(f"p.{cite_page}")
                                    meta_line = " · ".join(parts) or meta_line
                            st.markdown(
                                f"<div class='dq-hit'>"
                                f"<h4>{title}</h4>"
                                f"<p>{meta_line}</p>"
                                f"<p>{snippet}</p>"
                                f"</div>",
                                unsafe_allow_html=True,
                            )
                    else:
                        st.caption("No passages returned.")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    """
    <div style="margin-top:2.5rem; text-align:center; color: var(--muted-text); font-size:0.85rem;">
        Crafted with Streamlit · Customize colours via environment variables or CSS overrides.
    </div>
    """,
    unsafe_allow_html=True,
)
