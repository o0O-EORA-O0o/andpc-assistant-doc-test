from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import chromadb
import pandas as pd
import requests
import streamlit as st
from docx import Document
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


DOCUMENTS_DIR = Path("documents")
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "andpc_docs_test"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")


SYSTEM_INSTRUCTIONS = """
Tu es un assistant documentaire de test pour l'ANDPC.
Tu réponds uniquement à partir des extraits fournis.
Si les extraits ne suffisent pas, tu le dis clairement.
Tu cites les sources utilisées.
Tu ne prends jamais de décision métier automatique.
Tu formules les conclusions comme une aide à l'analyse, à valider par un agent.
""".strip()


@st.cache_resource
def load_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(DEFAULT_EMBEDDING_MODEL)


@st.cache_resource
def get_chroma_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_or_create_collection(name=COLLECTION_NAME)


def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def read_pdf_file(path: Path) -> str:
    reader = PdfReader(str(path))
    pages: list[str] = []
    for index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append(f"[Page {index}]\n{text}")
    return "\n\n".join(pages)


def read_docx_file(path: Path) -> str:
    doc = Document(str(path))
    return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())


def read_excel_file(path: Path) -> str:
    sheets = pd.read_excel(path, sheet_name=None)
    parts: list[str] = []
    for sheet_name, df in sheets.items():
        parts.append(f"# Feuille {sheet_name}")
        parts.append(df.fillna("").to_markdown(index=False))
    return "\n\n".join(parts)


def read_document(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".md", ".txt"}:
        return read_text_file(path)
    if suffix == ".pdf":
        return read_pdf_file(path)
    if suffix == ".docx":
        return read_docx_file(path)
    if suffix in {".xlsx", ".xls"}:
        return read_excel_file(path)
    return ""


def load_documents() -> list[dict[str, str]]:
    docs: list[dict[str, str]] = []
    if not DOCUMENTS_DIR.exists():
        return docs

    supported = {".md", ".txt", ".pdf", ".docx", ".xlsx", ".xls"}
    for path in sorted(DOCUMENTS_DIR.rglob("*")):
        if path.is_file() and path.suffix.lower() in supported:
            text = read_document(path)
            if text.strip():
                docs.append({
                    "id": path.as_posix().replace("/", "__"),
                    "title": path.name,
                    "path": path.as_posix(),
                    "text": text,
                })
    return docs


def chunk_text(text: str, max_chars: int = 1000) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        if len(current) + len(paragraph) + 2 <= max_chars:
            current = f"{current}\n\n{paragraph}".strip()
        else:
            if current:
                chunks.append(current)
            current = paragraph

    if current:
        chunks.append(current)

    return chunks


def reset_collection() -> None:
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    st.cache_resource.clear()


def index_documents() -> int:
    model = load_embedding_model()
    collection = get_chroma_collection()
    docs = load_documents()

    ids: list[str] = []
    texts: list[str] = []
    metadatas: list[dict[str, Any]] = []

    for doc in docs:
        for chunk_index, chunk in enumerate(chunk_text(doc["text"])):
            ids.append(f"{doc['id']}_{chunk_index}")
            texts.append(chunk)
            metadatas.append({
                "source": doc["title"],
                "path": doc["path"],
                "chunk": chunk_index,
            })

    if not texts:
        return 0

    embeddings = model.encode(texts, normalize_embeddings=True).tolist()
    collection.upsert(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)
    return len(texts)


def search_documents(question: str, n_results: int = 5) -> list[dict[str, Any]]:
    model = load_embedding_model()
    collection = get_chroma_collection()
    question_embedding = model.encode([question], normalize_embeddings=True).tolist()[0]
    results = collection.query(query_embeddings=[question_embedding], n_results=n_results)

    passages: list[dict[str, Any]] = []
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for doc, meta, distance in zip(documents, metadatas, distances):
        passages.append({
            "text": doc,
            "source": meta.get("source"),
            "path": meta.get("path"),
            "chunk": meta.get("chunk"),
            "distance": distance,
        })
    return passages


def build_context(passages: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for index, passage in enumerate(passages, start=1):
        parts.append(
            f"[Source {index} — {passage['source']} — chunk {passage['chunk']}]\n"
            f"{passage['text']}"
        )
    return "\n\n---\n\n".join(parts)


def ollama_is_available() -> bool:
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)
        return response.status_code == 200
    except Exception:
        return False


def call_ollama(question: str, passages: list[dict[str, Any]]) -> str:
    context = build_context(passages)
    prompt = f"""
{SYSTEM_INSTRUCTIONS}

# Question utilisateur
{question}

# Extraits documentaires disponibles
{context}

# Réponse attendue
Rédige une réponse courte, structurée et sourcée. Termine par une section "Sources utilisées".
""".strip()

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_ctx": 4096,
        },
    }
    response = requests.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload, timeout=180)
    response.raise_for_status()
    return response.json().get("response", "")


def build_retrieval_only_answer(question: str, passages: list[dict[str, Any]]) -> str:
    lines = [
        "## Réponse de test — recherche documentaire sourcée",
        "",
        "Cette version Codespaces utilise la recherche vectorielle et affiche les passages les plus pertinents.",
        "Si Ollama est installé et lancé, l'application peut aussi générer une réponse rédigée.",
        "",
    ]

    if "rapport" in question.lower() or "conform" in question.lower() or "a-001" in question.lower():
        lines.extend([
            "### Trame de rapport d'aide à l'évaluation",
            "",
            "**Statut :** brouillon de test, à valider par un agent.",
            "",
            "**Points à analyser :**",
            "- cohérence entre l'OP déclarée et les exigences de la fiche de cadrage ;",
            "- présence d'un objectif pédagogique explicite ;",
            "- conformité de la durée déclarée ;",
            "- cohérence du public cible ;",
            "- présence des pièces justificatives attendues.",
            "",
        ])

    lines.append("### Passages retrouvés")
    for index, passage in enumerate(passages, start=1):
        lines.append(f"#### Source {index} — {passage['source']}")
        lines.append(f"`{passage['path']}`")
        lines.append("")
        lines.append(passage["text"])
        lines.append("")

    return "\n".join(lines)


st.set_page_config(page_title="Assistant documentaire ANDPC", layout="wide")

st.title("Assistant documentaire ANDPC — Test Codespaces")
st.caption("Prototype avec données fictives uniquement. Ne pas utiliser de données réelles ANDPC.")

with st.sidebar:
    st.header("Administration")
    docs = load_documents()
    st.metric("Documents détectés", len(docs))

    if st.button("Réinitialiser l'index"):
        reset_collection()
        st.success("Index réinitialisé. Relance l'indexation.")

    if st.button("Indexer les documents", type="primary"):
        with st.spinner("Indexation en cours..."):
            count = index_documents()
        st.success(f"{count} passages indexés.")

    st.markdown("---")
    st.subheader("Corpus")
    for doc in docs:
        st.markdown(f"- `{doc['path']}`")

    st.markdown("---")
    st.subheader("LLM optionnel")
    st.write(f"Ollama : `{OLLAMA_BASE_URL}`")
    st.write(f"Modèle : `{OLLAMA_MODEL}`")
    st.write("Disponible : " + ("oui" if ollama_is_available() else "non"))

st.header("Question métier")

examples = [
    "Quels sont les points de vigilance pour contrôler une action rattachée à l'OP-001 ?",
    "L'action A-001 est-elle cohérente avec la durée minimale attendue ?",
    "Que faire si une action n'a pas d'objectif pédagogique explicite ?",
    "Génère un rapport d'aide à l'évaluation pour l'action A-001.",
    "Quels documents doivent être vérifiés par un agent POOL ?",
]

selected = st.selectbox("Exemples", examples)
question = st.text_area("Question", value=selected, height=110)
use_ollama = st.checkbox("Utiliser Ollama si disponible", value=False)
n_results = st.slider("Nombre de passages recherchés", min_value=2, max_value=10, value=5)

if st.button("Lancer la recherche", type="primary"):
    if not question.strip():
        st.warning("Saisis une question.")
    else:
        with st.spinner("Recherche documentaire..."):
            passages = search_documents(question, n_results=n_results)

        if not passages:
            st.error("Aucun passage trouvé. As-tu indexé les documents ?")
        else:
            if use_ollama and ollama_is_available():
                with st.spinner("Génération de la réponse via Ollama..."):
                    try:
                        answer = call_ollama(question, passages)
                    except Exception as exc:
                        st.error(f"Erreur Ollama : {exc}")
                        answer = build_retrieval_only_answer(question, passages)
            else:
                answer = build_retrieval_only_answer(question, passages)

            st.markdown(answer)

            with st.expander("Détails techniques des passages"):
                st.json(passages)
