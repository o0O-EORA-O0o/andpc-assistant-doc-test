from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path
from typing import Any

import chromadb
import pandas as pd
import streamlit as st
from docx import Document
from pypdf import PdfReader

DOCUMENTS_DIR = Path("documents")
CHROMA_DIR = "chroma_db_offline"
COLLECTION_NAME = "andpc_docs_test_offline"
EMBEDDING_DIM = 384
TOKEN_RE = re.compile(r"[a-zA-ZÀ-ÿ0-9]+")


def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def read_pdf_file(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = []
    for index, page in enumerate(reader.pages, start=1):
        pages.append(f"[Page {index}]\n{page.extract_text() or ''}")
    return "\n\n".join(pages)


def read_docx_file(path: Path) -> str:
    doc = Document(str(path))
    return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())


def read_excel_file(path: Path) -> str:
    sheets = pd.read_excel(path, sheet_name=None)
    parts = []
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
    supported = {".md", ".txt", ".pdf", ".docx", ".xlsx", ".xls"}
    docs = []
    if not DOCUMENTS_DIR.exists():
        return docs
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
    chunks = []
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


def hashed_embedding(text: str) -> list[float]:
    """Embedding local déterministe, sans téléchargement externe.

    Ce n'est pas un vrai modèle sémantique. C'est suffisant pour tester l'architecture RAG
    dans Codespaces quand Hugging Face est inaccessible.
    """
    vector = [0.0] * EMBEDDING_DIM
    tokens = [t.lower() for t in TOKEN_RE.findall(text)]

    features = []
    features.extend(tokens)
    features.extend(f"{a}_{b}" for a, b in zip(tokens, tokens[1:]))

    for feature in features:
        digest = hashlib.md5(feature.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % EMBEDDING_DIM
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign

    norm = math.sqrt(sum(x * x for x in vector)) or 1.0
    return [x / norm for x in vector]


def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_or_create_collection(name=COLLECTION_NAME)


def reset_collection() -> None:
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass


def index_documents() -> int:
    collection = get_collection()
    ids = []
    texts = []
    metadatas = []

    for doc in load_documents():
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

    embeddings = [hashed_embedding(text) for text in texts]
    collection.upsert(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)
    return len(texts)


def search_documents(question: str, n_results: int = 5) -> list[dict[str, Any]]:
    collection = get_collection()
    results = collection.query(
        query_embeddings=[hashed_embedding(question)],
        n_results=n_results,
    )

    passages = []
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


def build_answer(question: str, passages: list[dict[str, Any]]) -> str:
    lines = [
        "## Réponse de test — version offline",
        "",
        "Cette version utilise un embedding local déterministe, sans téléchargement Hugging Face.",
        "Elle sert à valider l'architecture, pas la qualité finale du modèle de recherche.",
        "",
    ]

    if "rapport" in question.lower() or "conform" in question.lower() or "a-001" in question.lower():
        lines.extend([
            "### Trame de rapport d'aide à l'évaluation",
            "",
            "**Statut :** brouillon de test, à valider par un agent.",
            "",
            "**Points à analyser :**",
            "- cohérence entre l'OP déclarée et la fiche de cadrage ;",
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


st.set_page_config(page_title="Assistant documentaire ANDPC — offline", layout="wide")
st.title("Assistant documentaire ANDPC — Test Codespaces offline")
st.caption("Prototype avec données fictives uniquement. Aucun téléchargement de modèle externe.")

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
    st.info("Version offline : pas de Hugging Face, pas de sentence-transformers, pas de modèle externe.")

examples = [
    "Quels sont les points de vigilance pour contrôler une action rattachée à l'OP-001 ?",
    "L'action A-001 est-elle cohérente avec la durée minimale attendue ?",
    "Que faire si une action n'a pas d'objectif pédagogique explicite ?",
    "Génère un rapport d'aide à l'évaluation pour l'action A-001.",
    "Quels documents doivent être vérifiés par un agent POOL ?",
]

st.header("Question métier")
selected = st.selectbox("Exemples", examples)
question = st.text_area("Question", value=selected, height=110)
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
            st.markdown(build_answer(question, passages))
            with st.expander("Détails techniques des passages"):
                st.json(passages)
