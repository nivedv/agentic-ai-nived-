"""aria_rag.py — PARTICIPANT FILE. Query-time pipeline against the portal-built index.
The index was built by the Import data (RAG) wizard; this file cannot modify it
(query key = read-only). Your work is the exercises: run, observe, modify."""
import os
from dotenv import load_dotenv

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.agents import create_agent
from langchain_core.tools import tool

load_dotenv()

SEARCH_ENDPOINT = os.environ["AZURE_SEARCH_ENDPOINT"]
SEARCH_KEY      = os.environ["AZURE_SEARCH_KEY"]
INDEX_NAME      = os.environ.get("AZURE_SEARCH_INDEX", "meridian-rag")
SEMANTIC_CONFIG = os.environ.get("AZURE_SEARCH_SEMANTIC_CONFIG",
                                 "meridian-rag-semantic-configuration")  # VERIFY exact name in portal

_credential = AzureKeyCredential(SEARCH_KEY)

def get_search_client(index_name: str | None = None) -> SearchClient:
    """Client per call, parameterised by index — Lab 4.2 flips between
    meridian-rag and meridian-rag-b through this one function."""
    return SearchClient(endpoint=SEARCH_ENDPOINT,
                        index_name=index_name or INDEX_NAME,
                        credential=_credential)

llm = AzureChatOpenAI(azure_deployment="gpt-5.4", temperature=0)
embeddings = AzureOpenAIEmbeddings(
    azure_deployment="text-embedding-3-small",   # name live-verified during wizard setup
    model="text-embedding-3-small",
)

SELECT_FIELDS = ["document_id", "title", "chunk"]


# ── RETRIEVAL (participant default: hybrid, no semantic quota drawn) ─────────
def retrieve(query: str, k: int = 5, mode: str = "hybrid",
             index_name: str | None = None) -> list[dict]:
    """mode: 'hybrid' (BM25 + vector, RRF), 'vector', 'keyword'."""
    vector_q = None
    if mode in ("hybrid", "vector"):
        vector_q = [VectorizedQuery(vector=embeddings.embed_query(query),
                                    k_nearest_neighbors=20, fields="text_vector")]
    results = get_search_client(index_name).search(
        search_text=query if mode in ("hybrid", "keyword") else None,
        vector_queries=vector_q,
        select=SELECT_FIELDS,
        top=k,
    )
    return [{"document_id": r["document_id"], "title": r["title"],
             "chunk": r["chunk"], "score": r["@search.score"]} for r in results]


def retrieve_semantic(query: str, k: int = 5,
                      index_name: str | None = None) -> list[dict]:
    """Hybrid + L2 semantic re-rank. Draws from the LIMITED monthly semantic
    allowance — used ONLY in Exercise 4 and by the trainer. Not the default."""
    vector_q = [VectorizedQuery(vector=embeddings.embed_query(query),
                                k_nearest_neighbors=20, fields="text_vector")]
    results = get_search_client(index_name).search(
        search_text=query,
        vector_queries=vector_q,
        query_type="semantic",
        semantic_configuration_name=SEMANTIC_CONFIG,
        select=SELECT_FIELDS,
        top=k,
    )
    return [{"document_id": r["document_id"], "title": r["title"],
             "chunk": r["chunk"], "score": r["@search.score"],
             "reranker_score": r["@search.reranker_score"]} for r in results]


# ── GROUNDED GENERATION ──────────────────────────────────────────────────────
GROUNDING_PROMPT = """You are ARIA, the Audit Research Intelligence Assistant for the \
Meridian Software Ltd FY2024 engagement.

Answer the question using ONLY the evidence provided below. Rules:
1. Every factual claim must come from the evidence. Cite the source as [DOC-ID] inline.
2. If the evidence does not contain the answer, reply exactly: \
"I could not find supporting documentation for this in the engagement file." Do not guess.
3. Do not use knowledge from outside the evidence, even if you believe you know the answer.

EVIDENCE:
{context}

QUESTION: {question}"""


def build_context(chunks: list[dict]) -> str:
    return "\n\n".join(f"[{c['document_id']} | {c['title']}]\n{c['chunk']}" for c in chunks)


def answer(question: str, k: int = 5, mode: str = "hybrid",
           index_name: str | None = None) -> tuple[str, list[dict]]:
    """Classic RAG. Returns (answer_text, chunks) — Lab 4.2 consumes both."""
    chunks = retrieve(question, k=k, mode=mode, index_name=index_name)
    prompt = GROUNDING_PROMPT.format(context=build_context(chunks), question=question)
    return llm.invoke(prompt).content, chunks


def answer_guarded(question: str, k: int = 5,
                   min_reranker: float = 1.8) -> tuple[str, list[dict]]:
    """Deterministic refusal BEFORE generation, thresholded on the semantic
    rerankerScore (0-4, calibrated) — the production-grade signal, live-verified
    on this service. Contrast: hybrid @search.score is an RRF rank aggregate
    (ceiling ~0.032) and is NOT calibrated. Starting band: 1.5-2.0."""
    chunks = retrieve_semantic(question, k=k)
    if not chunks or chunks[0]["reranker_score"] < min_reranker:
        return ("I could not find supporting documentation for this in the "
                "engagement file.", chunks)
    prompt = GROUNDING_PROMPT.format(context=build_context(chunks), question=question)
    return llm.invoke(prompt).content, chunks


# ── AGENTIC RAG ──────────────────────────────────────────────────────────────
@tool
def retrieve_evidence(query: str) -> str:
    """Search the Meridian FY2024 engagement file for evidence.
    Input: a focused plain-English query about contracts, revenue schedules,
    approvals, board minutes, auditor correspondence, or accounting policy.
    Returns: the top matching evidence chunks with DOC-IDs. Call again with a
    reformulated query if the results do not answer the question."""
    chunks = retrieve(query, k=4, mode="hybrid")
    return build_context(chunks) if chunks else "No evidence found for that query."


ARIA_RAG_PROMPT = """You are ARIA, the audit research assistant for the Meridian \
Software Ltd FY2024 engagement. Answer questions using the retrieve_evidence tool.

Rules:
- Retrieve BEFORE answering. If the first retrieval does not contain the answer, \
reformulate the query and retrieve again — maximum 3 retrievals per question.
- Cite every factual claim with its [DOC-ID].
- If no retrieval yields the answer, say you could not find supporting documentation. \
Never answer from memory.
- Multi-part questions may need one retrieval per part."""


def build_rag_agent():
    return create_agent(model=llm, tools=[retrieve_evidence],
                        system_prompt=ARIA_RAG_PROMPT)


if __name__ == "__main__":
    hits = retrieve("Amendment No. 3 pricing", k=3)
    print("Connected. Top hits:", [c["document_id"] for c in hits])
