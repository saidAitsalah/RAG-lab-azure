"""Shared helpers for the Day 3 RAG lab."""
import os

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizableTextQuery
from dotenv import load_dotenv

load_dotenv()

SEARCH_ENDPOINT = os.environ["SEARCH_ENDPOINT"]
SEARCH_KEY = os.environ["SEARCH_KEY"]
SEARCH_INDEX = os.environ.get("SEARCH_INDEX", "rag-day3")
CONTENT_FIELD = os.environ.get("CONTENT_FIELD", "chunk")
TITLE_FIELD = os.environ.get("TITLE_FIELD", "title")
VECTOR_FIELD = os.environ.get("VECTOR_FIELD", "text_vector")

PROJECT_ENDPOINT = os.environ.get("PROJECT_ENDPOINT", "")
OPENAI_ENDPOINT = os.environ.get("OPENAI_ENDPOINT", "").rstrip("/")
CHAT_DEPLOYMENT = os.environ.get("CHAT_DEPLOYMENT", "gpt-4.1-mini")
FOUNDRY_API_KEY = os.environ.get("FOUNDRY_API_KEY", "")


def search_client() -> SearchClient:
    return SearchClient(SEARCH_ENDPOINT, SEARCH_INDEX, AzureKeyCredential(SEARCH_KEY))


def retrieve(question: str, mode: str = "hybrid", k: int = 3) -> list[dict]:
    """Return the top-k chunks. mode = keyword | vector | hybrid."""
    vector_queries = None
    if mode in ("vector", "hybrid"):
        # The index vectorizer (created by the wizard) embeds the question.
        vector_queries = [
            VectorizableTextQuery(text=question, k_nearest_neighbors=k, fields=VECTOR_FIELD)
        ]
    search_text = question if mode in ("keyword", "hybrid") else None
    results = search_client().search(
        search_text=search_text,
        vector_queries=vector_queries,
        select=[TITLE_FIELD, CONTENT_FIELD],
        top=k,
    )
    return [
        {
            "title": r.get(TITLE_FIELD, ""),
            "content": r.get(CONTENT_FIELD, ""),
            "score": r["@search.score"],
        }
        for r in results
    ]


def chat_client():
    """OpenAI client for the Responses API.

    """
    if FOUNDRY_API_KEY:
        from openai import OpenAI

        return OpenAI(base_url=f"{OPENAI_ENDPOINT}/openai/v1/", api_key=FOUNDRY_API_KEY)
    from azure.ai.projects import AIProjectClient
    from azure.identity import DefaultAzureCredential

    project = AIProjectClient(endpoint=PROJECT_ENDPOINT, credential=DefaultAzureCredential())
    return project.get_openai_client()


def format_sources(chunks: list[dict]) -> str:
    return "\n\n".join(
        f"[{i}] (source: {c['title']})\n{c['content']}" for i, c in enumerate(chunks, start=1)
    )
