import re
import requests
from typing import List, Dict
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.core.rag import add_documents
from app.services.summarizer import summarize
from app.core.config import settings

_SERP_API_KEY = settings.SERPAPI_KEY
_SERP_API_URL = settings.SERPAPI_URL

_CONCERT_KEYWORDS = re.compile(
    r"\b(concert|tour|gig|setlist|venue|world tour|performance)\b",
    flags=re.I,
)


def is_concert_document(text: str) -> bool:
    return bool(_CONCERT_KEYWORDS.search(text))


_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,     # ~2 paragraphs / 250‑300 tokens
    chunk_overlap=50,
    separators=["\n\n", "\n", ". "],
)

async def ingest_document(text:str) -> str:
    summary = await summarize(text)

    chunks: List[str] = _text_splitter.split_text(text)

    docs: List[Document] = []

    for chunk in chunks:
        docs.append(Document(page_content=chunk, metadata={"type": "chunk"}))

    docs.append(Document(page_content=summary, metadata={"type": "summary"}))

    add_documents(docs)
    return summary


def _scrape_serper_events(artist: str) -> List[str]:
    headers = {
        "X-API-KEY": settings.SERPAPI_KEY,
        "Content-Type": "application/json"
    }

    payload = {
        "q": f"{artist} upcoming concerts 2025 2026"
    }

    response = requests.post(settings.SERPAPI_URL, json=payload, headers=headers, timeout=15)
    response.raise_for_status()
    data: Dict = response.json()

    events: List[str] = []

    for result in data.get("organic", []):
        title = result.get("title", "")
        snippet = result.get("snippet", "")
        link = result.get("link", "")

        # Only include links/snippets that clearly mention 2025/2026 tour dates
        if any(year in snippet for year in ["2025", "2026"]):
            events.append(f"{title} — {snippet} ({link})")

    return events

async def lookup_artist_online(artist: str) -> dict:
    """Fetch upcoming concerts via SerpAPI.  Separate from RAG flow."""
    try:
        events = _scrape_serper_events(artist)
    except Exception as exc:
        return {
            "artist": artist,
            "answer": f"Error while searching online: {exc}"
        }

    if not events:
        return {"artist": artist, "answer": "I couldn't find any concerts."}

    pretty = "\n".join(events)
    return {
        "artist": artist,
        "answer": f"Upcoming concerts found online:\n{pretty}"
    }
