import re
from typing import List

from langchain.docstore.document import Document

# from ingestion_service.app.core.rag import add_documents
# from ingestion_service.app.services.summarizer import summarize
from app.core.rag import add_documents
from app.services.summarizer import summarize

_CONCERT_KEYWORDS = re.compile(
    r"\b(concert|tour|gig|setlist|venue|world tour|performance)\b",
    flags=re.I,
)


def is_concert_document(text: str) -> bool:
    return bool(_CONCERT_KEYWORDS.search(text))


async def ingest_document(text:str) -> str:
    summary = await summarize(text)

    doc = Document(
        page_content=summary,
        metadata={"type": "summary"},
    )
    add_documents([doc])
    return summary


async def lookup_artist_online(name: str) -> dict:return {
        "artist": name,
        "answer": f"Live search not yet implemented for {name}.",
    }