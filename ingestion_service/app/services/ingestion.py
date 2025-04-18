import re
from typing import List
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.core.rag import add_documents
from app.services.summarizer import summarize

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


async def lookup_artist_online(name: str) -> dict:return {
        "artist": name,
        "answer": f"Live search not yet implemented for {name}.",
    }