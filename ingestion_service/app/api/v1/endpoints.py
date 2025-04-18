from fastapi import APIRouter, HTTPException, status
from app.core.models import DocIn, DocOut, ArtistOut

from app.services.ingestion import (
    ingest_document,
    is_concert_document,
    lookup_artist_online,  # BONUS – safe to stub “pass” if not needed yet
)

router = APIRouter(tags=["ingestion"])

@router.post("/ingest", response_model=DocOut, status_code=status.HTTP_201_CREATED)
async def ingest(doc: DocIn) -> DocOut:
    if not is_concert_document(doc.text):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Sorry, I can only ingest concert‑tour related documents.",
        )
    summary = await ingest_document(doc.text)
    return DocOut(summary=summary)




@router.get("/artist", response_model=ArtistOut)
async def artist(name: str) -> ArtistOut:
    data = await lookup_artist_online(name)
    return ArtistOut(**data)