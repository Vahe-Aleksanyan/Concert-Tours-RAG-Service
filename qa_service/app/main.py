from fastapi import FastAPI
from qa_service.app.api.v1.endpoints import router as v1_router
from qa_service.app.core.config import settings

app = FastAPI(title="Concert‑RAG Q&A API", version="1.0.0")
app.include_router(v1_router, prefix="/api/v1")
