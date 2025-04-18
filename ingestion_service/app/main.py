from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# from ingestion_service.app.api.v1.endpoints import router as v1_router
from app.api.v1.endpoints import router as v1_router

# from ingestion_service.app.core.config import settings
from app.core.config import settings


app = FastAPI(
    title="Concert-RAG Integration API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

#cors
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOW_ORIGINS,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

app.include_router(v1_router, prefix="/api/v1")