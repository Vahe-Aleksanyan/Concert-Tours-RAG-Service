from __future__ import annotations

from pathlib import Path
from typing import List

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from chromadb.config import Settings as ChromaClientSettings

from app.core.config import settings

import chromadb
from chromadb import HttpClient

client = HttpClient(host=settings.CHROMA_HOST, port=settings.CHROMA_PORT)

# Recreate collection if deleted
try:
    client.get_collection(settings.CHROMA_COLLECTION)
except:
    client.create_collection(settings.CHROMA_COLLECTION)


# Embedding function
_embedder = OpenAIEmbeddings(
    openai_api_key=settings.OPENAI_API_KEY,
    model=settings.EMBEDDING_MODEL_NAME,
)


chroma_client = HttpClient(
    host=settings.CHROMA_HOST,
    port=settings.CHROMA_PORT
)

vector_store = Chroma(
    client=chroma_client,
    persist_directory=str(settings.CHROMA_PERSIST_DIR),
    collection_name=settings.CHROMA_COLLECTION,
    embedding_function=_embedder,
)

# LLM for QA
_llm = ChatOpenAI(
    openai_api_key=settings.OPENAI_API_KEY,
    model_name=settings.MODEL_NAME,
    temperature=settings.TEMPERATURE,
    max_tokens=settings.MAX_TOKENS,
)

# Retrieval‑augmented QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=_llm,
    retriever=vector_store.as_retriever(search_kwargs={"k": settings.RETRIEVAL_K}),
    return_source_documents=False,
)


def add_documents(docs: List[Document]) -> None:
    """Add a batch of Document objects into ChromaDB."""
    vector_store.add_documents(docs)


def answer_query(question: str) -> str:
    """Run a retrieval‑augmented query against your ingested documents."""
    return qa_chain.run(question)
