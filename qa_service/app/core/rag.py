from __future__ import annotations

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from chromadb.config import Settings as ChromaClientSettings
from qa_service.app.core.config import settings


import chromadb
from chromadb import HttpClient

# Embeddings
_embedder = OpenAIEmbeddings(openai_api_key=settings.OPENAI_API_KEY, model=settings.EMBEDDING_MODEL_NAME)

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


# LLM
_llm = ChatOpenAI(
    openai_api_key=settings.OPENAI_API_KEY,
    model_name=settings.MODEL_NAME,
    temperature=settings.TEMPERATURE,
    max_tokens=settings.MAX_TOKENS,
)

qa_chain = RetrievalQA.from_chain_type(
    llm=_llm,
    retriever=vector_store.as_retriever(
        search_kwargs={"k": 20},  # <-- only k stays inside
    ),
    return_source_documents=False,
)

def answer_query(question: str) -> str:
    return qa_chain.run(question)