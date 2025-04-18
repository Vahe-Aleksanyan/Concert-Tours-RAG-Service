from __future__ import annotations

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate
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


RELEVANT_PROMPT = ChatPromptTemplate.from_template(
    "You are a concert‑tour assistant. You can ONLY use the context below. "
    "If the context does not help, respond with exactly 'I don't know.'\n\n"
    "Question: {question}\n"
    "Context:\n{context}\n\nAnswer:"
)

retriever = vector_store.as_retriever(search_kwargs={"k": settings.RETRIEVAL_K})


qa_chain = RetrievalQA.from_chain_type(
    llm=_llm,
    retriever=retriever,
    chain_type_kwargs={"prompt": RELEVANT_PROMPT},
    return_source_documents=False,
)

   # tweak: lower = stricter, higher = looser

# def answer_query(question: str) -> str:
#     """Answer only if we found a doc with decent similarity."""
#     docs_and_scores = vector_store.similarity_search_with_score(question, k=1)
#     if not docs_and_scores or docs_and_scores[0][1] > 2:
#         return "I don't know."
#
#     result = qa_chain({"query": question})
#     return result["result"]

def answer_query(question: str) -> str:
    return qa_chain.run(question)