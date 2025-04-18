from textwrap import dedent

from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# from ingestion_service.app.core.config import settings
from app.core.config import settings

_llm = ChatOpenAI(
    openai_api_key=settings.OPENAI_API_KEY,
    model_name=settings.MODEL_NAME,
    temperature=0.3,
    max_tokens=150,
)

_prompt = ChatPromptTemplate.from_template(
    dedent(
        """
        You are a helpful assistant summarising concert tour documents.
        Provide a 3–4 sentence summary covering:
          • Artist / tour name
          • Date range & regions
          • Notable venues or special guests
        Output plain text, no lists.
        Document:
        {document}
        """
    )
)

async def summarize(document: str) -> str:
    messages = _prompt.format_messages(document=document)
    response = await _llm.agenerate([messages])
    return response.generations[0][0].text.strip()
