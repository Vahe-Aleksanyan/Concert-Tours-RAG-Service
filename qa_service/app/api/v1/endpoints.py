from fastapi import APIRouter, HTTPException, status

from qa_service.app.core.models import QueryIn, AnswerOut
from qa_service.app.core.rag import answer_query

router = APIRouter(tags=["qa"])


@router.post("/ask", response_model=AnswerOut)
async def ask(query: QueryIn) -> AnswerOut:
    """Retrieve an answer strictly grounded in ingested concert docs."""
    if not query.question.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Question must not be empty.")
    answer = answer_query(query.question)
    return AnswerOut(answer=answer)