from pydantic import BaseModel, Field

class QueryIn(BaseModel):
    question: str = Field(..., example="Where will Lady Gaga perform in autumn 2025?")

class AnswerOut(BaseModel):
    answer: str