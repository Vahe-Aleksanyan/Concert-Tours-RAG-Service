from pydantic import BaseModel, Field

class DocIn(BaseModel):
    text: str = Field(..., description="Raw document in plain text")

class DocOut(BaseModel):
    summary: str = Field(..., description="LLM generated concise summary")

class ArtistOut(BaseModel):
    artist: str
    answer: str