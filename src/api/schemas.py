from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Question for the financial document agent.",
    )


class Source(BaseModel):
    citation_id: int
    source: str
    page: int
    document_type: str
    score: float | None = None
    chunk_id: int | None = None


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[Source] = Field(default_factory=list)
    tools_used: list[str] = Field(default_factory=list)
