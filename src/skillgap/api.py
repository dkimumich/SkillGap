"""HTTP API for SkillGap.

Run locally:  uvicorn skillgap.api:app --reload
Then open:    http://127.0.0.1:8000/docs
"""

from dataclasses import asdict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from skillgap.core import match
from fastapi.responses import RedirectResponse

MAX_CHARS = 50_000  # generous for a resume or posting; blocks huge payloads

app = FastAPI(
    title="SkillGap",
    version="0.1.0",
    description="Scores how well a resume matches a job posting and lists missing skills.",
)


class MatchRequest(BaseModel):
    resume_text: str = Field(min_length=1, max_length=MAX_CHARS)
    posting_text: str = Field(min_length=1, max_length=MAX_CHARS)


class MatchResponse(BaseModel):
    score: int
    skill_coverage: float | None
    tfidf_similarity: float
    matched_skills: list[str]
    missing_skills: list[str]


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/match", response_model=MatchResponse)
def match_endpoint(request: MatchRequest) -> MatchResponse:
    try:
        result = match(request.resume_text, request.posting_text)
    except ValueError as exc:  # e.g. text that is only whitespace
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return MatchResponse(**asdict(result))
