"""Local-only FastAPI application for independent support-policy review."""

from __future__ import annotations

from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from src.review_dashboard import REVIEW_DASHBOARD_HTML
from src.rating_dashboard import RATING_DASHBOARD_HTML
from src.review_store import ReplyRatingStore, TrainingReviewStore


app = FastAPI(title="Support policy review desk", docs_url=None, redoc_url=None)
store = TrainingReviewStore()
rating_store = ReplyRatingStore()


class ReviewSubmission(BaseModel):
    reviewer_id: str = Field(..., min_length=2, max_length=32)
    inbound_id: str
    source_sha256: str = Field(..., min_length=64, max_length=64)
    reviewed_intent: str
    reviewed_escalate: bool
    reviewed_reason_code: str
    reviewed_resolution_act: str
    ambiguous: bool = False
    review_notes: Optional[str] = Field(default="", max_length=1000)


class RatingSubmission(BaseModel):
    reviewer_id: str = Field(..., min_length=2, max_length=32)
    reply_id: str
    source_sha256: str = Field(..., min_length=64, max_length=64)
    human_groundedness: int = Field(..., ge=1, le=5)
    human_empathy: int = Field(..., ge=1, le=5)
    human_actionability: int = Field(..., ge=1, le=5)
    human_safety: int = Field(..., ge=1, le=5)
    human_notes: Optional[str] = Field(default="", max_length=1000)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return REVIEW_DASHBOARD_HTML


@app.get("/ratings", response_class=HTMLResponse)
def ratings_index() -> str:
    return RATING_DASHBOARD_HTML


@app.get("/api/review/next")
def next_review(reviewer_id: str):
    try:
        item = store.next_item(reviewer_id)
        if item is None:
            return {"complete": True, **store.progress(reviewer_id)}
        return {"complete": False, **item}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/review")
def save_review(submission: ReviewSubmission):
    try:
        return store.save(submission.reviewer_id, submission.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/ratings/next")
def next_rating(reviewer_id: str):
    try:
        item = rating_store.next_item(reviewer_id)
        if item is None:
            return {"complete": True, **rating_store.progress(reviewer_id)}
        return {"complete": False, **item}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/ratings")
def save_rating(submission: RatingSubmission):
    try:
        return rating_store.save(submission.reviewer_id, submission.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
