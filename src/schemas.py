"""
Pydantic v2 Request & Response Schemas for @AmazonHelp Support API.
"""

from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field

class TriageRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000, description="Incoming customer tweet or message")
    session_id: Optional[str] = Field(None, description="Optional customer session/thread ID for multi-turn tracking")

class ConformalRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000, description="Customer text to generate conformal set for")

class ConformalPredictionResponse(BaseModel):
    prediction_set: List[str]
    set_size: int
    is_uncertain: bool
    confidence: float
    conformal_quantile: Optional[float] = None

class OODDetectionResponse(BaseModel):
    is_ood: bool
    energy_score: float
    energy_threshold: Optional[float] = None
    status: Literal["IN_DISTRIBUTION", "OUT_OF_DISTRIBUTION"]

class EscalationResponse(BaseModel):
    should_escalate: bool
    stated_reason: str
    risk_score: float
    triggers: List[str]

class TriageResponse(BaseModel):
    intent: str
    confidence: float
    conformal_prediction: ConformalPredictionResponse
    ood_detection: OODDetectionResponse
    escalation: EscalationResponse
    crag_action: str
    draft_reply: str
    guardrail_passed: bool
    character_count: int
    latency_ms: float

class HealthResponse(BaseModel):
    status: str
    brand: str
    model_version: str
    corpus_size: int
    conformal_coverage_guarantee: str
