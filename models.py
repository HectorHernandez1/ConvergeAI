from pydantic import BaseModel, Field
from typing import Literal, Optional
from datetime import datetime

class Answer(BaseModel):
    question_number: int
    question_text: str
    answer: str
    reasoning: str
    confidence: Literal["high", "medium", "low"]
    references_cited: list[str] = Field(default_factory=list)

class SolverResponse(BaseModel):
    model_name: str
    iteration: int
    answers: list[Answer]
    tokens_used: int
    cost_usd: float

class ComparisonResult(BaseModel):
    agreement_percentage: float
    matching_questions: list[int] = Field(default_factory=list)
    differing_questions: list[dict] = Field(default_factory=list)

class FinalOutput(BaseModel):
    timestamp: datetime
    iterations_needed: int
    final_agreement: float
    consensus_answers: list[Answer]
    model_responses: dict[str, list[SolverResponse]]
    iteration_comparisons: list[ComparisonResult] = Field(default_factory=list)
    total_cost_usd: float

class IterationLog(BaseModel):
    iteration: int
    comparison: ComparisonResult
    responses: list[SolverResponse]
    timestamp: datetime
