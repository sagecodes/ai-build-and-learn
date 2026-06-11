"""Pydantic models for the research pipeline."""

from pydantic import BaseModel


class TopicReport(BaseModel):
    topic: str
    report: str


class QualityResult(BaseModel):
    score: int
    gaps: list[str]


class PipelineResult(BaseModel):
    query: str
    report: str
    sub_reports: list[TopicReport]
    score: int
    iterations: int
