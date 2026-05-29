from __future__ import annotations

from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


AspectName = Literal["performance", "design", "support", "price", "ads", "reliability"]
SentimentLabel = Literal["positive", "neutral", "negative"]
SupportLabel = Literal["supported", "weakly_supported", "not_supported"]


class Issue(BaseModel):
    category: Literal["performance", "design", "support", "price", "ads", "reliability", "other"]
    summary: str = Field(min_length=3, max_length=200)
    quote: str = Field(min_length=6)
    severity: Literal[1, 2, 3, 4, 5]


class Review(BaseModel):
    review_id: str
    platform: Literal["Google Play", "App Store", "RuStore", "Unknown"]
    review_date: str
    rating: Optional[int] = Field(default=None, ge=1, le=5)
    sentiment: SentimentLabel
    issues: list[Issue] = Field(default_factory=list)
    short_summary: str = Field(min_length=5, max_length=200)

    @field_validator("review_date")
    @classmethod
    def validate_review_date(cls, value: str) -> str:
        parsed = date.fromisoformat(value)
        if parsed > date.today():
            raise ValueError("review_date must not be in the future")
        return value


class AspectMention(BaseModel):
    aspect: AspectName
    sentiment: SentimentLabel
    score: Literal[-1, 0, 1]
    quote: str = Field(min_length=6)
    rationale: str = Field(min_length=5, max_length=180)


class ReviewAspects(BaseModel):
    review_id: str
    aspects: list[AspectMention] = Field(default_factory=list)


class ChunkSummary(BaseModel):
    review_ids: list[str] = Field(min_length=1)
    key_points: list[str] = Field(min_length=2)
    major_aspects: list[AspectName] = Field(min_length=1)
    notable_quotes: list[str] = Field(default_factory=list)


class DiscussionSummary(BaseModel):
    headline: str = Field(min_length=8, max_length=140)
    key_findings: list[str] = Field(min_length=3, max_length=8)
    action_items: list[str] = Field(min_length=3, max_length=8)
    risks: list[str] = Field(min_length=2, max_length=6)


class ActionVerdict(BaseModel):
    action_item: str
    support: SupportLabel
    evidence: list[str] = Field(default_factory=list)
    comment: str


class JudgeReport(BaseModel):
    verdicts: list[ActionVerdict] = Field(min_length=1)
    overall_score: float = Field(ge=0.0, le=1.0)
    summary: str


class SourceDocSummary(BaseModel):
    source_id: str
    review_count: int = Field(ge=1)
    dominant_aspects: list[AspectName] = Field(min_length=1)
    recurring_issues: list[str] = Field(min_length=2)
    notable_quotes: list[str] = Field(default_factory=list)


class MultiDocSummary(BaseModel):
    cross_source_patterns: list[str] = Field(min_length=3)
    source_specific_findings: list[str] = Field(min_length=3)
    consolidated_actions: list[str] = Field(min_length=3, max_length=8)
    confidence: float = Field(ge=0.0, le=1.0)
