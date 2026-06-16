from __future__ import annotations

from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

Category = Literal[
    "Account Suspension",
    "Bug Report",
    "Data Sync Issue",
    "Feature Request",
    "Login Issue",
    "Payment Problem",
    "Performance Issue",
    "Refund Request",
    "Security Concern",
    "Subscription Cancellation",
]
Priority = Literal["Low", "Medium", "High", "Urgent"]

PRIORITY_RANK = {"Low": 0, "Medium": 1, "High": 2, "Urgent": 3}

CATEGORY_PRIORITY_FLOOR = {
    "Security Concern": "High",
    "Payment Problem": "Medium",
    "Account Suspension": "Medium",
    "Refund Request": "Medium",
}


class IncomingTicket(BaseModel):
    subject: str = Field(min_length=3, max_length=300)
    issue_description: str = Field(min_length=10)
    product: str
    channel: str
    customer_segment: str
    subscription_type: str
    ticket_id: Optional[int] = None


class TriageResult(BaseModel):
    category: Category
    priority: Priority
    sentiment: Literal["frustrated", "neutral", "urgent_tone"]
    rationale: str = Field(min_length=10, max_length=400)
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("priority")
    @classmethod
    def security_not_low(cls, value: Priority, info) -> Priority:
        category = info.data.get("category")
        floor = CATEGORY_PRIORITY_FLOOR.get(category)
        if floor and PRIORITY_RANK[value] < PRIORITY_RANK[floor]:
            raise ValueError(f"{category} cannot have priority below {floor}")
        return value


class TicketCitation(BaseModel):
    ticket_id: int
    quote: str = Field(min_length=8, max_length=300)


class TicketResponse(BaseModel):
    category: Category
    priority: Priority
    draft_reply: str = Field(min_length=40, max_length=2000)
    action_items: list[str] = Field(min_length=1, max_length=5)
    citations: list[TicketCitation] = Field(default_factory=list)
    sla_first_response_hours: Optional[float] = Field(default=None, ge=0.1, le=168)
    sla_resolution_hours: Optional[float] = Field(default=None, ge=0.1, le=720)
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def resolution_not_shorter_than_first(self) -> "TicketResponse":
        if (
            self.sla_first_response_hours is not None
            and self.sla_resolution_hours is not None
            and self.sla_resolution_hours < self.sla_first_response_hours
        ):
            raise ValueError("sla_resolution_hours must be >= sla_first_response_hours")
        return self


class JudgeVerdict(BaseModel):
    category_correct: bool
    priority_adequate: bool
    reply_grounded: bool
    action_items_relevant: bool
    overall_score: float = Field(ge=0.0, le=1.0)
    comment: str = Field(min_length=5, max_length=400)


class CriticVerdict(BaseModel):
    ok: bool
    ghost_citations: list[str] = Field(default_factory=list)
    fabricated_numbers: list[str] = Field(default_factory=list)
    issue: str = ""


class HallucinationReport(BaseModel):
    ghost_citations: list[str] = Field(default_factory=list)
    fabricated_numbers: list[str] = Field(default_factory=list)
    total_citations: int = 0
    clean: bool = True


class SyntheticPersona(BaseModel):
    persona_id: str
    name: str
    segment: str
    subscription: str
    tone: Literal["polite", "angry", "technical", "confused"]
    ticket_subject: str
    ticket_body: str
    expected_category: Category
    expected_priority_min: Priority

    @field_validator("ticket_body")
    @classmethod
    def body_not_empty(cls, value: str) -> str:
        if len(value.strip()) < 20:
            raise ValueError("ticket_body too short for persona")
        return value


class PersonaBatch(BaseModel):
    personas: list[SyntheticPersona] = Field(min_length=3, max_length=10)


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_similar_tickets",
            "description": "RAG search over resolved support tickets. Returns similar cases with resolutions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query from ticket text"},
                    "category_hint": {"type": ["string", "null"], "description": "Optional category filter"},
                    "k": {"type": "integer", "description": "Number of hits", "default": 4},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_sla_policy",
            "description": "Return SLA first response and resolution hours for a priority level.",
            "parameters": {
                "type": "object",
                "properties": {
                    "priority": {
                        "type": "string",
                        "enum": ["Low", "Medium", "High", "Urgent"],
                    },
                },
                "required": ["priority"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_category_stats",
            "description": "Return corpus statistics for a ticket category: count and typical resolution themes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {"type": "string"},
                },
                "required": ["category"],
            },
        },
    },
]

SUBMIT_SCHEMA = {
    "type": "function",
    "function": {
        "name": "submit_response",
        "description": "Submit final structured ticket response when enough evidence is gathered.",
        "parameters": {
            "type": "object",
            "properties": {
                "category": {"type": "string"},
                "priority": {"type": "string"},
                "draft_reply": {"type": "string"},
                "action_items": {"type": "array", "items": {"type": "string"}},
                "citations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "ticket_id": {"type": "integer"},
                            "quote": {"type": "string"},
                        },
                        "required": ["ticket_id", "quote"],
                    },
                },
                "sla_first_response_hours": {"type": ["number", "null"]},
                "sla_resolution_hours": {"type": ["number", "null"]},
                "confidence": {"type": "number"},
            },
            "required": ["category", "priority", "draft_reply", "action_items", "confidence"],
        },
    },
}
