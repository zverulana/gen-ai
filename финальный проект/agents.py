from __future__ import annotations

import json
import uuid
from datetime import datetime
from json.decoder import JSONDecodeError
from pathlib import Path
from typing import Any

from llm_client import get_model, make_client, make_raw_client
from schemas import (
    CriticVerdict,
    IncomingTicket,
    JudgeVerdict,
    SUBMIT_SCHEMA,
    TOOL_SCHEMAS,
    TriageResult,
    TicketResponse,
)
from hallucination import TOOLS_IMPL, check_hallucinations
from rag import build_context_blob

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output"
TRACE_FILE = OUTPUT / "trace.jsonl"

PRICE_IN_PER_MTOK = 0.14
PRICE_OUT_PER_MTOK = 0.28

TRIAGE_SYSTEM = """You are a support triage specialist. Classify the ticket category and priority.
Security and payment issues need higher priority. Use only provided categories.
Categories: Account Suspension, Bug Report, Data Sync Issue, Feature Request, Login Issue,
Payment Problem, Performance Issue, Refund Request, Security Concern, Subscription Cancellation.
Priorities: Low, Medium, High, Urgent."""

RESPONSE_SYSTEM = """You are a support response agent with tools. Never invent SLA hours or ticket quotes.
1. Call search_similar_tickets with the issue text (use triage category as category_hint).
2. Call lookup_sla_policy for the triage priority — copy SLA hours exactly from the tool result.
3. Optionally call get_category_stats for the triage category.
4. When ready, call submit_response. draft_reply must address the customer's actual issue text.
Citations must be verbatim snippets from search_similar_tickets results."""

CRITIC_SYSTEM = """You are a hallucination critic. Check if citations appear in the tool log context
and if SLA numbers match lookup_sla_policy output. ok=false if quotes or SLA hours are fabricated."""

JUDGE_SYSTEM = """You are an LLM judge for support automation. Score if category/priority match expectations,
if the reply addresses the issue, and if action items are practical. overall_score 0..1."""


class UsageTracker:
    def __init__(self) -> None:
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.cost_usd = 0.0
        self.calls = 0

    def add(self, resp: Any) -> None:
        u = getattr(resp, "usage", None)
        if u is None:
            return
        pin = int(getattr(u, "prompt_tokens", 0) or 0)
        pout = int(getattr(u, "completion_tokens", 0) or 0)
        cost = pin / 1e6 * PRICE_IN_PER_MTOK + pout / 1e6 * PRICE_OUT_PER_MTOK
        self.prompt_tokens += pin
        self.completion_tokens += pout
        self.cost_usd += cost
        self.calls += 1

    def to_dict(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "calls": self.calls,
        }


def _append_trace(run_id: str, entry: dict) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    row = {"run_id": run_id, "ts": datetime.now().isoformat(timespec="seconds"), **entry}
    with TRACE_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _exec_tool(name: str, args: dict) -> dict:
    fn = TOOLS_IMPL.get(name)
    if fn is None:
        return {"error": f"unknown tool {name}"}
    try:
        return fn(**args)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def run_triage_agent(ticket: IncomingTicket, usage: UsageTracker) -> TriageResult:
    client = make_client()
    payload = ticket.model_dump()
    result, completion = client.chat.completions.create(
        model=get_model(),
        messages=[
            {"role": "system", "content": TRIAGE_SYSTEM},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        response_model=TriageResult,
        max_retries=3,
        temperature=0.0,
        with_completion=True,
    )
    usage.add(completion)
    return result


def run_response_agent(
    ticket: IncomingTicket,
    triage: TriageResult,
    usage: UsageTracker,
    run_id: str,
    max_iter: int = 6,
) -> tuple[TicketResponse | None, list[dict], str]:
    client = make_raw_client()
    model = get_model()
    tools = TOOL_SCHEMAS + [SUBMIT_SCHEMA]
    messages: list[dict] = [
        {"role": "system", "content": RESPONSE_SYSTEM},
        {
            "role": "user",
            "content": json.dumps(
                {"ticket": ticket.model_dump(), "triage": triage.model_dump()},
                ensure_ascii=False,
            ),
        },
    ]
    trace: list[dict] = []
    context_blob = ""

    for step in range(1, max_iter + 1):
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0.0,
        )
        usage.add(resp)
        msg = resp.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))

        if not msg.tool_calls:
            trace.append({"agent": "response", "step": step, "final_text": msg.content})
            _append_trace(run_id, trace[-1])
            return None, trace, context_blob

        submit = next((tc for tc in msg.tool_calls if tc.function.name == "submit_response"), None)
        others = [tc for tc in msg.tool_calls if tc is not submit]

        for tc in others:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except JSONDecodeError as e:
                args = {}
                obs = {"error": str(e)}
            else:
                obs = _exec_tool(tc.function.name, args)
            if tc.function.name == "search_similar_tickets" and "hits" in obs:
                context_blob = build_context_blob(obs["hits"])
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(obs, ensure_ascii=False),
                }
            )
            entry = {"agent": "response", "step": step, "call": tc.function.name, "args": args, "obs": obs}
            trace.append(entry)
            _append_trace(run_id, entry)

        if submit is not None:
            try:
                raw = json.loads(submit.function.arguments or "{}")
                answer = TicketResponse(**raw)
            except Exception as e:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": submit.id,
                        "content": f"submit_response invalid: {e}",
                    }
                )
                continue
            messages.append({"role": "tool", "tool_call_id": submit.id, "content": "accepted"})
            trace.append({"agent": "response", "step": step, "submit": answer.model_dump()})
            _append_trace(run_id, trace[-1])
            return answer, trace, context_blob

    return None, trace, context_blob


def run_critic_agent(
    answer: TicketResponse,
    trace: list[dict],
    context_blob: str,
    usage: UsageTracker,
) -> CriticVerdict:
    deterministic = check_hallucinations(
        answer.draft_reply,
        [c.model_dump() for c in answer.citations],
        context_blob,
        answer.sla_first_response_hours,
        answer.sla_resolution_hours,
        answer.priority,
        trace,
    )
    if deterministic["ghost_citations"] or deterministic["fabricated_numbers"]:
        return CriticVerdict(
            ok=False,
            ghost_citations=deterministic["ghost_citations"],
            fabricated_numbers=deterministic["fabricated_numbers"],
            issue="deterministic hallucination check failed",
        )

    client = make_client()
    facts = json.dumps(trace, ensure_ascii=False)[:4000]
    verdict, completion = client.chat.completions.create(
        model=get_model(),
        messages=[
            {"role": "system", "content": CRITIC_SYSTEM},
            {
                "role": "user",
                "content": f"Answer: {answer.model_dump()}\nContext:\n{context_blob[:3000]}\nTrace:\n{facts}",
            },
        ],
        response_model=CriticVerdict,
        max_retries=2,
        temperature=0.0,
        with_completion=True,
    )
    usage.add(completion)
    return verdict


def run_judge(
    ticket: IncomingTicket,
    triage: TriageResult,
    answer: TicketResponse,
    expected_category: str,
    expected_priority_min: str,
    gold_theme: str,
    usage: UsageTracker,
) -> JudgeVerdict:
    client = make_client()
    packet = {
        "ticket": ticket.model_dump(),
        "triage": triage.model_dump(),
        "answer": answer.model_dump(),
        "expected_category": expected_category,
        "expected_priority_min": expected_priority_min,
        "gold_resolution_theme": gold_theme,
    }
    verdict, completion = client.chat.completions.create(
        model=get_model(),
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": json.dumps(packet, ensure_ascii=False)},
        ],
        response_model=JudgeVerdict,
        max_retries=2,
        temperature=0.0,
        with_completion=True,
    )
    usage.add(completion)
    return verdict


def process_ticket(
    ticket: IncomingTicket,
    *,
    expected_category: str | None = None,
    expected_priority_min: str | None = None,
    gold_theme: str = "",
    run_id: str | None = None,
) -> dict[str, Any]:
    run_id = run_id or str(uuid.uuid4())
    usage = UsageTracker()

    triage = run_triage_agent(ticket, usage)
    _append_trace(run_id, {"agent": "triage", "result": triage.model_dump()})

    answer, trace, context_blob = run_response_agent(ticket, triage, usage, run_id)
    if answer is None:
        return {
            "run_id": run_id,
            "error": "response agent did not submit structured answer",
            "triage": triage.model_dump(),
            "trace": trace,
            "usage": usage.to_dict(),
        }

    critic = run_critic_agent(answer, trace, context_blob, usage)
    _append_trace(run_id, {"agent": "critic", "verdict": critic.model_dump()})

    hallucination = check_hallucinations(
        answer.draft_reply,
        [c.model_dump() for c in answer.citations],
        context_blob,
        answer.sla_first_response_hours,
        answer.sla_resolution_hours,
        answer.priority,
        trace,
    )

    judge = None
    if expected_category and expected_priority_min:
        judge = run_judge(
            ticket,
            triage,
            answer,
            expected_category,
            expected_priority_min,
            gold_theme,
            usage,
        )
        _append_trace(run_id, {"agent": "judge", "verdict": judge.model_dump()})

    tools_used = list({e["call"] for e in trace if "call" in e})
    steps = len([e for e in trace if e.get("agent") == "response"])

    return {
        "run_id": run_id,
        "triage": triage.model_dump(),
        "answer": answer.model_dump(),
        "critic": critic.model_dump(),
        "hallucination": hallucination,
        "judge": judge.model_dump() if judge else None,
        "trace": trace,
        "tools_used": tools_used,
        "steps": steps,
        "usage": usage.to_dict(),
    }
