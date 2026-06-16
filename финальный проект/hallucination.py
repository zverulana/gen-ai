from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from rag import build_context_blob, search_similar_tickets

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "input"

_SLA = json.loads((INPUT / "sla_policies.json").read_text(encoding="utf-8")) if (INPUT / "sla_policies.json").exists() else {}
_corpus_stats: dict | None = None


def _load_corpus_stats() -> dict:
    global _corpus_stats
    if _corpus_stats is None:
        path = INPUT / "tickets_corpus.csv"
        df = pd.read_csv(path)
        by_cat = {}
        for cat, grp in df.groupby("category"):
            themes = grp["resolution_notes"].astype(str).str[:120].head(3).tolist()
            by_cat[cat] = {"count": int(len(grp)), "sample_themes": themes}
        _corpus_stats = by_cat
    return _corpus_stats


def lookup_sla_policy(priority: str) -> dict:
    policy = _SLA.get(priority)
    if not policy:
        return {"error": f"unknown priority {priority}"}
    return {"priority": priority, **policy}


def get_category_stats(category: str) -> dict:
    stats = _load_corpus_stats()
    row = stats.get(category)
    if not row:
        return {"error": f"no stats for {category}"}
    return {"category": category, **row}


TOOLS_IMPL = {
    "search_similar_tickets": search_similar_tickets,
    "lookup_sla_policy": lookup_sla_policy,
    "get_category_stats": get_category_stats,
}


def _norm_num(value: float | int | str) -> str:
    try:
        f = float(value)
        if f.is_integer():
            return str(int(f))
        return str(round(f, 2))
    except (TypeError, ValueError):
        return str(value)


def _quote_in_context(quote: str, corpus_lower: str) -> bool:
    probe = quote[:50].strip().lower()
    if not probe:
        return True
    if probe in corpus_lower:
        return True
    words = [w for w in re.findall(r"[a-z0-9]{4,}", probe) if len(w) >= 4]
    if len(words) >= 3:
        hits = sum(1 for w in words if w in corpus_lower)
        return hits >= max(2, len(words) // 2)
    return False


def _collect_allowed_numbers(context_blob: str, trace: list[dict], priority: str) -> set[str]:
    allowed: set[str] = set()
    policy = _SLA.get(priority, {})
    for key in ("first_response_hours", "resolution_hours", "escalation_threshold"):
        val = policy.get(key)
        if val is not None:
            allowed.add(_norm_num(val))

    blob = context_blob + " " + json.dumps(trace, ensure_ascii=False)
    for m in re.findall(r"\b\d+(?:\.\d+)?\b", blob):
        allowed.add(_norm_num(m))
    return allowed


def check_hallucinations(
    response_text: str,
    citations: list[dict],
    context_blob: str,
    sla_first: float | None,
    sla_resolution: float | None,
    priority: str,
    trace: list[dict] | None = None,
) -> dict:
    trace = trace or []
    corpus_lower = context_blob.lower()
    ghost_citations: list[str] = []
    for c in citations:
        quote = c.get("quote", "")
        if not _quote_in_context(quote, corpus_lower):
            ghost_citations.append(quote)

    allowed = _collect_allowed_numbers(context_blob, trace, priority)
    fabricated_numbers: list[str] = []
    for val, label in ((sla_first, "sla_first_response_hours"), (sla_resolution, "sla_resolution_hours")):
        if val is None:
            continue
        if _norm_num(val) not in allowed:
            fabricated_numbers.append(f"{label}={val}")

    for num in re.findall(r"\$\d+(?:\.\d+)?|\b\d{1,4}(?:\.\d+)?\b", response_text):
        clean = num.lstrip("$")
        if _norm_num(clean) in allowed:
            continue
        if clean.isdigit() and int(clean) <= 10:
            continue
        if "$" in num or (clean.replace(".", "").isdigit() and float(clean) >= 20):
            fabricated_numbers.append(f"reply_number={num}")

    clean = not ghost_citations and not fabricated_numbers
    return {
        "ghost_citations": ghost_citations,
        "fabricated_numbers": fabricated_numbers,
        "total_citations": len(citations),
        "clean": clean,
    }
