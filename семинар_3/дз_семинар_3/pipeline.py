from __future__ import annotations

import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from llm_client import get_model, make_client
from prompts import (
    ASPECTS_SYSTEM,
    CHUNK_SYSTEM,
    IE_SYSTEM,
    JUDGE_SYSTEM,
    JUDGE_SYSTEM_STRICT,
    MULTI_DOC_SYSTEM,
    REDUCE_SYSTEM,
    REDUCE_SYSTEM_STRICT,
)
from schema import DiscussionSummary, JudgeReport, MultiDocSummary, Review, ReviewAspects, SourceDocSummary

ASPECT_ORDER = ["performance", "design", "support", "price", "ads", "reliability"]
REVIEW_SPLIT_RE = re.compile(r"(?=^ReviewID:\s)", re.MULTILINE)
MODEL = get_model()
client = make_client()


class UsageTracker:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.calls = 0

    def add(self, completion: Any) -> None:
        usage = getattr(completion, "usage", None)
        if usage is None:
            return
        with self._lock:
            self.prompt_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
            self.completion_tokens += int(getattr(usage, "completion_tokens", 0) or 0)
            self.total_tokens += int(getattr(usage, "total_tokens", 0) or 0)
            self.calls += 1

    def to_dict(self) -> dict[str, Any]:
        in_rate = float(Path(".model_input_price").read_text()) if Path(".model_input_price").exists() else 0.40
        out_rate = float(Path(".model_output_price").read_text()) if Path(".model_output_price").exists() else 1.60
        estimated_cost = (self.prompt_tokens / 1_000_000) * in_rate + (self.completion_tokens / 1_000_000) * out_rate
        return {
            "model": MODEL,
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": round(estimated_cost, 6),
        }


def _batched(items: list[Any], batch_size: int) -> list[list[Any]]:
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def _save_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_input_text(input_path: str) -> str:
    p = Path(input_path)
    if p.is_dir():
        parts: list[str] = []
        for file in sorted(p.glob("*.txt")):
            parts.append(file.read_text(encoding="utf-8").strip())
        return "\n\n".join(part for part in parts if part)
    return p.read_text(encoding="utf-8")


def _split_reviews(raw_text: str) -> list[str]:
    chunks = [chunk.strip() for chunk in REVIEW_SPLIT_RE.split(raw_text) if chunk.strip()]
    return chunks


def extract_reviews(review_blocks: list[str], usage: UsageTracker) -> tuple[list[Review], int]:
    parsed: list[Review] = []
    failed_validation = 0
    first_error: Exception | None = None
    for batch in _batched(review_blocks, 8):
        payload = "\n\n".join(batch)
        messages = [
            {"role": "system", "content": IE_SYSTEM},
            {"role": "user", "content": f"Извлеки структурированные отзывы:\n\n{payload}"},
        ]
        try:
            rows, completion = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                response_model=list[Review],
                max_retries=3,
                temperature=0.0,
                with_completion=True,
            )
            usage.add(completion)
            parsed.extend(rows)
        except Exception as exc:
            if first_error is None:
                first_error = exc
            failed_validation += len(batch)
    if not parsed and first_error is not None:
        raise RuntimeError(
            "IE этап не извлек ни одного отзыва. Проверь OPENAI_API_KEY/LLM_AUTH_TOKEN и доступ к модели."
        ) from first_error
    return parsed, failed_validation


def extract_aspects(reviews: list[Review], usage: UsageTracker) -> list[ReviewAspects]:
    results: list[ReviewAspects] = []
    for batch in _batched(reviews, 10):
        compact = [
            {
                "review_id": r.review_id,
                "rating": r.rating,
                "sentiment": r.sentiment,
                "text": r.short_summary,
                "issues": [i.model_dump() for i in r.issues],
            }
            for r in batch
        ]
        messages = [
            {"role": "system", "content": ASPECTS_SYSTEM},
            {"role": "user", "content": f"Определи аспектные оценки:\n{json.dumps(compact, ensure_ascii=False)}"},
        ]
        part, completion = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            response_model=list[ReviewAspects],
            max_retries=3,
            temperature=0.0,
            with_completion=True,
        )
        usage.add(completion)
        results.extend(part)
    return results


def check_quotes(aspects: list[ReviewAspects], corpus_text: str) -> list[tuple[str, str]]:
    corpus = corpus_text.lower()
    ghosts: list[tuple[str, str]] = []
    for row in aspects:
        for mention in row.aspects:
            probe = mention.quote[:30].strip().lower()
            if probe and probe not in corpus:
                ghosts.append((row.review_id, mention.quote))
    return ghosts


def build_heatmap(aspects: list[ReviewAspects], out_path: Path) -> None:
    review_ids = [row.review_id for row in aspects]
    if not review_ids:
        plt.figure(figsize=(9, 3))
        plt.text(0.5, 0.5, "No aspect data available", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_path, dpi=180)
        plt.close()
        return

    matrix = np.full((len(review_ids), len(ASPECT_ORDER)), np.nan, dtype=float)
    idx_map = {rid: idx for idx, rid in enumerate(review_ids)}
    aspect_idx = {a: i for i, a in enumerate(ASPECT_ORDER)}

    for row in aspects:
        ridx = idx_map[row.review_id]
        for mention in row.aspects:
            matrix[ridx, aspect_idx[mention.aspect]] = mention.score

    plt.figure(figsize=(12, max(6, int(len(review_ids) * 0.4))))
    sns.heatmap(
        matrix,
        xticklabels=ASPECT_ORDER,
        yticklabels=review_ids,
        cmap="RdYlGn",
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.3,
    )
    plt.title("Review x Aspect sentiment")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _map_chunk(chunk_payload: str, usage: UsageTracker) -> dict[str, Any]:
    from pydantic import BaseModel, Field
    from typing import Literal

    class ChunkSummary(BaseModel):
        review_ids: list[str] = Field(min_length=1)
        key_points: list[str] = Field(min_length=2)
        major_aspects: list[Literal["performance", "design", "support", "price", "ads", "reliability"]] = Field(min_length=1)
        notable_quotes: list[str] = Field(default_factory=list)

    messages = [
        {"role": "system", "content": CHUNK_SYSTEM},
        {"role": "user", "content": chunk_payload},
    ]
    parsed, completion = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        response_model=ChunkSummary,
        max_retries=3,
        temperature=0.1,
        with_completion=True,
    )
    usage.add(completion)
    return parsed.model_dump()


def summarize_discussion(
    reviews: list[Review],
    aspects: list[ReviewAspects],
    usage: UsageTracker,
    strict_reduce: bool = False,
) -> DiscussionSummary:
    if not reviews:
        raise ValueError("Нет валидных отзывов для Map-Reduce. Остановлено до этапа summary.")

    by_review = {r.review_id: r for r in reviews}
    aspect_by_review = {a.review_id: a for a in aspects}
    review_ids = list(by_review.keys())
    chunks = _batched(review_ids, 8)

    mapped: list[dict[str, Any]] = [None] * len(chunks)
    with ThreadPoolExecutor(max_workers=4) as pool:
        future_map = {}
        for idx, id_chunk in enumerate(chunks):
            payload = []
            for rid in id_chunk:
                review = by_review[rid]
                row = aspect_by_review.get(rid)
                payload.append(
                    {
                        "review_id": rid,
                        "summary": review.short_summary,
                        "issues": [issue.model_dump() for issue in review.issues],
                        "aspects": [a.model_dump() for a in row.aspects] if row else [],
                    }
                )
            future_map[pool.submit(_map_chunk, json.dumps(payload, ensure_ascii=False), usage)] = idx
        for future in as_completed(future_map):
            mapped[future_map[future]] = future.result()

    reduce_prompt = REDUCE_SYSTEM_STRICT if strict_reduce else REDUCE_SYSTEM
    messages = [
        {"role": "system", "content": reduce_prompt},
        {"role": "user", "content": f"Собери итог по mini summaries:\n{json.dumps(mapped, ensure_ascii=False)}"},
    ]
    summary, completion = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        response_model=DiscussionSummary,
        max_retries=3,
        temperature=0.2,
        with_completion=True,
    )
    usage.add(completion)
    return summary


def judge_summary(
    reviews: list[Review],
    summary: DiscussionSummary,
    usage: UsageTracker,
    strict: bool = False,
) -> JudgeReport:
    evidence = []
    for r in reviews:
        evidence.append(
            {
                "review_id": r.review_id,
                "issues": [i.model_dump() for i in r.issues],
            }
        )
    packet = {
        "action_items": summary.action_items,
        "key_findings": summary.key_findings,
        "evidence_reviews": evidence,
    }
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_STRICT if strict else JUDGE_SYSTEM},
        {"role": "user", "content": json.dumps(packet, ensure_ascii=False)},
    ]
    report, completion = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        response_model=JudgeReport,
        max_retries=3,
        temperature=0.0,
        with_completion=True,
    )
    usage.add(completion)
    return report


def _load_multi_doc_sources(source_dir: Path) -> dict[str, str]:
    files = sorted(source_dir.glob("*.txt"))
    return {f.stem: f.read_text(encoding="utf-8").strip() for f in files if f.read_text(encoding="utf-8").strip()}


def run_multi_doc(
    source_dir: Path,
    out_dir: Path,
    usage: UsageTracker,
) -> tuple[dict[str, Any], MultiDocSummary]:
    sources = _load_multi_doc_sources(source_dir)
    if len(sources) < 5:
        raise ValueError("Для Multi-doc нужно минимум 5 источников (.txt) в input/multi_doc.")

    source_summaries: list[SourceDocSummary] = []
    pivot: dict[str, dict[str, int]] = {aspect: {} for aspect in ASPECT_ORDER}

    for source_id, text in sources.items():
        messages = [
            {
                "role": "system",
                "content": (
                    "Ты аналитик одного источника отзывов. Пиши по-русски. "
                    "Выдели доминирующие аспекты, повторяющиеся проблемы и 1-3 характерные цитаты."
                ),
            },
            {"role": "user", "content": f"Источник: {source_id}\n\n{text}"},
        ]
        summary, completion = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            response_model=SourceDocSummary,
            max_retries=3,
            temperature=0.1,
            with_completion=True,
        )
        usage.add(completion)
        source_summaries.append(summary)

        for aspect in summary.dominant_aspects:
            pivot[aspect][source_id] = pivot[aspect].get(source_id, 0) + 1

    pivot_payload = {
        "sources_count": len(sources),
        "sources": sorted(sources.keys()),
        "aspect_source_matrix": pivot,
        "source_summaries": [s.model_dump() for s in source_summaries],
    }

    messages = [
        {"role": "system", "content": MULTI_DOC_SYSTEM},
        {"role": "user", "content": json.dumps(pivot_payload, ensure_ascii=False)},
    ]
    consolidation, completion = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        response_model=MultiDocSummary,
        max_retries=3,
        temperature=0.2,
        with_completion=True,
    )
    usage.add(completion)
    _save_json(out_dir / "multi_doc_pivot.json", pivot_payload)
    _save_json(out_dir / "multi_doc_summary.json", consolidation.model_dump())
    return pivot_payload, consolidation


def analyze(input_path: str) -> dict[str, Any]:
    started = time.time()
    out_dir = Path("output")
    out_dir.mkdir(parents=True, exist_ok=True)
    usage = UsageTracker()

    raw_text = _load_input_text(input_path)
    raw_reviews = _split_reviews(raw_text)

    reviews, failed_validation = extract_reviews(raw_reviews, usage)
    if not reviews:
        raise ValueError(
            "После IE не осталось валидных отзывов. Проверь ключ API и корректность входных данных."
        )
    _save_json(out_dir / "reviews.json", [r.model_dump() for r in reviews])

    aspects = extract_aspects(reviews, usage)
    _save_json(out_dir / "aspects.json", [a.model_dump() for a in aspects])
    build_heatmap(aspects, out_dir / "heatmap.png")

    ghosts = check_quotes(aspects, raw_text)
    _save_json(
        out_dir / "ghost_quotes.json",
        [{"review_id": rid, "quote": q} for rid, q in ghosts],
    )

    summary = summarize_discussion(reviews, aspects, usage, strict_reduce=False)
    _save_json(out_dir / "summary.json", summary.model_dump())

    judge_report = judge_summary(reviews, summary, usage, strict=False)
    if judge_report.overall_score < 0.7:
        summary = summarize_discussion(reviews, aspects, usage, strict_reduce=True)
        _save_json(out_dir / "summary.json", summary.model_dump())
        judge_report = judge_summary(reviews, summary, usage, strict=False)
    _save_json(out_dir / "judge_report.json", judge_report.model_dump())
    judge_report_strict = judge_summary(reviews, summary, usage, strict=True)
    _save_json(out_dir / "judge_report_strict.json", judge_report_strict.model_dump())

    multi_doc_enabled = False
    multi_doc_sources = 0
    multi_doc_dir = Path("input/multi_doc")
    if multi_doc_dir.exists():
        sources = _load_multi_doc_sources(multi_doc_dir)
        multi_doc_sources = len(sources)
        if multi_doc_sources >= 5:
            run_multi_doc(multi_doc_dir, out_dir, usage)
            multi_doc_enabled = True

    usage_stats = usage.to_dict()
    elapsed_sec = time.time() - started
    metrics = {
        "input_reviews": len(raw_reviews),
        "validated_reviews": len(reviews),
        "validation_errors": failed_validation,
        "ghost_quotes": len(ghosts),
        "ghost_quote_share": round((len(ghosts) / max(1, sum(len(a.aspects) for a in aspects))), 4),
        "overall_score": judge_report.overall_score,
        "overall_score_strict": judge_report_strict.overall_score,
        "elapsed_seconds": round(elapsed_sec, 2),
        "multi_doc_enabled": multi_doc_enabled,
        "multi_doc_sources": multi_doc_sources,
        "usage": usage_stats,
    }
    _save_json(out_dir / "metrics.json", metrics)

    print("Pipeline completed.")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return metrics


if __name__ == "__main__":
    analyze("input/reviews.txt")
