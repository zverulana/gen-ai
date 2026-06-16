from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agents import process_ticket
from schemas import IncomingTicket, PRIORITY_RANK

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "input"
OUTPUT = ROOT / "output"

PASS_JUDGE_THRESHOLD = 0.65


def priority_ok(actual: str, minimum: str) -> bool:
    return PRIORITY_RANK.get(actual, -1) >= PRIORITY_RANK.get(minimum, 0)


def run_case(case: dict) -> dict:
    ticket = IncomingTicket(
        subject=case["subject"],
        issue_description=case["issue_description"],
        product=case["product"],
        channel=case["channel"],
        customer_segment=case["customer_segment"],
        subscription_type=case["subscription_type"],
        ticket_id=case.get("ticket_id"),
    )
    result = process_ticket(
        ticket,
        expected_category=case["expected_category"],
        expected_priority_min=case["expected_priority_min"],
        gold_theme=case.get("gold_resolution_theme", ""),
        run_id=f"eval_{case['id']}",
    )

    triage = result.get("triage") or {}
    answer = result.get("answer") or {}
    judge = result.get("judge") or {}
    hallucination = result.get("hallucination") or {}
    tools_used = result.get("tools_used") or []
    usage = result.get("usage") or {}

    category_match = triage.get("category") == case["expected_category"]
    priority_match = priority_ok(triage.get("priority", "Low"), case["expected_priority_min"])
    tool_match = all(t in tools_used for t in case.get("expected_tools", []))
    judge_ok = (judge.get("overall_score") or 0) >= PASS_JUDGE_THRESHOLD
    hallucination_clean = bool(hallucination.get("clean", False))
    has_answer = bool(answer.get("draft_reply"))
    ok = has_answer and category_match and priority_match and tool_match and judge_ok

    return {
        "id": case["id"],
        "subject": case["subject"][:80],
        "ok": ok,
        "category_match": category_match,
        "priority_match": priority_match,
        "tool_match": tool_match,
        "judge_score": judge.get("overall_score"),
        "judge_ok": judge_ok,
        "hallucination_clean": hallucination_clean,
        "ghost_citations": len(hallucination.get("ghost_citations", [])),
        "fabricated_numbers": len(hallucination.get("fabricated_numbers", [])),
        "predicted_category": triage.get("category"),
        "predicted_priority": triage.get("priority"),
        "expected_category": case["expected_category"],
        "expected_priority_min": case["expected_priority_min"],
        "tools_used": tools_used,
        "steps": result.get("steps"),
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "cost_usd": usage.get("cost_usd", 0),
        "error": result.get("error"),
    }


def main() -> None:
    cases = json.loads((INPUT / "eval_cases.json").read_text(encoding="utf-8"))
    print(f"Evaluating {len(cases)} cases...\n")

    rows = []
    total_ghost = 0
    total_fabricated = 0
    for case in cases:
        print(f"[Q{case['id']}] {case['subject'][:60]}")
        row = run_case(case)
        rows.append(row)
        total_ghost += row["ghost_citations"]
        total_fabricated += row["fabricated_numbers"]
        mark = "PASS" if row["ok"] else "FAIL"
        print(
            f"  {mark} | cat={row['category_match']} pri={row['priority_match']} "
            f"tools={row['tool_match']} judge={row['judge_score']} "
            f"ghost={row['ghost_citations']} fabricated={row['fabricated_numbers']} "
            f"steps={row['steps']} cost=${row['cost_usd']:.4f}\n"
        )

    passed = sum(1 for r in rows if r["ok"])
    summary = {
        "total": len(rows),
        "passed": passed,
        "pass_rate": round(passed / len(rows), 3),
        "total_ghost_citations": total_ghost,
        "total_fabricated_numbers": total_fabricated,
        "avg_steps": round(sum(r["steps"] or 0 for r in rows) / len(rows), 2),
        "avg_cost_usd": round(sum(r["cost_usd"] for r in rows) / len(rows), 5),
        "avg_judge_score": round(
            sum(r["judge_score"] or 0 for r in rows) / len(rows), 3
        ),
    }

    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "eval_results.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=" * 70)
    print(f"Итого: {passed}/{len(rows)} ({summary['pass_rate']*100:.1f}%)")
    print(f"Ghost citations caught: {total_ghost}")
    print(f"Fabricated numbers caught: {total_fabricated}")
    print(f"Avg steps: {summary['avg_steps']}, avg cost: ${summary['avg_cost_usd']}")
    print(f"Results: {OUTPUT / 'eval_results.json'}")


if __name__ == "__main__":
    main()
