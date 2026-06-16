from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from agents import process_ticket
from personas import generate_personas, save_personas
from rag import ingest
from schemas import IncomingTicket

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "input"
OUTPUT = ROOT / "output"


def run_single(case: dict) -> dict:
    ticket = IncomingTicket(
        subject=case["subject"],
        issue_description=case["issue_description"],
        product=case["product"],
        channel=case["channel"],
        customer_segment=case["customer_segment"],
        subscription_type=case["subscription_type"],
        ticket_id=case.get("ticket_id"),
    )
    return process_ticket(
        ticket,
        expected_category=case.get("expected_category"),
        expected_priority_min=case.get("expected_priority_min"),
        gold_theme=case.get("gold_resolution_theme", ""),
        run_id=f"case_{case['id']}",
    )


def run_all_eval() -> dict:
    cases = json.loads((INPUT / "eval_cases.json").read_text(encoding="utf-8"))
    results = []
    for case in cases:
        print(f"Processing case {case['id']}: {case['subject'][:50]}...")
        results.append({"case": case, "result": run_single(case)})
    summary = {
        "cases": len(results),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "pipeline_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUTPUT / "pipeline_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["prepare", "ingest", "run", "personas", "full"])
    ap.add_argument("--case-id", type=int, default=None)
    args = ap.parse_args()

    if args.command == "prepare":
        from prepare_data import main as prep_main
        prep_main([])
        return

    if args.command == "ingest":
        stats = ingest()
        print(json.dumps(stats, ensure_ascii=False, indent=2))
        return

    if args.command == "personas":
        personas = generate_personas(5)
        path = save_personas(personas)
        print(f"Saved {len(personas)} personas to {path}")
        return

    if args.command == "run":
        cases = json.loads((INPUT / "eval_cases.json").read_text(encoding="utf-8"))
        if args.case_id is not None:
            cases = [c for c in cases if c["id"] == args.case_id]
        if not cases:
            raise SystemExit("case not found")
        result = run_single(cases[0])
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.command == "full":
        from prepare_data import main as prep_main
        prep_main([])
        ingest()
        personas = generate_personas(5)
        save_personas(personas)
        summary = run_all_eval()
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return


if __name__ == "__main__":
    main()
