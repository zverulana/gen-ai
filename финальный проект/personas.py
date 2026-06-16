from __future__ import annotations

import json
from pathlib import Path

from llm_client import get_model, make_client
from schemas import PersonaBatch, SyntheticPersona

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "input"
OUTPUT = ROOT / "output"

PERSONA_SYSTEM = """Generate realistic synthetic customer support personas for testing.
Each persona has a distinct tone and a ticket that maps to one category.
Do not copy real names from training data. English tickets only.
Categories: Account Suspension, Bug Report, Data Sync Issue, Feature Request, Login Issue,
Payment Problem, Performance Issue, Refund Request, Security Concern, Subscription Cancellation."""


def generate_personas(n: int = 5) -> list[SyntheticPersona]:
    client = make_client()
    batch, _ = client.chat.completions.create(
        model=get_model(),
        messages=[
            {"role": "system", "content": PERSONA_SYSTEM},
            {
                "role": "user",
                "content": f"Generate exactly {n} diverse personas with expected_category and expected_priority_min.",
            },
        ],
        response_model=PersonaBatch,
        max_retries=3,
        temperature=0.7,
        with_completion=True,
    )
    return batch.personas


def save_personas(personas: list[SyntheticPersona], path: Path | None = None) -> Path:
    path = path or OUTPUT / "synthetic_personas.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([p.model_dump() for p in personas], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def load_personas(path: Path | None = None) -> list[SyntheticPersona]:
    path = path or OUTPUT / "synthetic_personas.json"
    if not path.exists():
        return []
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [SyntheticPersona(**r) for r in rows]


if __name__ == "__main__":
    personas = generate_personas(5)
    path = save_personas(personas)
    print(f"Saved {len(personas)} personas to {path}")
