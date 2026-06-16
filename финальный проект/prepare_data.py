from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "input"
KAGGLE_DEFAULT = Path.home() / ".cache/kagglehub/datasets/mirzayasirabdullah07/customer-support-tickets-dataset-200k-records/versions/1/customer_support_tickets_200k.csv"

CATEGORIES = [
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
PRIORITIES = ["Low", "Medium", "High", "Urgent"]

SLA_POLICIES = {
    "Low": {"first_response_hours": 24, "resolution_hours": 72, "escalation_threshold": 96},
    "Medium": {"first_response_hours": 12, "resolution_hours": 48, "escalation_threshold": 60},
    "High": {"first_response_hours": 4, "resolution_hours": 24, "escalation_threshold": 36},
    "Urgent": {"first_response_hours": 1, "resolution_hours": 8, "escalation_threshold": 12},
}

CATEGORY_KEYWORDS = {
    "Account Suspension": ["suspend", "suspended", "locked", "blocked", "ban", "access denied"],
    "Bug Report": ["bug", "crash", "glitch", "error", "broken"],
    "Data Sync Issue": ["sync", "syncing", "synchron"],
    "Feature Request": ["feature", "request", "add", "would like", "suggest"],
    "Login Issue": ["login", "log in", "password", "credentials", "sign in"],
    "Payment Problem": ["payment", "charge", "charged", "billing", "transaction failed"],
    "Performance Issue": ["slow", "performance", "timeout", "lag", "loading"],
    "Refund Request": ["refund", "money back", "reimburse"],
    "Security Concern": ["security", "unauthorized", "hack", "breach", "suspicious"],
    "Subscription Cancellation": ["cancel", "cancellation", "unsubscribe", "terminate"],
}

CATEGORY_PRIORITY_FLOOR = {
    "Security Concern": "High",
    "Payment Problem": "Medium",
    "Account Suspension": "Medium",
    "Refund Request": "Medium",
}


def _matches_category(text: str, category: str) -> bool:
    low = text.lower()
    keys = CATEGORY_KEYWORDS.get(category, [])
    return any(k in low for k in keys)


SYNTHETIC_CASES = [
    {
        "subject": "Unauthorized charge on premium subscription",
        "issue_description": "I was charged $49.99 twice this month for Premium. I only have one account and this looks fraudulent. Please reverse immediately.",
        "product": "Web Portal",
        "channel": "Email",
        "customer_segment": "Enterprise",
        "subscription_type": "Premium",
        "expected_category": "Payment Problem",
        "expected_priority_min": "High",
        "gold_resolution_theme": "duplicate charge refund verification",
        "comment": "Синтетический кейс: двойное списание",
    },
    {
        "subject": "App crashes after login on iOS 18",
        "issue_description": "After the latest update the mobile app closes immediately after I enter OTP. Reinstall did not help. Device: iPhone 15, iOS 18.1.",
        "product": "Mobile App",
        "channel": "Chat",
        "customer_segment": "Consumer",
        "subscription_type": "Free",
        "expected_category": "Bug Report",
        "expected_priority_min": "Medium",
        "gold_resolution_theme": "crash troubleshooting steps",
        "comment": "Синтетический кейс: краш после логина",
    },
    {
        "subject": "Request: export all customer data to CSV",
        "issue_description": "We need a bulk export of usage metrics for compliance audit. Is there an API endpoint or admin panel feature for this?",
        "product": "API Gateway",
        "channel": "Email",
        "customer_segment": "Enterprise",
        "subscription_type": "Enterprise",
        "expected_category": "Feature Request",
        "expected_priority_min": "Low",
        "gold_resolution_theme": "feature request logged roadmap",
        "comment": "Синтетический кейс: feature request",
    },
    {
        "subject": "Suspicious login from another country",
        "issue_description": "I received alerts about logins from Brazil while I am in Germany. Please lock the account and review access logs.",
        "product": "Web Portal",
        "channel": "Phone",
        "customer_segment": "Small Business",
        "subscription_type": "Premium",
        "expected_category": "Security Concern",
        "expected_priority_min": "Urgent",
        "gold_resolution_theme": "account lock security review",
        "comment": "Синтетический кейс: безопасность",
    },
    {
        "subject": "Cancel annual plan and prorated refund",
        "issue_description": "I want to cancel my annual Enterprise plan effective today. What is the prorated refund amount for unused months?",
        "product": "Web Portal",
        "channel": "Email",
        "customer_segment": "Enterprise",
        "subscription_type": "Enterprise",
        "expected_category": "Subscription Cancellation",
        "expected_priority_min": "Medium",
        "gold_resolution_theme": "cancellation prorated refund policy",
        "comment": "Синтетический кейс: отмена подписки",
    },
    {
        "subject": "Dashboard loads 40 seconds, timeouts",
        "issue_description": "Since yesterday the analytics dashboard takes 40+ seconds and often times out. Other pages work fine. Region: EU.",
        "product": "Analytics Dashboard",
        "channel": "Chat",
        "customer_segment": "Small Business",
        "subscription_type": "Premium",
        "expected_category": "Performance Issue",
        "expected_priority_min": "High",
        "gold_resolution_theme": "performance degradation investigation",
        "comment": "Синтетический кейс: деградация производительности",
    },
    {
        "subject": "Cannot reset password, email not arriving",
        "issue_description": "Password reset emails never arrive. Checked spam. Tried three times over 2 days. Need access urgently for billing.",
        "product": "Web Portal",
        "channel": "Email",
        "customer_segment": "Consumer",
        "subscription_type": "Premium",
        "expected_category": "Login Issue",
        "expected_priority_min": "High",
        "gold_resolution_theme": "password reset email delivery",
        "comment": "Синтетический кейс: сброс пароля",
    },
    {
        "subject": "Mobile and web data out of sync",
        "issue_description": "Changes made on mobile do not appear on web for 6+ hours. Last successful sync was 2024-11-02. Premium user.",
        "product": "Mobile App",
        "channel": "Email",
        "customer_segment": "Consumer",
        "subscription_type": "Premium",
        "expected_category": "Data Sync Issue",
        "expected_priority_min": "Medium",
        "gold_resolution_theme": "sync queue backlog fix",
        "comment": "Синтетический кейс: рассинхрон",
    },
]


def resolve_source(path: str | None) -> Path:
    if path:
        return Path(path)
    if KAGGLE_DEFAULT.exists():
        return KAGGLE_DEFAULT
    raise FileNotFoundError("CSV не найден. Скачайте датасет kagglehub или передайте --source.")


def load_df(source: Path) -> pd.DataFrame:
    df = pd.read_csv(source)
    df = df[df["category"].isin(CATEGORIES)]
    df = df[df["priority"].isin(PRIORITIES)]
    df["issue_description"] = df["issue_description"].fillna("").astype(str)
    df["resolution_notes"] = df["resolution_notes"].fillna("").astype(str)
    return df


def build_corpus(df: pd.DataFrame, n: int = 1200) -> pd.DataFrame:
    closed = df[df["status"].isin(["Closed", "Resolved"]) & (df["resolution_notes"].str.len() > 30)]
    parts = []
    per_cat = max(1, n // len(CATEGORIES))
    for cat in CATEGORIES:
        pool = closed[closed["category"] == cat]
        if pool.empty:
            continue
        parts.append(pool.sample(n=min(per_cat, len(pool)), random_state=42))
    corpus = pd.concat(parts, ignore_index=True).drop_duplicates("ticket_id")
    return corpus.head(n)


def build_eval_cases(df: pd.DataFrame) -> list[dict]:
    eval_cases: list[dict] = []
    used_ids: set[int] = set()
    case_id = 1
    closed = df[df["status"].isin(["Closed", "Resolved"]) & (df["resolution_notes"].str.len() > 40)]
    for cat in CATEGORIES:
        pool = closed[(closed["category"] == cat) & (~closed["ticket_id"].isin(used_ids))]
        consistent = pool[pool["issue_description"].apply(lambda t: _matches_category(str(t), cat))]
        if len(consistent) >= 3:
            pool = consistent
        if pool.empty:
            continue
        row = pool.sample(n=1, random_state=100 + case_id).iloc[0]
        tid = int(row["ticket_id"])
        used_ids.add(tid)
        floor = CATEGORY_PRIORITY_FLOOR.get(cat, "Low")
        eval_cases.append(
            {
                "id": case_id,
                "ticket_id": tid,
                "subject": f"{cat}: {row['product']}",
                "issue_description": row["issue_description"],
                "product": row["product"],
                "channel": row["channel"],
                "customer_segment": row["customer_segment"],
                "subscription_type": row["subscription_type"],
                "expected_category": cat,
                "expected_priority_min": floor,
                "expected_tools": ["search_similar_tickets", "lookup_sla_policy"],
                "gold_resolution_theme": row["resolution_notes"][:200],
                "comment": f"Hold-out тикет категории {cat}",
            }
        )
        case_id += 1
    for s in SYNTHETIC_CASES:
        eval_cases.append(
            {
                "id": case_id,
                "ticket_id": None,
                "subject": s["subject"],
                "issue_description": s["issue_description"],
                "product": s["product"],
                "channel": s["channel"],
                "customer_segment": s["customer_segment"],
                "subscription_type": s["subscription_type"],
                "expected_category": s["expected_category"],
                "expected_priority_min": s["expected_priority_min"],
                "expected_tools": ["search_similar_tickets", "lookup_sla_policy"],
                "gold_resolution_theme": s["gold_resolution_theme"],
                "comment": s["comment"],
                "synthetic": True,
            }
        )
        case_id += 1
    return eval_cases


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=None)
    ap.add_argument("--corpus-size", type=int, default=1200)
    args = ap.parse_args(argv)

    source = resolve_source(args.source)
    df = load_df(source)
    INPUT.mkdir(parents=True, exist_ok=True)

    corpus = build_corpus(df, n=args.corpus_size)
    corpus.to_csv(INPUT / "tickets_corpus.csv", index=False)

    eval_cases = build_eval_cases(df)
    (INPUT / "eval_cases.json").write_text(json.dumps(eval_cases, ensure_ascii=False, indent=2), encoding="utf-8")
    (INPUT / "sla_policies.json").write_text(json.dumps(SLA_POLICIES, ensure_ascii=False, indent=2), encoding="utf-8")
    (INPUT / "category_rules.json").write_text(json.dumps(CATEGORY_PRIORITY_FLOOR, ensure_ascii=False, indent=2), encoding="utf-8")

    meta = {
        "source": str(source),
        "total_rows": len(df),
        "corpus_rows": len(corpus),
        "eval_cases": len(eval_cases),
        "categories": CATEGORIES,
        "priorities": PRIORITIES,
    }
    (INPUT / "dataset_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
