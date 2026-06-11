import argparse
import json
from pathlib import Path

from pipeline import ingest, retrieve

GOLD_PATH = Path(__file__).parent / "data" / "gold.json"


def load_gold() -> list[dict]:
    return json.loads(GOLD_PATH.read_text(encoding="utf-8"))


def hit_rate(retrieved_ids: list[str], gold_sources: list[str]) -> float:
    retrieved_sources = {rid.split("__")[0] for rid in retrieved_ids}
    found = [g for g in gold_sources if g in retrieved_sources]
    return len(found) / len(gold_sources)


def run_eval(strategy: str, k: int = 5, verbose: bool = True) -> dict:
    gold = load_gold()
    total = 0.0
    results = []

    label = "FIXED (A)" if strategy == "fixed" else "RECURSIVE (B)"
    if verbose:
        print(f"\n=== {label} ===\n")

    for item in gold:
        q = item["question"]
        gold_sources = item["gold_sources"]

        hits = retrieve(q, k=k, strategy=strategy)
        retrieved_ids = hits["ids"][0]
        retrieved_sources = [rid.split("__")[0] for rid in retrieved_ids]

        score = hit_rate(retrieved_ids, gold_sources)
        total += score

        results.append(
            {
                "id": item["id"],
                "type": item["type"],
                "question": q,
                "score": score,
                "gold": gold_sources,
                "retrieved_ids": retrieved_ids,
                "retrieved_sources": retrieved_sources,
            }
        )

        if verbose:
            mark = "✓" if score == 1.0 else ("◐" if score > 0 else "✗")
            print(
                f"  [{item['id']:2d}] {item['type']:12s}  "
                f"hit@{k} = {score:.2f}  {mark}  {q[:60]}"
            )

    mean = total / len(gold)
    if verbose:
        print(f"\n  ИТОГО: hit-rate@{k} = {mean:.2f}  ({total:.1f} / {len(gold)})")

    return {"strategy": strategy, "mean": mean, "total": total, "results": results}


def compare(k: int = 5) -> dict:
    summary = {}
    for strategy in ("fixed", "recursive"):
        ingest(strategy)
        summary[strategy] = run_eval(strategy, k=k)

    print("\n" + "=" * 50)
    print("СРАВНЕНИЕ СТРАТЕГИЙ")
    print("=" * 50)
    print(f"  Fixed (A):     hit-rate@{k} = {summary['fixed']['mean']:.2f}")
    print(f"  Recursive (B): hit-rate@{k} = {summary['recursive']['mean']:.2f}")

    winner = (
        "recursive"
        if summary["recursive"]["mean"] > summary["fixed"]["mean"]
        else "fixed"
    )
    if summary["recursive"]["mean"] == summary["fixed"]["mean"]:
        print("  Итог: стратегии показали одинаковый результат")
    else:
        print(f"  Итог: выиграла стратегия {winner}")

    out_path = Path(__file__).parent / "eval_results.json"
    out_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nРезультаты сохранены в {out_path.name}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", choices=["fixed", "recursive"], default=None)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if args.strategy:
        ingest(args.strategy)
        run_eval(args.strategy, k=args.k, verbose=not args.quiet)
    else:
        compare(k=args.k)


if __name__ == "__main__":
    main()
