"""
Расширенный анализ заявок 
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix == ".csv":
        return pd.read_csv(p, encoding="utf-8")
    import json

    with p.open(encoding="utf-8") as f:
        data = json.load(f)
    flat = []
    for item in data:
        row = dict(item)
        if isinstance(row.get("address"), dict):
            addr = row.pop("address")
            row.setdefault("city", addr.get("city"))
            row.setdefault("district", addr.get("district"))
        flat.append(row)
    return pd.DataFrame(flat)


def plot_bar(series: pd.Series, title: str, out: str, color: str) -> pd.Series:
    counts = series.value_counts()
    plt.figure(figsize=(9, 4))
    counts.plot.bar(color=color, edgecolor="white")
    plt.title(title)
    plt.ylabel("Число заявок")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()
    return counts


def cross_table(df: pd.DataFrame) -> pd.DataFrame:
    return pd.crosstab(df["city"], df["speciality"])


def write_report(df: pd.DataFrame, ct: pd.DataFrame, out: str) -> None:
    n = len(df)
    lines = [f"# Отчёт по {n} заявкам на ДПО\n"]

    cities = df["city"].value_counts()
    top_city_pct = cities.iloc[0] / n * 100
    lines.append("## Города\n")
    lines.append(f"- Уникальных: {len(cities)}")
    lines.append(f"- Топ-1: **{cities.index[0]}** — {cities.iloc[0]} ({top_city_pct:.1f}%)")
    if top_city_pct > 40:
        lines.append("- ⚠ Превышен порог 40% → mode collapse по городам")
    else:
        lines.append("- ✓ Порог 40% не превышен (стратификация 5×10)")
    lines.append("")

    spec = df["speciality"].value_counts()
    top_spec_pct = spec.iloc[0] / n * 100
    lines.append("## Специальности\n")
    lines.append(f"- Уникальных: {len(spec)}")
    lines.append(f"- Топ-1: **{spec.index[0]}** — {spec.iloc[0]} ({top_spec_pct:.1f}%)")
    if top_spec_pct > 35:
        lines.append("- ⚠ Превышен порог 35%")
    lines.append("")

    course = df["desired_course"].value_counts()
    lines.append("## Желаемые курсы\n")
    for name, cnt in course.items():
        lines.append(f"- {name}: {cnt}")
    lines.append("")

    names = df["full_name"].value_counts()
    dupes = names[names > 1]
    lines.append("## ФИО\n")
    lines.append(f"- Уникальных: {len(names)} из {n}")
    if len(dupes):
        lines.append(f"- Повторы: {dict(dupes.head(5))}")
    lines.append("")

    lines.append("## Кросс-таблица город × специальность\n")
    lines.append("```")
    lines.append(ct.to_string())
    lines.append("```\n")

    lines.append("## Подозрительные комбинации (для обсуждения)\n")
    suspicious = [
        ("врач", "DevOps", "редко, но возможно при смене карьеры"),
        ("учитель", "налоговое право", "малоправдоподобно без переквалификации"),
        ("IT-специалист", "медицинская документация", "скорее ошибка модели"),
    ]
    for spec, course, comment in suspicious:
        mask = (df["speciality"] == spec) & (df["desired_course"] == course)
        if mask.any():
            lines.append(f"- **{spec}** → **{course}**: {mask.sum()} шт. ({comment})")
    if not any(
        ((df["speciality"] == s) & (df["desired_course"] == c)).any()
        for s, c, _ in suspicious
    ):
        lines.append("- В данных нет экстремальных пар из списка — см. кросс-таблицу.")

    Path(out).write_text("\n".join(lines), encoding="utf-8")


def main(path: str = "applications.csv") -> None:
    df = load(path)
    print(f"Загружено: {len(df)} заявок из {path}")

    plot_bar(df["city"], "Распределение по городам", "cities.png", "#7AB66E")
    plot_bar(
        df["speciality"],
        "Распределение по специальностям",
        "specialities.png",
        "#D97A4A",
    )
    ct = cross_table(df)
    write_report(df, ct, "report.md")

    print("Сохранено: cities.png, specialities.png, report.md")
    print(f"Топ-город: {df['city'].value_counts().index[0]}")
    print(f"Топ-специальность: {df['speciality'].value_counts().index[0]}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "applications.csv")
