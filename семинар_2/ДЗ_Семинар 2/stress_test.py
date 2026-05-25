"""
Стресс-тест: промпт «придумай оригинальный курс» vs desired_course: Literal[...].
"""

from __future__ import annotations

import time
from typing import Literal

from llm_client import get_model, make_client
from pydantic import BaseModel, Field

from schema import DESIRED_COURSES

client = make_client()
MODEL = get_model()

CONFLICT_PROMPT = (
    "Сгенерируй заявку на ДПО. В поле desired_course укажи УНИКАЛЬНЫЙ, "
    "ОРИГИНАЛЬНЫЙ курс, которого нет в стандартных каталогах — "
    "не используй типовые названия вроде «цифровая грамотность» или «DevOps»."
)


class StrictApplication(BaseModel):
    full_name: str
    age: int = Field(ge=22, le=65)
    desired_course: Literal[
        "цифровая грамотность",
        "управление проектами",
        "налоговое право",
        "медицинская документация",
        "педагогика ДО",
        "DevOps",
        "Excel для аналитиков",
    ]


def run(max_retries: int) -> None:
    print(f"\n━━━ max_retries={max_retries} ━━━")
    t0 = time.time()
    try:
        obj = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": CONFLICT_PROMPT},
                {"role": "user", "content": "Создай одну заявку, только JSON."},
            ],
            response_model=StrictApplication,
            max_retries=max_retries,
            temperature=0.9,
        )
        print(f"  ✓ за {time.time() - t0:.1f}s: desired_course={obj.desired_course!r}")
    except Exception as e:
        print(f"  ✗ за {time.time() - t0:.1f}s (~{max_retries + 1} запросов)")
        print(f"    {type(e).__name__}: {str(e)[:240]}")


def main() -> None:
    print(f"Модель: {MODEL}")
    print(f"Допустимые курсы в схеме: {list(DESIRED_COURSES)}")
    print("Промпт требует «оригинальный» курс вне каталога.\n")
    for r in (1, 3, 5):
        run(r)
    print("\nОжидание: при жёстком Literal модель часто всё равно выбирает значение из схемы")
    print("(JSON Schema в system). При max_retries=5 токены тратятся, успех не гарантирован.")


if __name__ == "__main__":
    main()
