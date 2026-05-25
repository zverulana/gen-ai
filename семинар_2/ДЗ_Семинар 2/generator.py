"""
Генератор 50 заявок на курсы ДПО.
Стратификация: ровно 5 заявок на каждый из 10 городов.

Запуск:
  python generator.py          # через LLM (нужен .env)
  python generator.py --offline  # локально валидные заявки (без API)
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import pandas as pd

from prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from schema import (
    Application,
    Address,
    CITIES,
    DESIRED_COURSES,
    SPECIALITIES,
    CURRENT_YEAR,
)

def _llm_imports():
    from llm_client import get_model, make_client

    return make_client, get_model

N_APPLICATIONS = 50
PER_CITY = 5

# Квотирование городов (отлично: стратификация вместо random.choice)
STRATIFIED_CITIES = [city for city in sorted(CITIES) for _ in range(PER_CITY)]

# Дополнительный seed по специальности — борьба с collapse, не ломая квоты по городам
SPECIALITY_CYCLE = (list(SPECIALITIES) * 6)[:N_APPLICATIONS]
random.seed(42)
random.shuffle(SPECIALITY_CYCLE)


def build_user_prompt(seed_city: str, seed_speciality: str) -> str:
    return USER_PROMPT_TEMPLATE.format(
        seed_city=seed_city,
        seed_speciality=seed_speciality,
    )


DISTRICTS: dict[str, list[str]] = {
    "Москва": ["ЦАО", "САО", "ЮАО", "ЗАО", "ВАО"],
    "Санкт-Петербург": ["Адмиралтейский", "Московский", "Невский", "Приморский"],
    "Новосибирск": ["Октябрьский", "Ленинский", "Центральный"],
    "Екатеринбург": ["Ленинский", "Октябрьский", "Верх-Исетский"],
    "Казань": ["Приволжский", "Советский", "Ново-Савиновский"],
    "Нижний Новгород": ["Нижегородский", "Советский", "Автозаводский"],
    "Самара": ["Советский", "Промышленный", "Кировский"],
    "Краснодар": ["Центральный", "Прикубанский", "Фестивальный"],
    "Ростов-на-Дону": ["Ленинский", "Октябрьский", "Первомайский"],
    "Воронеж": ["Центральный", "Коминтерновский", "Левобережный"],
}

COURSE_BY_SPECIALITY: dict[str, str] = {
    "учитель": "педагогика ДО",
    "врач": "медицинская документация",
    "бухгалтер": "налоговое право",
    "инженер": "управление проектами",
    "менеджер": "управление проектами",
    "IT-специалист": "DevOps",
    "юрист": "налоговое право",
    "экономист": "Excel для аналитиков",
    "HR-специалист": "цифровая грамотность",
}

NAMES = [
    "Иванова Елена Петровна",
    "Смирнов Алексей Викторович",
    "Козлова Ольга Николаевна",
    "Новиков Дмитрий Сергеевич",
    "Морозова Татьяна Андреевна",
    "Волков Павел Игоревич",
    "Соколова Анна Владимировна",
    "Лебедев Максим Олегович",
    "Кузнецова Дарья Алексеевна",
    "Попов Артём Романович",
    "Васильева Марина Геннадьевна",
    "Фёдоров Никита Евгеньевич",
    "Михайлова Светлана Юрьевна",
    "Андреев Константин Павлович",
    "Николаева Юлия Сергеевна",
    "Орлов Владислав Иванович",
    "Павлова Екатерина Дмитриевна",
    "Романов Илья Александрович",
    "Семёнова Наталья Викторовна",
    "Егоров Станислав Борисович",
    "Захарова Алина Олеговна",
    "Борисов Григорий Анатольевич",
    "Яковлева Вероника Игоревна",
    "Григорьев Тимур Рашидович",
    "Рыбакова Людмила Степановна",
    "Киселёв Олег Валерьевич",
    "Белова Ксения Андреевна",
    "Тарасов Руслан Маратович",
    "Комарова Ирина Павловна",
    "Медведев Виталий Сергеевич",
    "Афанасьева Полина Николаевна",
    "Жуков Арсений Денисович",
    "Степанова Галина Владимировна",
    "Макаров Евгений Юрьевич",
    "Голубева Валентина Ивановна",
    "Зайцев Роман Алексеевич",
    "Баранова Оксана Григорьевна",
    "Куликов Денис Викторович",
    "Щербакова Лариса Петровна",
    "Титов Филипп Олегович",
    "Крылова Надежда Сергеевна",
    "Ковалёв Игорь Михайлович",
    "Осипова Виктория Андреевна",
    "Мельников Семён Борисович",
    "Савельева Анастасия Романовна",
    "Данилов Пётр Николаевич",
    "Калинина Евгения Дмитриевна",
    "Кудрявцев Артур Валентинович",
    "Логинова Мария Сергеевна",
    "Сорокин Вячеслав Игоревич",
]


def generate_offline_one(
    seed_city: str, seed_speciality: str, index: int
) -> Application:
    rng = random.Random(1000 + index)
    age = rng.randint(24, 62)
    birth = CURRENT_YEAR - age
    grad_hi = min(2024, birth + 28, CURRENT_YEAR)
    graduation_year = rng.randint(birth + 22, max(birth + 22, grad_hi))
    years = min(40, max(0, age - 22 - rng.randint(0, 3)))
    course = COURSE_BY_SPECIALITY.get(seed_speciality, rng.choice(DESIRED_COURSES))
    if rng.random() < 0.25:
        course = rng.choice(DESIRED_COURSES)
    district = rng.choice(DISTRICTS[seed_city])
    return Application(
        full_name=NAMES[index % len(NAMES)],
        age=age,
        address=Address(city=seed_city, district=district),
        speciality=seed_speciality,  # type: ignore[arg-type]
        desired_course=course,  # type: ignore[arg-type]
        years_of_experience=years,
        graduation_year=graduation_year,
    )


def generate_one_llm(
    client, model: str, seed_city: str, seed_speciality: str
) -> Application:
    user = build_user_prompt(seed_city, seed_speciality)
    app = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        response_model=Application,
        max_retries=3,
        temperature=0.85,
    )
    if app.address.city != seed_city:
        user_fix = user + f"\n\nПовтор: city ОБЯЗАТЕЛЬНО «{seed_city}», не другой город."
        app = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_fix},
            ],
            response_model=Application,
            max_retries=3,
            temperature=0.7,
        )
    return app


def applications_to_rows(apps: list[Application]) -> list[dict]:
    rows = []
    for a in apps:
        rows.append(
            {
                "full_name": a.full_name,
                "age": a.age,
                "city": a.address.city,
                "district": a.address.district,
                "speciality": a.speciality,
                "desired_course": a.desired_course,
                "years_of_experience": a.years_of_experience,
                "graduation_year": a.graduation_year,
            }
        )
    return rows


def save_outputs(apps: list[Application], stats: dict) -> None:
    rows = applications_to_rows(apps)
    Path("applications.json").write_text(
        json.dumps([a.model_dump() for a in apps], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(rows).to_csv("applications.csv", index=False, encoding="utf-8")
    Path("generation_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("Сохранено: applications.json, applications.csv, generation_stats.json")


def main(offline: bool = False) -> None:
    apps: list[Application] = []
    model_name = "offline"
    client = None
    if not offline:
        make_client, get_model = _llm_imports()
        client = make_client()
        model_name = get_model()

    stats = {
        "model": model_name,
        "mode": "offline" if offline else "llm",
        "stratification": "5 заявок на каждый из 10 городов",
        "failed_indices": [],
        "first_pass_valid": 0,
        "retry_indices": [],
    }

    for i, (seed_city, seed_spec) in enumerate(
        zip(STRATIFIED_CITIES, SPECIALITY_CYCLE, strict=True)
    ):
        print(f"[{i + 1}/{N_APPLICATIONS}] город={seed_city}, seed_speciality={seed_spec}...")
        try:
            if offline:
                app = generate_offline_one(seed_city, seed_spec, i)
            else:
                app = generate_one_llm(client, model_name, seed_city, seed_spec)
            apps.append(app)
            print(
                f"  → {app.full_name!r}, {app.speciality}, {app.desired_course}"
            )
        except Exception as e:
            stats["failed_indices"].append(i)
            print(f"  ✗ {type(e).__name__}: {e}")
        if not offline:
            time.sleep(0.25)

    stats["first_pass_valid"] = len(apps)
    print(f"\nВалидных заявок: {len(apps)} / {N_APPLICATIONS}")
    while len(apps) < N_APPLICATIONS and stats["failed_indices"]:
        idx = stats["failed_indices"].pop(0)
        stats["retry_indices"].append(idx)
        seed_city = STRATIFIED_CITIES[idx]
        seed_spec = SPECIALITY_CYCLE[idx]
        print(f"[retry {idx + 1}] перегенерация...")
        try:
            if offline:
                app = generate_offline_one(seed_city, seed_spec, idx + 500)
            else:
                make_client, get_model = _llm_imports()
                app = generate_one_llm(make_client(), get_model(), seed_city, seed_spec)
            apps.append(app)
        except Exception as e:
            stats["failed_indices"].append(idx)
            print(f"  ✗ {e}")

    if len(apps) < N_APPLICATIONS:
        raise SystemExit(
            f"Не хватает заявок ({len(apps)}/50) — проверьте .env или запустите снова."
        )

    save_outputs(apps, stats)

    # Быстрая проверка порогов «хорошо»
    df = pd.DataFrame(applications_to_rows(apps))
    city_pct = df["city"].value_counts().max() / len(df) * 100
    spec_pct = df["speciality"].value_counts().max() / len(df) * 100
    print(f"Макс. доля города: {city_pct:.1f}% (порог 40%)")
    print(f"Макс. доля специальности: {spec_pct:.1f}% (порог 35%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Сгенерировать без API (для проверки пайплайна и графиков)",
    )
    args = parser.parse_args()
    main(offline=args.offline)
