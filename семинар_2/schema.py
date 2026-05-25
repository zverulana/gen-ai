"""
Pydantic-схема заявки на курс ДПО.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

CITIES = frozenset(
    {
        "Москва",
        "Санкт-Петербург",
        "Новосибирск",
        "Екатеринбург",
        "Казань",
        "Нижний Новгород",
        "Самара",
        "Краснодар",
        "Ростов-на-Дону",
        "Воронеж",
    }
)

SPECIALITIES = (
    "учитель",
    "врач",
    "бухгалтер",
    "инженер",
    "менеджер",
    "IT-специалист",
    "юрист",
    "экономист",
    "HR-специалист",
)

DESIRED_COURSES = (
    "цифровая грамотность",
    "управление проектами",
    "налоговое право",
    "медицинская документация",
    "педагогика ДО",
    "DevOps",
    "Excel для аналитиков",
)

Speciality = Literal[
    "учитель",
    "врач",
    "бухгалтер",
    "инженер",
    "менеджер",
    "IT-специалист",
    "юрист",
    "экономист",
    "HR-специалист",
]

DesiredCourse = Literal[
    "цифровая грамотность",
    "управление проектами",
    "налоговое право",
    "медицинская документация",
    "педагогика ДО",
    "DevOps",
    "Excel для аналитиков",
]

CURRENT_YEAR = date.today().year


class Address(BaseModel):
    city: str
    district: str = Field(min_length=2, max_length=60)

    @field_validator("city")
    @classmethod
    def city_must_be_in_list(cls, v: str) -> str:
        if v not in CITIES:
            raise ValueError(f"Город «{v}» не из утверждённого списка: {sorted(CITIES)}")
        return v


class Application(BaseModel):
    full_name: str = Field(min_length=5, max_length=120)
    age: int = Field(ge=22, le=65)
    address: Address
    speciality: Speciality
    desired_course: DesiredCourse
    years_of_experience: int = Field(ge=0, le=40)
    graduation_year: int = Field(ge=1980, le=2024)

    @model_validator(mode="after")
    def graduation_year_matches_age(self) -> Application:
        """Выпуск не раньше, чем через 22 года после рождения (≈ current_year - age)."""
        birth_year = CURRENT_YEAR - self.age
        min_graduation = birth_year + 22
        if self.graduation_year < min_graduation:
            raise ValueError(
                f"graduation_year={self.graduation_year} слишком ранний для age={self.age}: "
                f"ожидается ≥ {min_graduation}"
            )
        if self.graduation_year > CURRENT_YEAR:
            raise ValueError(
                f"graduation_year={self.graduation_year} не может быть позже {CURRENT_YEAR}"
            )
        max_experience = max(0, self.age - 22)
        if self.years_of_experience > max_experience + 5:
            raise ValueError(
                f"years_of_experience={self.years_of_experience} несовместим с age={self.age}"
            )
        return self

    @property
    def city(self) -> str:
        return self.address.city
