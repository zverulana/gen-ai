# **Support Ticket Intelligence (финальный проект)**

Автоматическая маршрутизация и черновик ответа по тикетам поддержки на датасете [Customer Support Tickets 200K](https://www.kaggle.com/datasets/mirzayasirabdullah07/customer-support-tickets-dataset-200k-records).

## **Одна команда запуска**

```bash
cp .env.example .env
pip3 install -r requirements.txt
python3 pipeline.py full
python3 eval.py
```

pipeline.py full выполняет: подготовку данных - индексацию RAG - генерацию синтетических персон - прогон всех кейсов.

## **Структура**

| Файл / папка | Назначение |
|---|---|
| prepare_data.py | Сэмплирование корпуса и eval-кейсов из Kaggle CSV |
| rag.py | ChromaDB RAG по resolved-тикетам |
| schemas.py | Pydantic-схемы + field_validator |
| agents.py | Мультиагент: triage - response (tools) - critic |
| personas.py | Синтетические персоны клиентов |
| hallucination.py | Ghost-цитаты и выдуманные SLA-числа |
| pipeline.py | CLI оркестратор |
| eval.py | 18 тестов: категория, приоритет, путь, LLM-judge |
| input/ | tickets_corpus.csv, eval_cases.json, SLA-политики |
| output/ | eval_results.json, trace.jsonl, pipeline_results.json |
| отчёт.md | Отчет по проекту |

## **Техники курса**

1. RAG - search_similar_tickets (ChromaDB + multilingual embeddings)
2. Агент с инструментами - response-агент: RAG, SLA, статистика категорий
3. Мультиагент - triage-агент, response-агент, critic-агент
4. LLM-as-judge - оценка качества ответа в eval.py
5. Синтетические персоны - personas.py для расширения тестов

## **Отдельные команды**

```bash
python3 prepare_data.py
python3 pipeline.py ingest
python3 pipeline.py run --case-id 1
python3 pipeline.py personas
```

## **Переменные окружения**

См. .env.example. Без LLM_AUTH_TOKEN пайплайн не запустится.
