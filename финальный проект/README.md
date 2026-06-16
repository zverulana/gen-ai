# **Support Ticket Intelligence**

Автоматическая маршрутизация и черновик ответа по тикетам поддержки на датасете [Customer Support Tickets 200K](https://www.kaggle.com/datasets/mirzayasirabdullah07/customer-support-tickets-dataset-200k-records).

## **Команда запуска**

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

