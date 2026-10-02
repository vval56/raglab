"""Interactive RAG demonstration against a local LM Studio model."""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from raglab import Doc, HashEmbedder, LMStudioLLM, CosineVectorStore
from raglab.strategies import (
    AgenticRAG,
    GraphRAG,
    MemoryRAG,
    MultiQueryRAG,
    SimpleRAG,
)


DOCS = [
    Doc("policy", "Возврат товара возможен в течение 30 дней при наличии чека."),
    Doc("exception", "Для товаров с производственным дефектом срок обращения составляет 90 дней."),
    Doc("delivery", "Стандартная доставка занимает 3 рабочих дня. Экспресс-доставка занимает 1 рабочий день."),
    Doc("energy-aep", "AEP — почасовой набор данных энергопотребления. Единица измерения: MW."),
    Doc("energy-link", "Пиковая нагрузка обычно наблюдается в будние дни вечером."),
    Doc(
        "graph-company",
        "Компания Northwind использует стандартную доставку для заказов из региона Delta.",
    ),
    Doc(
        "graph-region",
        "Регион Delta подключён к энергосети AEP, а AEP публикует почасовые измерения MW.",
    ),
]


def runner(cls, llm):
    store = CosineVectorStore()
    instance = cls(llm, HashEmbedder(), store, top_k=4, max_steps=3)
    instance.index(DOCS)
    return instance


def show(title: str, result) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")
    print("ОТВЕТ:\n", result.answer)
    print("\nИСТОЧНИКИ:")
    for source in result.sources:
        print(f"  - {source.id}: {source.metadata}")
    print("\nТРАССИРОВКА:")
    for trace in result.traces:
        print(f"  - {trace.step}: {trace.detail}")


def main() -> int:
    model = os.getenv("LMSTUDIO_MODEL", "google/gemma-4-e2b")
    llm = LMStudioLLM(
        model=model,
        timeout=float(os.getenv("LMSTUDIO_TIMEOUT", "120")),
        max_tokens=int(os.getenv("LMSTUDIO_MAX_TOKENS", "256")),
    )
    print(f"Модель: {model}")

    show(
        "1. SIMPLE RAG: точный вопрос",
        runner(__import__("raglab").SimpleRAG, llm).run("Сколько дней можно вернуть товар?"),
    )
    show(
        "2. MEMORY RAG: вопрос с местоимением",
        runner(MemoryRAG, llm).run(
            "А если у него дефект?",
            history=[
                {"role": "user", "content": "Я купил товар вчера."},
                {"role": "assistant", "content": "Уточните, есть ли чек."},
            ],
        ),
    )
    show(
        "3. MULTI-QUERY RAG: широкий запрос",
        runner(MultiQueryRAG, llm).run("Как связаны доставка, регион Delta и энергопотребление?"),
    )
    show(
        "4. GRAPH RAG: поиск по связям",
        runner(GraphRAG, llm).run("Какая энергосеть связана с заказами Northwind?"),
    )
    show(
        "5. AGENTIC RAG: план поиска",
        runner(AgenticRAG, llm).run("Сравни стандартную и экспресс-доставку и укажи сроки."),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
