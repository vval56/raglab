from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from .core import Doc, Embedder, LLM, RAGResult, Trace, VectorStore


class BaseRAG:
    name = "base"

    def __init__(
        self,
        llm: LLM,
        embedder: Embedder,
        store: VectorStore,
        top_k: int = 4,
        max_steps: int = 3,
    ):
        if top_k < 1 or max_steps < 1:
            raise ValueError("top_k and max_steps must be positive")
        self.llm, self.embedder, self.store = llm, embedder, store
        self.top_k, self.max_steps = top_k, max_steps
        self._docs: list[Doc] = []

    def index(self, docs: list[Doc] | list[str]) -> None:
        self._docs = [
            doc if isinstance(doc, Doc) else Doc(f"doc-{i}", doc)
            for i, doc in enumerate(docs)
        ]
        self.store.add(self._docs, self.embedder.embed([doc.text for doc in self._docs]))

    def retrieve(self, query: str, top_k: int | None = None) -> list[tuple[Doc, float]]:
        return self.store.search(
            self.embedder.embed([query])[0], top_k or self.top_k
        )

    def _complete(self, prompt: str, system: str | None = None, fallback: str = "") -> str:
        try:
            return self.llm.complete(prompt, system=system).strip()
        except (TimeoutError, ConnectionError, OSError, KeyError, ValueError):
            return fallback

    def answer(
        self,
        query: str,
        docs: list[tuple[Doc, float]],
        traces: list[Trace],
        system: str | None = None,
    ) -> RAGResult:
        context = "\n---\n".join(doc.text for doc, _ in docs)
        traces.append(Trace("retrieve", f"{len(docs)} documents"))
        answer = self._complete(
            f"Контекст:\n{context}\n\nВопрос: {query}",
            system=system,
            fallback="Не удалось получить ответ от модели.",
        )
        return RAGResult(
            answer=answer,
            docs=[doc for doc, _ in docs],
            sources=[
                doc.as_source(score=score)
                for doc, score in docs
            ],
            traces=traces,
            strategy=self.name,
        )

    def run(self, query: str, **_: Any) -> RAGResult:
        return self.answer(query, self.retrieve(query), [])


class SimpleRAG(BaseRAG):
    name = "simple"


class MemoryRAG(BaseRAG):
    name = "memory"

    def run(self, query: str, history: list[dict[str, str]] | None = None, **_: Any) -> RAGResult:
        history = history or []
        transcript = "\n".join(
            f"{message.get('role', 'user')}: {message.get('content', '')}"
            for message in history
        )
        rewritten = query
        traces = []
        if transcript:
            rewritten = self._complete(
                f"Переформулируй последний вопрос без местоимений.\nИстория:\n{transcript}\n"
                f"Вопрос: {query}",
                fallback=query,
            )
            traces.append(Trace("memory", f"rewrote query to: {rewritten}"))
        else:
            traces.append(Trace("memory", "no history"))
        return self.answer(rewritten, self.retrieve(rewritten), traces)


class MultiQueryRAG(BaseRAG):
    name = "multi_query"

    def run(self, query: str, **_: Any) -> RAGResult:
        raw = self._complete(
            f"Сгенерируй {self.max_steps} поисковых запроса, по одному в строке: {query}",
            fallback=query,
        )
        queries = [line.strip("- •\t ") for line in raw.splitlines() if line.strip()]
        queries = list(dict.fromkeys(queries or [query]))[: self.max_steps]
        found: dict[str, tuple[Doc, float]] = {}
        for expanded in queries:
            for doc, score in self.retrieve(expanded):
                found.setdefault(doc.id, (doc, score))
        traces = [Trace("multi_query", f"fan-out={len(queries)}")]
        return self.answer(query, list(found.values())[: self.top_k], traces)


class HyDERAG(BaseRAG):
    name = "hyde"

    def run(self, query: str, **_: Any) -> RAGResult:
        hypothetical = self._complete(
            f"Напиши короткий гипотетический документ, отвечающий на вопрос: {query}",
            fallback=query,
        )
        return self.answer(
            query,
            self.retrieve(hypothetical),
            [Trace("hyde", "retrieved with hypothetical document")],
        )


class AdaptiveRAG(BaseRAG):
    name = "adaptive"

    def run(self, query: str, **_: Any) -> RAGResult:
        words = len(query.split())
        mode = "broad" if words > 8 or "?" not in query else "focused"
        depth = min(self.top_k + (2 if mode == "broad" else 0), self.top_k * 2)
        traces = [Trace("adaptive", f"route={mode}, top_k={depth}")]
        return self.answer(query, self.retrieve(query, depth), traces)


class CorrectiveRAG(BaseRAG):
    name = "corrective"

    def run(self, query: str, **_: Any) -> RAGResult:
        first = self.retrieve(query)
        relevant = [hit for hit in first if hit[1] > 0.05]
        traces = [Trace("corrective", f"initial_relevant={len(relevant)}")]
        if not relevant:
            rewritten = self._complete(
                f"Уточни поисковый запрос для поиска фактов: {query}", fallback=query
            )
            relevant = self.retrieve(rewritten)
            traces.append(Trace("corrective_retry", f"query={rewritten}"))
        return self.answer(query, relevant, traces)


class SelfRAG(BaseRAG):
    name = "self_rag"

    def run(self, query: str, **_: Any) -> RAGResult:
        docs = self.retrieve(query)
        draft = self._complete(
            f"Ответь по контексту и явно укажи, достаточно ли фактов:\n"
            f"{'---'.join(d.text for d, _ in docs)}\nВопрос: {query}",
            fallback="",
        )
        critique = self._complete(
            f"Проверь ответ на наличие неподтверждённых утверждений: {draft}",
            fallback="проверка недоступна",
        )
        traces = [Trace("self_rag_draft", draft[:200]), Trace("self_rag_critique", critique[:200])]
        result = self.answer(query, docs, traces)
        if draft:
            result.answer = draft
        return result


class AgenticRAG(BaseRAG):
    name = "agentic"

    def run(self, query: str, **_: Any) -> RAGResult:
        plan = self._complete(
            f"Составь не более {self.max_steps} шагов поиска для вопроса: {query}",
            fallback="retrieve",
        )
        docs: dict[str, tuple[Doc, float]] = {}
        for step in [line for line in plan.splitlines() if line.strip()][: self.max_steps]:
            for doc, score in self.retrieve(step):
                docs.setdefault(doc.id, (doc, score))
        if not docs:
            docs.update({doc.id: (doc, score) for doc, score in self.retrieve(query)})
        return self.answer(
            query,
            list(docs.values())[: self.top_k],
            [Trace("agentic_plan", plan[:300]), Trace("agentic", "bounded tool calls")],
        )


class MultimodalRAG(BaseRAG):
    name = "multimodal"

    def run(self, query: str, **_: Any) -> RAGResult:
        hits = self.retrieve(query)
        media = sum(1 for doc, _ in hits if doc.metadata.get("media_type"))
        traces = [Trace("multimodal", f"media_sources={media}")]
        return self.answer(query, hits, traces)


class GraphRAG(BaseRAG):
    name = "graph_rag"

    def _graph_hits(self, query: str) -> list[tuple[Doc, float]]:
        direct = self.retrieve(query)
        terms = set(re.findall(r"\w+", query.lower()))
        adjacency: dict[str, set[str]] = defaultdict(set)
        for left in self._docs:
            left_terms = set(re.findall(r"\w+", left.text.lower()))
            for right in self._docs:
                if left.id != right.id and left_terms & set(re.findall(r"\w+", right.text.lower())):
                    adjacency[left.id].add(right.id)
        seeds = {doc.id for doc, score in direct if score > 0}
        neighbors = {neighbor for seed in seeds for neighbor in adjacency[seed]}
        expanded = [(doc, 0.01) for doc in self._docs if doc.id in neighbors]
        return direct + expanded

    def run(self, query: str, **_: Any) -> RAGResult:
        hits = self._graph_hits(query)
        unique = list({doc.id: (doc, score) for doc, score in hits}.values())[: self.top_k]
        return self.answer(
            query,
            unique,
            [Trace("graph", f"nodes={len(self._docs)}, expanded={len(unique)}")],
        )


STRATEGIES = {
    strategy.name: strategy
    for strategy in (
        SimpleRAG,
        MemoryRAG,
        MultiQueryRAG,
        HyDERAG,
        AdaptiveRAG,
        CorrectiveRAG,
        SelfRAG,
        AgenticRAG,
        MultimodalRAG,
        GraphRAG,
    )
}
