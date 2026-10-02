from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

@dataclass(frozen=True)
class Source:
    id: str
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float | None = None

@dataclass
class Doc:
    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    source: Source | None = None
    def as_source(self, score: float | None = None) -> Source:
        return self.source or Source(self.id, dict(self.metadata), score)

@dataclass
class Trace:
    step: str
    detail: str = ""

@dataclass
class RAGResult:
    answer: str
    docs: list[Doc] = field(default_factory=list)
    sources: list[Source] = field(default_factory=list)
    traces: list[Trace] = field(default_factory=list)
    strategy: str = ""
    def __post_init__(self) -> None:
        if not self.sources: self.sources = [d.as_source() for d in self.docs]
    def to_dict(self) -> dict[str, Any]:
        return {"answer": self.answer, "strategy": self.strategy, "sources":[{"id":s.id,"metadata":s.metadata,"score":s.score} for s in self.sources], "traces":[{"step":t.step,"detail":t.detail} for t in self.traces]}

class LLM(Protocol):
    def complete(self, prompt: str, system: str | None = None) -> str: ...
class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...
class VectorStore(Protocol):
    def add(self, docs: Sequence[Doc], vectors: Sequence[Sequence[float]]) -> None: ...
    def search(self, vector: Sequence[float], top_k: int = 4) -> list[tuple[Doc,float]]: ...
class Strategy(Protocol):
    def run(self, query: str, **kwargs: Any) -> RAGResult: ...
# Backward-compatible spelling used by early examples.
VectoreStore = VectorStore
