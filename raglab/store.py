from __future__ import annotations
import math
from typing import Sequence
from .core import Doc
class CosineVectorStore:
    def __init__(self): self._items:list[tuple[Doc,list[float]]]=[]
    def add(self, docs: Sequence[Doc], vectors: Sequence[Sequence[float]]) -> None:
        if len(docs)!=len(vectors): raise ValueError("docs and vectors length mismatch")
        self._items.extend((d,list(v)) for d,v in zip(docs,vectors))
    def clear(self): self._items.clear()
    def search(self, vector: Sequence[float], top_k:int=4):
        if top_k<1: return []
        norm=math.sqrt(sum(x*x for x in vector)) or 1.0
        scored=[]
        for doc,v in self._items:
            vn=math.sqrt(sum(x*x for x in v)) or 1.0
            score=sum(a*b for a,b in zip(vector,v))/(norm*vn); scored.append((doc,score))
        return sorted(scored,key=lambda x:x[1],reverse=True)[:top_k]
    def __len__(self): return len(self._items)
