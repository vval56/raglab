from __future__ import annotations
import hashlib, math, re
from collections import Counter
from typing import Sequence
class HashEmbedder:
    def __init__(self, dimensions: int = 256): self.dimensions = dimensions
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        out=[]
        for text in texts:
            v=[0.0]*self.dimensions
            for token in re.findall(r"\w+", text.lower()):
                h=int.from_bytes(hashlib.blake2b(token.encode(),digest_size=8).digest(),'big')%self.dimensions
                v[h]+=1.0
            norm=math.sqrt(sum(x*x for x in v)) or 1.0
            out.append([x/norm for x in v])
        return out
class TfidfEmbedder:
    def __init__(self, dimensions: int = 512): self.dimensions=dimensions; self.idf: dict[str,float]={}
    def fit(self, texts: Sequence[str]) -> TfidfEmbedder:
        n=len(texts)
        # document frequency
        counts=Counter()
        for t in texts: counts.update(set(re.findall(r"\w+",t.lower())))
        self.idf={k:math.log((n+1)/(v+1))+1 for k,v in counts.items()}; return self
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not self.idf: self.fit(texts)
        out=[]
        for t in texts:
            v=[0.0]*self.dimensions; words=Counter(re.findall(r"\w+",t.lower()))
            for w,c in words.items():
                h=int.from_bytes(hashlib.blake2b(w.encode(),digest_size=8).digest(),'big')%self.dimensions; v[h]+=c*self.idf.get(w,1.0)
            norm=math.sqrt(sum(x*x for x in v)) or 1.0; out.append([x/norm for x in v])
        return out
