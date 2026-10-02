from __future__ import annotations
from .core import Doc
from .embeddings import HashEmbedder
from .store import CosineVectorStore
from .strategies import STRATEGIES, SimpleRAG
from .core import LLM
from .lmstudio import LMStudioLLM

class RAGService:
    def __init__(self, llm: LLM | None = None):
        self.embedder=HashEmbedder()
        self.store=CosineVectorStore()
        self.docs=[]
        self.llm=llm or _EchoLLM()
    def index_documents(self, documents:list[dict]|list[str])->dict:
        self.docs=[d if isinstance(d,Doc) else Doc(str(i), d.get('text',''), d.get('metadata',{})) if isinstance(d,dict) else Doc(str(i),d) for i,d in enumerate(documents)]
        self.store.clear(); self.store.add(self.docs,self.embedder.embed([d.text for d in self.docs])); return {'count':len(self.docs)}
    def list_strategies(self)->list[str]: return list(STRATEGIES)
    def query_rag(self,query:str,strategy='simple',top_k=4)->dict:
        cls=STRATEGIES.get(strategy,SimpleRAG)
        runner=cls(self.llm,self.embedder,self.store,top_k=top_k)
        # The service owns the already-built index; avoid adding the same
        # vectors again just to provide graph-aware strategies their corpus.
        runner._docs = list(self.docs)
        return runner.run(query).to_dict()
    def compare_strategies(self,query:str)->dict: return {n:self.query_rag(query,n) for n in STRATEGIES}
class _EchoLLM:
    def complete(self,prompt,system=None): return 'Ответ на основе предоставленного контекста.'

def create_mcp(service=None):
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as e: raise RuntimeError('MCP integration requires optional dependency mcp') from e
    svc=service or RAGService(); m=FastMCP('raglab')
    m.tool()(svc.index_documents); m.tool()(svc.list_strategies); m.tool()(svc.query_rag); m.tool()(svc.compare_strategies); return m
def main():
    import os
    llm = None
    if os.getenv("RAGLAB_USE_LMSTUDIO", "").lower() in {"1", "true", "yes"}:
        llm = LMStudioLLM(
            base_url=os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1"),
            model=os.getenv("LMSTUDIO_MODEL", "local-model"),
        )
    create_mcp(RAGService(llm=llm)).run()
