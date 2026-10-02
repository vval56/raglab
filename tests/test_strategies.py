import pytest
from raglab import *
class Fake:
 def complete(self,prompt,system=None): return '1\n2\n3' if 'по одному' in prompt else 'answer'
@pytest.mark.parametrize('name',list(STRATEGIES))
def test_strategy(name):
 e=HashEmbedder(); s=CosineVectorStore(); cls=STRATEGIES[name]; x=cls(Fake(),e,s); x.index([Doc('x','some context')]); r=x.run('question'); assert r.answer and r.strategy==name and r.sources

def test_memory_rewrites_query():
 class MemoryFake(Fake):
     def complete(self, prompt, system=None):
         if 'Переформулируй' in prompt:
             return 'some context'
         return 'answer'
 e, s = HashEmbedder(), CosineVectorStore()
 runner = MemoryRAG(MemoryFake(), e, s)
 runner.index([Doc('x', 'some context')])
 result = runner.run('Что это?', history=[{'role': 'user', 'content': 'some context'}])
 assert any(trace.step == 'memory' and 'some context' in trace.detail for trace in result.traces)

def test_corrective_retries_when_no_relevant_hits():
 class CorrectiveFake(Fake):
     def complete(self, prompt, system=None):
         if 'Уточни' in prompt:
             return 'some context'
         return 'answer'
 e, s = HashEmbedder(), CosineVectorStore()
 runner = CorrectiveRAG(CorrectiveFake(), e, s)
 runner.index([Doc('x', 'some context')])
 result = runner.run('unrelated')
 assert any(trace.step == 'corrective_retry' for trace in result.traces)

def test_graph_expands_related_documents():
 e, s = HashEmbedder(), CosineVectorStore()
 runner = GraphRAG(Fake(), e, s)
 runner.index([Doc('a', 'alpha relation'), Doc('b', 'alpha connected')])
 result = runner.run('alpha')
 assert any(trace.step == 'graph' and 'nodes=2' in trace.detail for trace in result.traces)
