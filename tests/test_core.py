from raglab import *
def test_core_store_and_chunk():
    docs=[Doc('a','alpha beta'),Doc('b','gamma')]; e=HashEmbedder(32); s=CosineVectorStore(); s.add(docs,e.embed([d.text for d in docs]))
    assert s.search(e.embed(['alpha'])[0],1)[0][0].id=='a'; assert chunk_text('a b c',2,0)==['a b','c']

def test_result_serialization():
    assert RAGResult('x',[Doc('a','t')]).to_dict()['sources'][0]['id']=='a'
