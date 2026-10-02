from raglab.mcp_server import RAGService
def test_mcp_json_shape():
 s=RAGService(); s.index_documents([{'text':'hello','metadata':{'a':1}}]); x=s.query_rag('hello'); assert 'answer' in x and x['sources']; assert len(s.compare_strategies('hello'))==10
 assert x['sources'][0]['score'] is not None

def test_mcp_graph_sees_indexed_corpus():
 s = RAGService()
 s.index_documents([{'text': 'alpha relation'}, {'text': 'alpha connected'}])
 result = s.query_rag('alpha', 'graph_rag')
 assert any(t['step'] == 'graph' and 'nodes=2' in t['detail'] for t in result['traces'])
