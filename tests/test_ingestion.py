from raglab.ingestion import load_text,load_csv

def test_ingestion(tmp_path):
    p=tmp_path/'a.md'; p.write_text('# hi',encoding='utf8'); assert load_text(p)[0].text=='# hi'
    c=tmp_path/'a.csv'; c.write_text('q,a\nhello,world\n',encoding='utf8'); assert 'hello' in load_csv(c)[0].text
