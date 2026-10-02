from __future__ import annotations
import json, urllib.request
class LMStudioLLM:
    def __init__(self, base_url='http://localhost:1234/v1', model='local-model', timeout=60, max_tokens=256):
        self.base_url=base_url.rstrip('/')
        self.model=model
        self.timeout=timeout
        self.max_tokens=max_tokens
    def complete(self,prompt:str,system:str|None=None)->str:
        messages=([{"role":"system","content":system}] if system else []) + [{"role":"user","content":prompt}]
        data=json.dumps({"model":self.model,"messages":messages,"max_tokens":self.max_tokens}).encode()
        req=urllib.request.Request(self.base_url+'/chat/completions',data=data,headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(req,timeout=self.timeout) as r: obj=json.loads(r.read())
        return str(obj['choices'][0]['message']['content'])
OpenAICompatibleLLM = LMStudioLLM
LMStudioAdapter = LMStudioLLM
