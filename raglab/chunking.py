from __future__ import annotations
def chunk_text(text: str, chunk_size:int=500, overlap:int=50) -> list[str]:
    if chunk_size<=0 or overlap<0 or overlap>=chunk_size: raise ValueError("invalid chunk parameters")
    words=text.split()
    if not words:return []
    result=[]; step=chunk_size-overlap
    for i in range(0,len(words),step):
        part=" ".join(words[i:i+chunk_size]);
        if part: result.append(part)
        if i+chunk_size>=len(words): break
    return result
