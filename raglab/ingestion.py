from __future__ import annotations
import csv
from pathlib import Path
from .core import Doc

def load_text(path: str|Path) -> list[Doc]:
    p=Path(path); return [Doc(str(p),p.read_text(encoding='utf-8'),{"path":str(p)})]
def load_csv(path: str|Path, text_columns: list[str]|None=None) -> list[Doc]:
    p=Path(path); rows=[]
    with p.open(newline='',encoding='utf-8') as f:
        for i,row in enumerate(csv.DictReader(f)):
            cols=text_columns or list(row); text='\n'.join(f'{k}: {row[k]}' for k in cols if k in row)
            rows.append(Doc(f'{p}#{i}',text,{"path":str(p),"row":i}))
    return rows
def load_pdf(path: str|Path) -> list[Doc]:
    try:
        from pypdf import PdfReader
    except ImportError as e: raise RuntimeError('PDF ingestion requires optional dependency pypdf') from e
    p=Path(path); return [Doc(f'{p}#{i}',page.extract_text() or '',{"path":str(p),"page":i}) for i,page in enumerate(PdfReader(str(p)).pages)]
def load_image(path: str|Path) -> list[Doc]:
    try:
        from PIL import Image
    except ImportError as e: raise RuntimeError('image ingestion requires optional dependency Pillow') from e
    p=Path(path)
    with Image.open(p) as im: text=f'Image {p.name}, size={im.size}, mode={im.mode}'
    return [Doc(str(p),text,{"path":str(p),"media_type":"image"})]
def ingest(path: str|Path) -> list[Doc]:
    p=Path(path); ext=p.suffix.lower()
    if ext=='.csv': return load_csv(p)
    if ext=='.pdf': return load_pdf(p)
    if ext in {'.png','.jpg','.jpeg','.webp'}: return load_image(p)
    return load_text(p)
