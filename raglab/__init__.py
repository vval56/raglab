from .core import Doc, Source, RAGResult, Trace, LLM, Embedder, VectorStore, Strategy
from .embeddings import HashEmbedder, TfidfEmbedder
from .store import CosineVectorStore
from .chunking import chunk_text
from .ingestion import ingest, load_text, load_csv, load_pdf, load_image
from .lmstudio import LMStudioLLM
from .strategies import *
__version__='0.1.0'
