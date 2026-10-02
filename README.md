# RAGLab

### Ten retrieval strategies for local LLMs

RAGLab is a lightweight Python laboratory for comparing ten Retrieval-Augmented
Generation patterns against a local model running in [LM Studio](https://lmstudio.ai/).
It exposes the same corpus, vector store, sources, traces and evaluation flow to every
strategy, so differences are visible instead of hidden behind framework magic.

> **Local-first:** no cloud API is required. The default tests run offline, while
> generation can be delegated to any OpenAI-compatible local endpoint.

## What is included

| # | Strategy | Main idea |
|---:|---|---|
| 01 | **Simple RAG** | One query, one retrieval pass |
| 02 | **Memory RAG** | Rewrite follow-up questions using chat history |
| 03 | **Multi-query RAG** | Fan out into several search queries |
| 04 | **HyDE RAG** | Retrieve using a hypothetical answer |
| 05 | **Adaptive RAG** | Route retrieval depth by query shape |
| 06 | **Corrective RAG** | Filter weak evidence and retry with a rewritten query |
| 07 | **Self-RAG** | Draft an answer and critique its factual support |
| 08 | **Agentic RAG** | Build a bounded multi-step retrieval plan |
| 09 | **Multimodal RAG** | Combine text with image/document metadata |
| 10 | **Graph RAG** | Expand retrieval through local document relationships |

Every result contains:

- the generated answer;
- source IDs, metadata and similarity scores;
- an execution trace explaining retrieval, rewrites, retries and plans.

## Quick start

```bash
git clone <your-repository-url>
cd raglab

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]"

python -m pytest -q
```

Expected result:

```text
18 passed
```

Optional integrations:

```bash
python -m pip install -e ".[mcp]"       # LM Studio MCP plugin
python -m pip install -e ".[pdf,image]" # PDF and image ingestion
```

## See the strategies in action

The demo uses a small connected corpus designed to make the differences obvious:
policy exceptions, conversation history, delivery facts and a
`Northwind → Delta → AEP → MW` relationship chain.

```bash
LMSTUDIO_MODEL="google/gemma-4-e2b" \
LMSTUDIO_MAX_TOKENS=256 \
python scripts/demo_rag.py
```

The output prints the model answer, retrieved sources and trace for Simple, Memory,
Multi-query, Graph and Agentic RAG.

For a quick connectivity check:

```bash
curl http://localhost:1234/v1/models
LMSTUDIO_MODEL="google/gemma-4-e2b" python scripts/smoke_lmstudio.py
```

## Connect LM Studio as an MCP plugin

LM Studio can act as an MCP host. RAGLab is registered as a local MCP plugin; LM Studio
starts the process automatically when it needs a tool.

1. Start LM Studio's local server and load your model.
2. Install RAGLab's MCP extra:

   ```bash
   .venv/bin/python -m pip install -e ".[mcp]"
   ```

3. Open **LM Studio → Program → Install → Edit `mcp.json`**.
4. Add `raglab` inside the existing `mcpServers` object:

   ```json
   {
     "mcpServers": {
       "raglab": {
         "command": "/ABSOLUTE/PATH/raglab/.venv/bin/raglab-mcp",
         "env": {
           "RAGLAB_USE_LMSTUDIO": "1",
           "LMSTUDIO_BASE_URL": "http://localhost:1234/v1",
           "LMSTUDIO_MODEL": "google/gemma-4-e2b",
           "LMSTUDIO_TIMEOUT": "120"
         }
       }
     }
   }
   ```

   If other servers already exist, keep them in the same object and add a comma
   between entries. The full ready-to-edit template is in `mcp.json.example`.

RAGLab currently pins `mcp<2` because its FastMCP integration uses the stable 1.x API.
If an environment previously installed MCP 2.x:

```bash
.venv/bin/python -m pip install --force-reinstall "mcp<2"
```

Available MCP tools:

```text
list_strategies
index_documents
query_rag
compare_strategies
```

## Use the library directly

```python
from raglab import Doc, HashEmbedder, CosineVectorStore, SimpleRAG
from raglab.lmstudio import LMStudioLLM

llm = LMStudioLLM(
    model="google/gemma-4-e2b",
    base_url="http://localhost:1234/v1",
)
rag = SimpleRAG(llm, HashEmbedder(), CosineVectorStore(), top_k=4)
rag.index([
    Doc("refund", "Возврат товара возможен в течение 30 дней при наличии чека."),
    Doc("defect", "Для производственного дефекта срок обращения составляет 90 дней."),
])

result = rag.run("Какой срок возврата товара?")
print(result.answer)
print(result.to_dict())
```

## Data ingestion

The repository includes sample energy-consumption CSV files in `archive/`.

```python
from raglab.ingestion import ingest

documents = ingest("archive/AEP_hourly.csv")
```

Supported inputs:

| Format | Functionality |
|---|---|
| TXT / Markdown | Full-text documents |
| CSV | One document per row with metadata |
| PDF | One document per page (`pypdf`) |
| PNG / JPEG / WebP | Image metadata (`Pillow`) |

## Project layout

```text
raglab/
├── core.py          # shared types and protocols
├── embeddings.py    # deterministic hash and TF-IDF embeddings
├── store.py         # local cosine vector store
├── ingestion.py     # TXT, CSV, PDF and image ingestion
├── lmstudio.py      # OpenAI-compatible LM Studio adapter
├── strategies.py    # all ten RAG strategies
└── mcp_server.py    # LM Studio MCP tools
scripts/
├── demo_rag.py      # rich local demonstration
└── smoke_lmstudio.py
tests/               # offline regression tests
archive/             # energy-consumption sample data
```

## Design goals

- **Comparable:** the strategies share one result and tracing contract.
- **Reproducible:** core tests use deterministic embeddings and fake LLMs.
- **Inspectable:** every result exposes sources and intermediate steps.
- **Local:** LM Studio is the only runtime model dependency.
- **Bounded:** Agentic and Graph RAG have explicit limits for safe experiments.

## License

This project is intended as an educational and experimental RAG laboratory.
