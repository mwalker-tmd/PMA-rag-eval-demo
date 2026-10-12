# RAG Pipeline Chat

A Streamlit chatbot that answers questions about the PDFs in `../data` using three RAG
pipelines. Ask through one pipeline, or compare all three side by side, and choose the
model's reasoning level.

| Pipeline | Retrieval |
|---|---|
| Baseline | 50-char chunks, top-3 similarity |
| Rerank | 500/30 chunks, top-20 similarity → Cohere `rerank-v3.5` (top 3) |
| Parent-Child | search 200/40 child chunks (top 8), return their 1000/100 parents |

## Setup

Put the PDFs you want to search in `../data`. The folder is in the repo, but its
contents are git-ignored, so you need to add them yourself.

```bash
uv sync
cp .env.sample .env          # then fill in OPENAI_API_KEY, COHERE_API_KEY, LLM_MODEL
uv run python scripts/ingest.py
uv run streamlit run app.py
```

The app builds the index on first launch if it doesn't exist, so the ingest step is
optional. The index is stored in `.index/` and rebuilt automatically when a PDF, the
embedding model or a pipeline setting changes. Use `ingest.py --force` or the
**Rebuild index** button to rebuild on demand.

> Embedded Qdrant allows one process at a time. Stop the app before running
> `ingest.py`.

## Architecture (Ports and Adapters)

```
app.py (Streamlit, driving adapter)
  └─ ChatService ── pipelines (LangGraph: retrieve → generate)
                       └─ generate ── LLMPort ◄── OpenAILLMAdapter (Responses API)
                                               ◄── FakeLLMAdapter (tests / offline)
container.py — composition root; the only place concrete adapters are chosen
```

- `domain/` holds plain types (`LLMRequest`, `LLMResponse`, `ReasoningEffort`, …) and
  errors, with no SDK imports.
- `ports/llm.py` defines the `LLMPort` protocol.
- `adapters/llm/` contains the provider implementations. The UI's reasoning-level list
  comes from `supported_reasoning_efforts()`.

### Adding a provider

1. Create `adapters/llm/<provider>_adapter.py` implementing `LLMPort`. Map provider
   exceptions to the `domain.errors` types.
2. Add a branch for it in `container.build_llm`.
3. Add it to the `adapter` fixture params in `tests/test_llm_port_contract.py`.

## Reasoning levels

The OpenAI adapter picks levels by model name (`gpt-4*` → none, `gpt-5*` →
minimal/low/medium/high, `o*` → low/medium/high, anything else → low/medium/high).
Set `LLM_REASONING_EFFORTS` in `.env` to override this, for example `low,medium,high`,
or `unsupported` to hide the selector.

## Tests

```bash
uv run pytest                                   # offline: fake LLM, fake embeddings, in-memory Qdrant
RUN_LIVE_TESTS=1 uv run pytest -m live          # also runs the port contract against the real OpenAI model
LLM_PROVIDER=fake uv run streamlit run app.py   # click through the UI without generation cost
```

The fake-LLM run still needs `OPENAI_API_KEY` if the index has to be built, because
embeddings always come from OpenAI.

## Implementation notes

- All three pipelines share the same prompt and the same LangGraph shape: a retrieve node
  followed by a generate node. The generate node calls the model through `LLMPort`.
- Each PDF's pages are merged into one Document before splitting, so chunks aren't cut
  at page boundaries. Splitters use `add_start_index=True`, which lets answers cite the
  source page. This adds metadata only and doesn't change where chunks break.
- Indexes are saved to disk: local Qdrant for all three pipelines, plus a file-backed
  docstore for the Parent-Child parents.
- `CohereRerank` is built once when the pipeline is created, not on every query.
