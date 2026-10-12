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
- `ports/` defines the protocols the app depends on:
  - `llm.py`: `LLMPort`, for text generation
  - `datasets.py`: `DatasetRepository`, for saving and loading test sets
  - `testset_generator.py`: `TestsetGeneratorPort`, for generating synthetic test samples
- `adapters/` contains the implementations:
  - `llm/`: the OpenAI and fake LLM adapters. The UI's reasoning-level list comes from
    `supported_reasoning_efforts()`.
  - `datasets/`: the file-based and in-memory test set repositories
  - `sdg/`: the Ragas and fake test set generators. Ragas is imported only when
    generation actually runs.

### Background jobs

Test set generation and evaluations take minutes, so the UI runs them as background jobs
through `services/jobs.py` → `JobRunner`, owned by the container:

- **One job at a time.** Starting a second job while one is queued or running raises
  `JobAlreadyRunning`. This keeps API rate limits under control, and avoids two jobs
  writing to the embedded Qdrant index at once.
- **In-process.** Jobs run on a worker thread in the Streamlit server process. Their
  status survives page reloads and is shared across tabs, because the container is
  cached. It **does not survive a server restart**. Services that need durable status,
  such as evaluation runs, record it in their own storage.
- **Status as snapshots.** `status()`, `active()` and `recent()` return immutable
  `JobStatus` snapshots (state, progress, message, error, result id). The UI polls
  them, and never sees a half-updated job.

Every port has a shared **contract test suite** (`tests/test_*_contract.py`). Each
adapter of that port must pass it, which is what makes the adapters interchangeable.

### Adding a provider

1. Create `adapters/llm/<provider>_adapter.py` implementing `LLMPort`. Map provider
   exceptions to the `domain.errors` types.
2. Add a branch for it in `container.build_llm`.
3. Add it to the `adapter` fixture params in `tests/test_llm_port_contract.py`.

### Adding a storage adapter

The same pattern applies to `DatasetRepository`. For example, to store test sets in
LangSmith Datasets:
1. Implement the protocol in `adapters/datasets/`.
2. Add it to the `repo` fixture params in `tests/test_dataset_repository_contract.py`
   and make the suite pass.
3. Choose it in `build_container`.

## Generating test sets

Generate a synthetic test set from the PDFs in `data/` with the CLI. The Test Sets page
in the app will offer the same thing.

```bash
uv run python scripts/generate_testset.py --name core-corpus --size 10
uv run python scripts/generate_testset.py --name core-corpus --size 20 --reuse-kg-from <testset_id>
```

1. **Estimate and confirm.** The CLI prints the model, the corpus size and an estimate
   of input and output tokens, then asks `Proceed? [y/N]`. `--yes` skips the prompt. A
   cost is shown only when both `SDG_PRICE_INPUT_PER_MTOK` and
   `SDG_PRICE_OUTPUT_PER_MTOK` are set. Each must be a finite, non-negative number.
   Prices change, so none are built in.
2. **The estimate gets better with use.** The first estimate uses rough constants
   (documented in `services/sdg_service.py`). After that, it's calibrated against the
   actual usage of your most recent test set made from the same corpus with the same
   model.
3. **Reusing a knowledge graph** (`--reuse-kg-from`) skips the most expensive step.
   Use it to make a larger test set, or another mix of questions, from the same corpus.
4. **Requested vs. generated.** Ragas can produce fewer distinct questions than asked
   for on a small corpus, and the CLI says so when that happens. A test set's `size`
   is what was generated, and `requested_size` is what was asked for.

The CLI never opens the vector index, so it can run while the app is running.

## Test sets (storage)

Synthetic test sets are saved under `DATASETS_DIR` (default `.datasets/`), one folder
per test set:

```
.datasets/<id>/manifest.json         # name, version, models, corpus fingerprint, usage, …
.datasets/<id>/samples.jsonl         # one question / reference / reference contexts per line
.datasets/<id>/knowledge_graph.json  # optional; lets a new test set reuse the graph
```

- **Immutable.** A saved test set never changes. A change produces a new version
  (`version` counts up per `name`, and `parent_id` links back), so runs scored against
  a test set always stay comparable.
- **IDs:** `ts_<yyyymmdd>_<hash8>`, where the hash covers the name, the version and
  the samples.
- **Versions:** a new test set with an existing name becomes the next version, with
  `parent_id` pointing to the newest version saved at that moment. Versions are never
  overwritten. Each name and version can be saved only once, even when the CLI and the
  app save at the same time. A save first claims `.datasets/.versions/<name>@<version>`
  with an exclusive create, and the loser retries with the next number. A save
  interrupted by a crash leaves its claim behind, so that version number is skipped,
  which leaves a gap in the numbering but doesn't block the name.
- **Names:** 1–64 letters, digits, `-` or `_`.
- **Corpus fingerprint:** the SHA-256 of every PDF in `data/` when the test set was
  made. If any PDF has since been added, removed or changed, the test set is **stale**.
  Its questions may refer to content that no longer exists, so its scores aren't
  comparable with runs on the current corpus.
- **Atomic saves:** a save is written to a temporary folder and renamed into place, so
  an interrupted save never leaves a partial test set behind.
- `.datasets/` is git-ignored. Test sets contain passages from the PDFs, which are
  never committed.

## Choosing models for SDG and judging

Test set generation (`SDG_LLM_MODEL`) and, later, evaluation judging use **non-reasoning
models by default**: `gpt-4.1` for generation, with `text-embedding-3-small` embeddings.
The reasons:

- **Compatibility.** Ragas sets the temperature to about 0 (`1e-8`) on every call, so
  that generation and scoring are as repeatable as possible. Many reasoning models only
  accept a temperature of 1. LangChain removes the setting for `gpt-5*` models, but not
  for o-series models or other names, so those requests can be rejected. The generator's
  `preflight()` check makes one tiny call the same way Ragas does, and reports
  `ModelIncompatible` before any real spending.
- **Little benefit.** The work is extraction (summaries, headlines, entities) and writing
  questions from given text, or grading against a rubric. These gain little from reasoning.
- **Cost.** Ragas makes many LLM calls: one generation run is dozens, and one evaluation
  is hundreds. Reasoning tokens multiply the cost of every one.

If you use a reasoning model anyway, set `SDG_REASONING_EFFORT=low` and run the
preflight check first.

**Sample counts:** Ragas doesn't return an exact number of samples. It splits the
requested size across question types and rounds each share up, and it can fall short on
a small corpus. The generator trims any extra, keeping the mix of question types, and
tops up a shortfall up to twice. It never returns more than requested. It can return
fewer if the corpus can't support more distinct questions.

**Token usage** recorded for a test set covers LLM calls only, not embeddings, which
cost very little.

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
