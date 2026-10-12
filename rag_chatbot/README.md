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

## Pages

`app.py` is the entry point. It sets up `st.navigation` with three pages:

| Page | File | What it does |
|---|---|---|
| **Chat** | `src/rag_chatbot/ui/chat_page.py` | Ask one pipeline, or compare all three side by side. The first visit builds the vector index if needed. |
| **Test Sets** | `src/rag_chatbot/ui/test_sets_page.py` | Generate synthetic test sets with an estimate and confirmation step, watch progress, and browse saved test sets. Never builds the vector index. |
| **Evaluations** | `src/rag_chatbot/ui/evaluations_page.py` | Estimate and run evaluations of one or more pipelines against a test set, watch progress, then filter, compare (with Δ against a reference run) and inspect runs. |

All pages share one cached container (`ui/state.py`), so a job started on one page or
tab is visible in all of them. Only one job runs at a time.

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
  - `eval_runs.py`: `EvalRunRepository`, for saving evaluation runs and reading them back
  - `evaluator.py`: `EvaluatorPort`, for scoring answers against reference answers
- `adapters/` contains the implementations:
  - `llm/`: the OpenAI and fake LLM adapters. The UI's reasoning-level list comes from
    `supported_reasoning_efforts()`.
  - `datasets/`: the file-based and in-memory test set repositories
  - `eval_runs/`: the file-based and in-memory evaluation run repositories
  - `evaluation/`: the Ragas and fake evaluators. Shared Ragas plumbing (judge LLM,
    embeddings, preflight) is in `adapters/ragas_support.py`.
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

**In the app:** open **Test Sets**, then:
1. Choose a name and how many questions you want. Optionally pick a saved knowledge
   graph to reuse.
2. Click **Estimate**.
3. Check the tokens and cost.
4. Click **Confirm and generate**.

Changing any input, or the PDFs in `data/`, after estimating means you need a new estimate. Generation runs in
the background with a progress bar, and the form is locked until it finishes. The new
test set then appears in the list.

**From the CLI**, the same steps:

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

## Running evaluations

**In the app:** open **Evaluations**, then:
1. Choose a test set, the pipelines and a reasoning effort.
2. Click **Estimate**. It shows how many answers will be generated, plus answering and
   judging tokens, each with a cost when that model's prices are set
   (`LLM_PRICE_*` and `EVAL_JUDGE_PRICE_*`).
3. Click **Confirm and run**. Changing any input after estimating means you need a new
   estimate.
4. When the run finishes, its batch is selected for comparison. The comparison table
   shows metrics as rows and one column per run, with a Δ column per run against the
   reference run (green is better, red is worse, except for
   `noise_sensitivity_relevant`, where lower is better).
5. Pick any runs to compare with **Runs to compare**, or a whole batch with **Select
   batch**, and filter the runs table by test set, pipeline or status.

Judging dominates the cost. For 10 questions across 3 pipelines, the estimate is about
621k judging tokens, versus about 27k for answering.

An evaluation answers every question in a saved test set with one or more pipelines
(using `LLM_MODEL`), scores the answers with the judge (`EVAL_JUDGE_MODEL`), and saves
one run per pipeline. Runs started together share a batch.

```bash
uv run python scripts/run_eval.py --testset <testset_id>                       # all available pipelines
uv run python scripts/run_eval.py --testset <testset_id> --pipelines baseline,parent_child --effort low
uv run python scripts/run_eval.py --testset <testset_id> --reference parent_child
uv run python scripts/run_eval.py --testset <testset_id> --yes                 # skip the confirmation
```

The CLI prints the same estimate as the app (with any warnings) and asks
`Proceed? [y/N]` before running anything.

**Stop the app first.** Evaluating needs the vector index, and the embedded index allows
one process at a time. If the app is running, the CLI exits with an "already accessed"
error and touches nothing.

After confirming, the CLI shows progress and ends with a comparison table: metrics as rows,
one column per pipeline, and a Δ column against `--reference` (default: the first
pipeline). It exits with 0 when every run completed, 2 when some failed, and 1 on errors.

How a batch runs:
1. **The judge is checked first** (`preflight`), before any answering, so an
   incompatible judge model costs nothing. If it fails, every run in the batch fails
   with that message.
2. **Each pipeline in turn** answers every question, pausing `EVAL_ROW_DELAY_SECONDS`
   (default 2) between them. If a question fails to answer, the error is kept and all
   of its scores are `None`, and the run continues.
3. **The successful answers are scored.** If the judge fails for one pipeline, that run
   fails, keeping its answers, and the next pipeline still runs. A run where every
   answer failed also fails ("no answers to score").
4. **Aggregates** are the mean of each metric over the answers that have a score.

**What each run records:**
- the test set
- the pipeline and a snapshot of its settings
- the answering model and reasoning effort
- the judge model and effort, the embeddings and Ragas version
- the metrics
- a digest identifying the vector index's definition (PDFs, embedding model, pipeline
  settings)
- answering and judging token usage
- per question: the answer, retrieved contexts, scores, latency and any error

While an evaluation (or any job) is running, the Chat page's **Rebuild index** is
disabled, because rebuilding would change the index under the running evaluation. An
evaluation started during a rebuild waits for it to finish. A failed run still records
the answering tokens it spent.

If the server stops mid-evaluation, its runs are marked failed ("Interrupted (server
restarted)") the next time the app or CLI starts.

**Reading the results:** in a live check against two Northwind test documents, Baseline
(50-character chunks) retrieved fragments too short to answer from, so it scored 0.0 on
`context_recall`. Parent-Child retrieved whole passages and scored 1.0. That gap is
exactly what comparing runs is meant to surface. Ragas occasionally times out on a
single judge call, and that score is then `None`.

## Evaluation runs (storage)

Evaluation runs are saved under `EVALS_DIR` (default `.evals/`), one folder per run:

```
.evals/<run_id>/manifest.json   # test set, pipeline + settings, models, status, progress, aggregate scores, usage
.evals/<run_id>/samples.jsonl   # per question: answer, retrieved contexts, scores, latency, usage, error
```

- **Lifecycle:** a run is created as `running`. Its progress is updated as it goes, then
  it ends as `completed` or `failed`. A finished run is **final**: any further write is
  rejected (`RunFinalized`), so a result can't be changed after the fact. A failed run
  keeps the samples answered before the failure.
- **Crash-safe:** every file is written to a temporary file and moved into place, so a
  reader sees either the previous version or the next, never half a file. The manifest
  is written last, so its status is what counts. A run's samples are only shown once
  it's finished. A run left `running` by a crash can still be marked failed. Failing it
  without samples discards anything an interrupted completion left behind.
- **Startup cleanup:** a new repository removes leftover temp files, and run folders
  whose creation was interrupted before the manifest was written. It does this while
  holding the write lock, so it never touches another writer's in-progress files.
- **One writer at a time:** writes take a lock shared by every repository instance in
  the process, plus a file lock on `.evals/.lock` across processes. So two writers can
  never both finish the same run.
- **Stays inside `.evals/`:** symlinked run folders are ignored, run files are opened
  without following symlinks, and cleanup never reaches through a link. A run whose
  files can't be read safely (a symlink, broken JSON, an unknown status) raises
  `RunCorrupt` when opened, and is left out of listings. One damaged run never hides
  the others.
- **Scores** are numbers or `None`. `NaN` and infinity are rejected: JSON can't hold
  them, and a metric that couldn't be scored is stored as `None`.
- `.evals/` is git-ignored. Runs contain questions, answers and passages from the PDFs.

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

If you use a reasoning model anyway, set `SDG_REASONING_EFFORT=low` (or
`EVAL_JUDGE_REASONING_EFFORT=low`) and run the preflight check first.

**The judge** (`EVAL_JUDGE_MODEL`, default `gpt-4.1-mini`) scores answers on six Ragas
metrics: `context_recall`, `faithfulness`, `factual_correctness`, `answer_relevancy`,
`context_entity_recall` and `noise_sensitivity_relevant` (lower is better for the last
one).
- **It's call-heavy.** In a live check, scoring 2 answers on all six metrics took about
  27,600 judge tokens, roughly 14k per answer. A 10-question test set run against all
  three pipelines is therefore about 400k judge tokens. That's why the default judge is
  an inexpensive model.
- **It shouldn't be the answering model.** A model grading its own answers tends to be
  generous. The app warns when `EVAL_JUDGE_MODEL` equals `LLM_MODEL`.
- **Read metrics together.** In the same live check, a made-up answer scored 0.0 on
  `faithfulness` and `factual_correctness`, but *higher* than the correct answer on
  `answer_relevancy`. That metric only checks whether the answer addresses the
  question, not whether it's true.
- A metric that can't be computed for an answer is recorded as `None`. Ragas reports
  it as `NaN`.

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
uv run pytest                                   # offline: fakes, in-memory Qdrant, Streamlit AppTest for the UI
RUN_LIVE_TESTS=1 uv run pytest -m live          # also runs the contract suites and end-to-end tests against OpenAI
LLM_PROVIDER=fake SDG_PROVIDER=fake EVAL_PROVIDER=fake uv run streamlit run app.py   # click through the UI without model cost
```

The UI tests (`tests/test_ui_*.py`) replace the app's cached container with one built
from fakes, so they need no network.

The fake-provider `streamlit run` makes no model calls, but it still needs
`OPENAI_API_KEY` to start. The app creates the OpenAI embeddings client for the vector
index at startup on every page, and #16 tracks making that lazy. The Chat page also uses
that key to embed documents the first time it builds the index. The Test Sets page never
builds the index. The evaluator is created only when an evaluation actually runs, so
judge settings are never needed just to open a page.

## Implementation notes

- All three pipelines share the same prompt and the same LangGraph shape: a retrieve node
  followed by a generate node. The generate node calls the model through `LLMPort`.
- Each PDF's pages are merged into one Document before splitting, so chunks aren't cut
  at page boundaries. Splitters use `add_start_index=True`, which lets answers cite the
  source page. This adds metadata only and doesn't change where chunks break.
- Indexes are saved to disk: local Qdrant for all three pipelines, plus a file-backed
  docstore for the Parent-Child parents.
- `CohereRerank` is built once when the pipeline is created, not on every query.
