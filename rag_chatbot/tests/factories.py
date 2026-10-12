"""Shared builders for test data."""

from datetime import UTC, datetime

from rag_chatbot.domain.datasets import (
    CorpusFingerprint,
    SourceDocument,
    GeneratorInfo,
    TestSample,
    Testset,
    TestsetManifest,
    new_testset_id,
    samples_sha256,
)
from rag_chatbot.domain.models import TokenUsage

T0 = datetime(2026, 10, 12, 9, 0, tzinfo=UTC)


def make_testset(
    name: str = "core-corpus",
    version: int = 1,
    *,
    created_at: datetime = T0,
    n: int = 3,
    parent_id: str | None = None,
    with_kg: bool = False,
    tag: str = "",
    requested_size: int | None = None,
) -> Testset:
    samples = [
        TestSample(
            user_input=f"Question {i}{tag}?",
            reference=f"Answer {i}{tag}.",
            reference_contexts=[f"context {i}a", f"context {i}b"],
            synthesizer_name="single_hop_specifc_query_synthesizer",
        )
        for i in range(n)
    ]
    digest = samples_sha256(samples)
    manifest = TestsetManifest(
        id=new_testset_id(created_at, name, version, digest),
        name=name,
        version=version,
        parent_id=parent_id,
        kg_source_id=None,
        created_at=created_at,
        size=n,
        generator=GeneratorInfo("gpt-4.1", "text-embedding-3-small", None, "0.2.10"),
        corpus=CorpusFingerprint({"a.pdf": "1" * 64, "b.pdf": "2" * 64}),
        samples_sha256=digest,
        usage=TokenUsage(input_tokens=1000, output_tokens=200, total_tokens=1200),
        has_knowledge_graph=with_kg,
        requested_size=requested_size,
    )
    return Testset(manifest=manifest, samples=samples)


# ---------- test set generation ----------

# Long enough for Ragas (it rejects documents of 100 tokens or fewer), with entities shared
# across both documents so multi-hop questions are possible. Used offline and by live tests.
NORTHWIND_QUALITY = """\
Northwind Labs runs every release through a three-stage quality process. First, engineers pair on
code review within a day of opening a pull request, because Northwind's data shows that defects found
in review cost about a tenth as much to fix as defects found in production. Second, the platform team
maintains a testing pyramid: thousands of fast unit tests, a few hundred integration tests against
real databases, and a small set of end-to-end tests that exercise the checkout flow. Third, a release
captain, a rotating role held by a senior engineer for one week, signs off only when error budgets
are healthy. Priya Raman, Northwind's head of quality, introduced the release captain role in 2023
after two outages traced back to rushed Friday deployments. Since then, change failure rate has
dropped from 18 percent to 6 percent. Priya argues that quality is a habit built by many small,
repeated practices rather than a phase at the end of a project, and she measures it with four
indicators: deployment frequency, lead time for changes, change failure rate and time to restore.
"""

NORTHWIND_LEADERSHIP = """\
Northwind Labs treats engineering leadership as the work of setting direction, removing blockers and
growing people. Engineering managers hold weekly one-on-ones and spend at least a third of their time
on coaching. Priya Raman, who leads quality, partners with the engineering managers to make the
release captain role a growth opportunity: engineers who serve as release captain learn incident
response, risk assessment and stakeholder communication. Leadership reviews the four delivery
indicators, deployment frequency, lead time for changes, change failure rate and time to restore, in a
monthly operations review, but managers are told never to use the indicators to rank individuals.
Instead, the indicators guide investment: when lead time rose in early 2024, leadership funded a
dedicated platform team to speed up the integration test suite. Northwind's leadership principles say
that a team's long-term velocity depends on psychological safety, clear ownership and a sustainable
pace, and that heroics during outages are a symptom of missing process rather than something to reward.
"""

SDG_DOCS = [
    SourceDocument(NORTHWIND_QUALITY, {"source": "northwind-quality.pdf"}),
    SourceDocument(NORTHWIND_LEADERSHIP, {"source": "northwind-leadership.pdf"}),
]

# (user_input, reference, reference_contexts, synthesizer_name)
RAGAS_SAMPLES = [
    ("What is quality?", "A habit.", ["Quality is a habit."], "single_hop_specifc_query_synthesizer"),
    ("Who removes blockers?", "Leaders.", ["Leaders remove blockers.", "They grow people."], "multi_hop_abstract_query_synthesizer"),
]


def ragas_testset(samples):
    """Build a real Ragas Testset from (user_input, reference, reference_contexts, synthesizer_name) tuples."""
    from ragas.dataset_schema import SingleTurnSample
    from ragas.testset.synthesizers.testset_schema import Testset, TestsetSample

    return Testset(
        samples=[
            TestsetSample(
                eval_sample=SingleTurnSample(user_input=q, reference=r, reference_contexts=ctx),
                synthesizer_name=name,
            )
            for q, r, ctx, name in samples
        ]
    )


# ---------- evaluation runs ----------


def make_run_manifest(
    run_id: str = "run_20261012-090000_baseline_ab12",
    *,
    batch_id: str = "batch_20261012-090000_cd34",
    testset_id: str = "ts_20261012_aaaaaaaa",
    pipeline: str = "baseline",
    started_at: datetime = T0,
):
    from rag_chatbot.domain.evals import EvalRunManifest, RunStatus

    return EvalRunManifest(
        run_id=run_id,
        batch_id=batch_id,
        testset_id=testset_id,
        pipeline=pipeline,
        pipeline_settings={"chunk_size": 50, "chunk_overlap": 0, "k": 3},
        answer_model="gpt-5-mini",
        reasoning_effort="low",
        judge_model="gpt-4.1-mini",
        judge_reasoning_effort=None,
        embedding_model="text-embedding-3-small",
        metrics=["faithfulness", "answer_relevancy"],
        ragas_version="0.2.10",
        index_manifest_digest="d" * 64,
        status=RunStatus.RUNNING,
        progress=0.0,
        message="Starting",
        started_at=started_at,
        finished_at=None,
        error=None,
    )


def make_sample_results(n: int = 2):
    from rag_chatbot.domain.evals import SampleResult

    return [
        SampleResult(
            user_input=f"Question {i}? — café 日本語",
            reference=f"Answer {i}.",
            response=f"Response {i}." if i % 2 == 0 else None,
            retrieved_contexts=[f"context {i}a", f"context {i}b"],
            scores={"faithfulness": 0.75 if i % 2 == 0 else None, "answer_relevancy": 0.5},
            latency_ms=123.4,
            usage=TokenUsage(input_tokens=100, output_tokens=20, total_tokens=120),
            error=None if i % 2 == 0 else "LLMRateLimitError: throttled",
        )
        for i in range(n)
    ]
