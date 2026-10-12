"""Live end to end: real PDFs, real index (OpenAI embeddings), real answers and the real Ragas judge.

    RUN_LIVE_TESTS=1 uv run pytest -m live
LLM_MODEL comes from LIVE_LLM_MODEL (default gpt-5-mini), since .env may leave LLM_MODEL blank.
"""

import os
from dataclasses import replace
from datetime import UTC, datetime

import pymupdf
import pytest
from factories import NORTHWIND_LEADERSHIP, NORTHWIND_QUALITY, make_eval_samples

from rag_chatbot.config import load_settings
from rag_chatbot.container import build_container
from rag_chatbot.domain.datasets import (
    GeneratorInfo,
    TestSample,
    Testset,
    TestsetManifest,
    new_testset_id,
    samples_sha256,
)
from rag_chatbot.domain.evals import DEFAULT_METRICS, RunStatus
from rag_chatbot.ingestion.fingerprint import corpus_fingerprint
from rag_chatbot.services.evaluation_service import StartEvaluationRequest

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(os.getenv("RUN_LIVE_TESTS") != "1", reason="set RUN_LIVE_TESTS=1 to call the real API"),
]


def write_pdf(path, text):
    doc = pymupdf.open()
    doc.new_page().insert_textbox(pymupdf.Rect(36, 36, 576, 806), text, fontsize=9)
    doc.save(str(path))


def handwritten_testset(data_dir) -> Testset:
    samples = [
        TestSample(user_input=s.user_input, reference=s.reference, reference_contexts=s.reference_contexts, synthesizer_name="handwritten")
        for s in make_eval_samples()[:1]
    ] + [
        TestSample(
            user_input="Which four indicators does Northwind use to measure delivery quality?",
            reference="Deployment frequency, lead time for changes, change failure rate and time to restore.",
            reference_contexts=[NORTHWIND_QUALITY],
            synthesizer_name="handwritten",
        )
    ]
    now = datetime.now(UTC)
    digest = samples_sha256(samples)
    manifest = TestsetManifest(
        id=new_testset_id(now, "live-e2e", 1, digest),
        name="live-e2e",
        version=1,
        parent_id=None,
        kg_source_id=None,
        created_at=now,
        size=len(samples),
        generator=GeneratorInfo("handwritten", "n/a", None, "n/a"),
        corpus=corpus_fingerprint(data_dir),
        samples_sha256=digest,
        usage=None,
        has_knowledge_graph=False,
    )
    return Testset(manifest=manifest, samples=samples)


def test_one_pipeline_against_a_two_question_test_set(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    write_pdf(data / "northwind-quality.pdf", NORTHWIND_QUALITY)
    write_pdf(data / "northwind-leadership.pdf", NORTHWIND_LEADERSHIP)
    settings = replace(
        load_settings(),
        llm_provider="openai",
        llm_model=os.getenv("LIVE_LLM_MODEL", "gpt-5-mini"),
        eval_provider="ragas",
        embedding_provider="openai",
        eval_row_delay_seconds=0,
        data_dir=data,
        index_dir=tmp_path / "index",
        datasets_dir=tmp_path / "datasets",
        evals_dir=tmp_path / "evals",
    )
    container = build_container(settings)
    try:
        testset = handwritten_testset(data)
        container.datasets.save(testset)
        [summary] = container.evaluation.run_evaluation(
            StartEvaluationRequest(testset_id=testset.manifest.id, pipelines=("parent_child",))  # retrieves full passages
        )
        run = container.evaluation.get_run(summary.run_id)
        m = run.manifest
        print(f"\naggregate: {m.aggregate}\nanswer usage: {m.answer_usage}\njudge usage: {m.judge_usage}")
        assert m.status is RunStatus.COMPLETED, m.error
        assert len(run.samples) == 2 and all(s.response for s in run.samples)
        assert set(m.aggregate) == set(DEFAULT_METRICS)
        # Ragas occasionally times out on a single judge call; that score is then None, not an error.
        missing = [k for k in DEFAULT_METRICS if m.aggregate[k] is None]
        assert len(missing) <= 1, f"metrics without a score: {missing}"
        assert m.aggregate["faithfulness"] >= 0.5 and m.aggregate["context_recall"] >= 0.5  # grounded answers
        assert m.answer_usage is not None and m.answer_usage.total_tokens > 0
        assert m.judge_usage is not None and m.judge_usage.total_tokens > 0
        assert m.answer_model == settings.llm_model and m.judge_model == settings.eval_judge_model
    finally:
        container.close()
