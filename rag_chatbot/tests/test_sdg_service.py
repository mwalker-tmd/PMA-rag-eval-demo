"""SdgService with the fake generator and in-memory repository, plus the CLI and job integration."""

import importlib.util
import threading
from dataclasses import replace
from pathlib import Path

import pymupdf
import pytest
from factories import make_testset
from qdrant_client import QdrantClient

from rag_chatbot.adapters.datasets.file_repository import FileDatasetRepository
from rag_chatbot.adapters.datasets.memory_repository import InMemoryDatasetRepository
from rag_chatbot.adapters.sdg.fake_generator import FakeTestsetGenerator
from rag_chatbot.domain.datasets import GenerationError, GenerationResult, TestsetNotFound
from rag_chatbot.domain.errors import ModelIncompatible
from rag_chatbot.ingestion.fingerprint import corpus_fingerprint
from rag_chatbot.services.jobs import JobAlreadyRunning, JobRunner, JobState
from rag_chatbot.services.sdg_service import (
    GenerateTestsetRequest,
    InvalidRequest,
    SdgService,
)

TIMEOUT = 10
SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load_script(name):
    """Import scripts/<name>.py (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location(f"script_{name}", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def repo():
    return InMemoryDatasetRepository()


@pytest.fixture
def generator():
    return FakeTestsetGenerator()


def make_service(repo, generator, data_dir, **kwargs):
    return SdgService(datasets=repo, generator=generator, jobs=JobRunner(), data_dir=data_dir, **kwargs)


@pytest.fixture
def service(repo, generator, data_dir):
    return make_service(repo, generator, data_dir)


def req(name="core-corpus", size=3, reuse=None):
    return GenerateTestsetRequest(name=name, size=size, reuse_kg_from=reuse)


# ---------- generate: versioning, reuse, manifest ----------


def test_versions_count_up_per_name_and_link_to_the_previous_version(service, repo):
    v1 = service.generate(req())
    assert (v1.version, v1.parent_id) == (1, None)
    v2 = service.generate(req())
    assert (v2.version, v2.parent_id) == (2, v1.id)
    other = service.generate(req(name="other"))
    assert (other.version, other.parent_id) == (1, None)
    assert {s.id for s in repo.list()} == {v1.id, v2.id, other.id}


def test_reusing_a_graph_passes_its_bytes_and_records_the_source(service, generator, repo):
    first = service.generate(req())
    saved_graph = repo.load_knowledge_graph(first.id)

    second = service.generate(req(reuse=first.id))
    assert generator.calls[-1]["knowledge_graph"] == saved_graph
    assert repo.get(second.id).manifest.kg_source_id == first.id


def test_reusing_an_unknown_test_set_raises_before_generating(service, generator):
    with pytest.raises(TestsetNotFound):
        service.generate(req(reuse="ts_20260101_deadbeef"))
    assert generator.calls == []


@pytest.mark.parametrize(
    "bad",
    [
        req(name=""),
        req(name="has space"),
        req(name="x" * 65),
        req(name="../escape"),
        req(name="ok\n"),
        req(size=0),
        req(size=101),
        req(size=True),
        GenerateTestsetRequest(name="ok", size="10"),
    ],
    ids=["empty-name", "space", "too-long", "path", "trailing-newline", "size-0", "size-101", "size-bool", "size-str"],
)
def test_invalid_requests_are_rejected(service, generator, bad):
    with pytest.raises(InvalidRequest):
        service.generate(bad)
    with pytest.raises(InvalidRequest):
        service.estimate(bad)
    assert generator.calls == []


def test_reusing_a_test_set_without_a_graph_is_invalid(service, repo):
    no_graph = make_testset(name="legacy", with_kg=False)
    repo.save(no_graph)
    with pytest.raises(InvalidRequest, match="no saved knowledge graph"):
        service.generate(req(reuse=no_graph.manifest.id))


def test_manifest_records_generator_usage_counts_corpus_and_hash(service, generator, repo, data_dir):
    summary = service.generate(req(size=4))
    m = repo.get(summary.id).manifest
    assert m.generator == generator.info
    assert m.usage is not None and m.usage.total_tokens > 0
    assert (m.size, m.requested_size) == (4, 4)
    assert m.corpus == corpus_fingerprint(data_dir)
    assert m.has_knowledge_graph
    assert len(m.samples_sha256) == 64


class ShortGenerator(FakeTestsetGenerator):
    """Returns fewer samples than requested, like Ragas on a small corpus."""

    def __init__(self, keep):
        super().__init__()
        self.keep = keep

    def generate(self, documents, size, **kwargs):
        result = super().generate(documents, size, **kwargs)
        return replace(result, samples=result.samples[: self.keep])


def test_a_shortfall_is_saved_with_requested_and_actual_sizes(repo, data_dir):
    service = make_service(repo, ShortGenerator(keep=2), data_dir)
    summary = service.generate(req(size=5))
    assert (summary.size, summary.requested_size) == (2, 5)


def test_generating_no_samples_fails_and_saves_nothing(repo, data_dir):
    service = make_service(repo, ShortGenerator(keep=0), data_dir)
    with pytest.raises(GenerationError, match="no samples"):
        service.generate(req())
    assert repo.list() == []


class IncompatibleGenerator(FakeTestsetGenerator):
    def preflight(self):
        raise ModelIncompatible("temperature 1e-08 is not supported")


def test_an_incompatible_model_saves_nothing_and_reaches_the_caller(repo, data_dir):
    service = make_service(repo, IncompatibleGenerator(), data_dir)
    with pytest.raises(ModelIncompatible):
        service.generate(req())
    assert repo.list() == []


def test_an_incompatible_model_fails_the_job(repo, data_dir):
    service = make_service(repo, IncompatibleGenerator(), data_dir)
    status = service._jobs.wait(service.start_generation(req()), timeout=TIMEOUT)
    assert status.state is JobState.FAILED
    assert "ModelIncompatible" in status.error and "temperature" in status.error
    assert repo.list() == []


# ---------- progress ----------


def collect():
    events = []
    return events, lambda progress, message: events.append((progress, message))


def test_progress_is_ordered_and_never_decreases(service):
    events, report = collect()
    service.generate(req(), report)
    fractions = [p for p, _ in events]
    assert fractions == sorted(fractions)
    assert fractions[0] == 0.0 and fractions[-1] == 1.0
    assert any("graph" in m.lower() for _, m in events)  # the build stage is reported


def test_reusing_a_graph_reports_no_build_stage(service):
    first = service.generate(req())
    events, report = collect()
    service.generate(req(reuse=first.id), report)
    assert not any("graph" in m.lower() for _, m in events)
    fractions = [p for p, _ in events]
    assert fractions == sorted(fractions) and fractions[-1] == 1.0


# ---------- estimate ----------


def test_reusing_a_graph_lowers_the_input_estimate(service):
    first = service.generate(req())
    fresh = make_service(InMemoryDatasetRepository(), FakeTestsetGenerator(), service._data_dir)
    without = fresh.estimate(req())
    with_reuse = service.estimate(req(reuse=first.id))
    assert with_reuse.reuses_kg and not without.reuses_kg
    assert with_reuse.est_input_tokens < without.est_input_tokens
    assert without.corpus_tokens > 0
    assert without.model == "fake-sdg-model"


def test_cost_is_none_without_pricing_and_tokens_times_price_with_it(repo, generator, data_dir):
    unpriced = make_service(repo, generator, data_dir).estimate(req())
    assert unpriced.est_cost_usd is None

    priced = make_service(repo, generator, data_dir, price_input_per_mtok=2.0, price_output_per_mtok=8.0).estimate(req())
    expected = priced.est_input_tokens / 1e6 * 2.0 + priced.est_output_tokens / 1e6 * 8.0
    assert priced.est_cost_usd == pytest.approx(expected, abs=1e-4)


def test_pricing_needs_both_rates(repo, generator, data_dir):
    assert make_service(repo, generator, data_dir, price_input_per_mtok=2.0).estimate(req()).est_cost_usd is None


def test_calibration_uses_a_matching_past_test_set(service, repo):
    assert service.estimate(req(size=3)).calibrated is False  # nothing to calibrate from yet

    past = service.generate(req(size=3))
    usage = repo.get(past.id).manifest.usage
    est = service.estimate(req(size=3))
    assert est.calibrated is True
    # For the past test set's own size and mode, the calibrated estimate is its actual usage.
    assert (est.est_input_tokens, est.est_output_tokens) == (usage.input_tokens, usage.output_tokens)
    # Scaling still applies to other sizes.
    assert service.estimate(req(size=6)).est_input_tokens > est.est_input_tokens


def test_calibration_ignores_other_models_other_corpora_and_missing_usage(service, repo, data_dir):
    digest_files = corpus_fingerprint(data_dir).files
    other_model = make_testset(name="a")
    other_model = replace(
        other_model,
        manifest=replace(other_model.manifest, corpus=replace(other_model.manifest.corpus, files=digest_files)),
    )  # same corpus, but generated with "gpt-4.1", not the fake model
    no_usage = make_testset(name="b", version=1)
    no_usage = replace(
        no_usage,
        manifest=replace(
            no_usage.manifest,
            corpus=replace(no_usage.manifest.corpus, files=digest_files),
            generator=FakeTestsetGenerator().info,
            usage=None,
        ),
    )
    other_corpus = make_testset(name="c")  # fixture corpus fingerprint differs from data_dir
    other_corpus = replace(other_corpus, manifest=replace(other_corpus.manifest, generator=FakeTestsetGenerator().info))
    for ts in (other_model, no_usage, other_corpus):
        repo.save(ts)
    assert service.estimate(req()).calibrated is False


# ---------- staleness ----------


def test_test_sets_become_stale_when_the_corpus_changes(service, data_dir):
    service.generate(req())
    assert [i.is_stale for i in service.list_testsets()] == [False]

    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "A brand new document.")
    doc.save(str(data_dir / "new.pdf"))
    assert [i.is_stale for i in service.list_testsets()] == [True]


def test_list_is_newest_first_and_get_loads_samples(service):
    a = service.generate(req(name="a"))
    b = service.generate(req(name="b"))
    assert [i.summary.id for i in service.list_testsets()] == [b.id, a.id]
    assert len(service.get_testset(a.id).samples) == 3
    with pytest.raises(TestsetNotFound):
        service.get_testset("ts_20260101_deadbeef")


# ---------- jobs (integration with the real JobRunner) ----------


class BlockingGenerator(FakeTestsetGenerator):
    def __init__(self):
        super().__init__()
        self.started, self.release = threading.Event(), threading.Event()

    def generate(self, documents, size, **kwargs) -> GenerationResult:
        self.started.set()
        self.release.wait(TIMEOUT)
        return super().generate(documents, size, **kwargs)


def test_start_generation_runs_as_a_job(service):
    job_id = service.start_generation(req())
    status = service._jobs.wait(job_id, timeout=TIMEOUT)
    assert status.state is JobState.COMPLETED
    assert status.kind == "sdg" and "core-corpus" in status.label
    assert service.get_testset(status.result_id).manifest.name == "core-corpus"
    assert service.job_status(job_id) == status
    assert service.active_job() is None


def test_only_one_generation_runs_at_a_time(repo, data_dir):
    generator = BlockingGenerator()
    service = make_service(repo, generator, data_dir)
    first = service.start_generation(req())
    assert generator.started.wait(TIMEOUT)
    assert service.active_job().job_id == first
    with pytest.raises(JobAlreadyRunning):
        service.start_generation(req(name="second"))
    generator.release.set()
    assert service._jobs.wait(first, timeout=TIMEOUT).state is JobState.COMPLETED


def test_start_generation_validates_before_queueing(service):
    with pytest.raises(InvalidRequest):
        service.start_generation(req(size=0))
    assert service._jobs.recent() == []


# ---------- container ----------


def test_container_sdg_service_shares_the_container_job_runner(container):
    assert isinstance(container.sdg, SdgService)
    assert container.sdg is container.sdg
    assert container.sdg._jobs is container.jobs


# ---------- CLI ----------


@pytest.fixture
def cli_env(monkeypatch, tmp_path, data_dir):
    monkeypatch.setattr("rag_chatbot.config.load_dotenv", lambda *_a, **_k: None)
    monkeypatch.setenv("SDG_PROVIDER", "fake")
    monkeypatch.setenv("DATA_DIR", str(data_dir))
    monkeypatch.setenv("DATASETS_DIR", str(tmp_path / "cli-datasets"))
    monkeypatch.setenv("INDEX_DIR", str(tmp_path / "cli-index"))
    for var in ("SDG_PRICE_INPUT_PER_MTOK", "SDG_PRICE_OUTPUT_PER_MTOK"):
        monkeypatch.delenv(var, raising=False)
    return load_script("generate_testset").main


def test_cli_with_yes_generates_and_prints_the_id(cli_env, capsys, tmp_path):
    assert cli_env(["--name", "core-corpus", "--size", "2", "--yes"]) == 0
    out = capsys.readouterr().out
    assert "Estimate:" in out and "cost unknown" in out
    saved = list((tmp_path / "cli-datasets").glob("ts_*"))
    assert len(saved) == 1 and saved[0].name in out


def test_cli_answer_no_generates_nothing(cli_env, capsys, monkeypatch, tmp_path):
    monkeypatch.setattr("builtins.input", lambda _prompt: "n")
    assert cli_env(["--name", "core-corpus", "--size", "2"]) == 0
    assert "Cancelled" in capsys.readouterr().out
    assert not list((tmp_path / "cli-datasets").glob("ts_*"))


@pytest.mark.parametrize("size", ["0", "500"])
def test_cli_invalid_size_exits_non_zero(cli_env, capsys, size):
    assert cli_env(["--name", "core-corpus", "--size", size, "--yes"]) == 1
    assert "Size must be" in capsys.readouterr().err


def test_cli_non_numeric_size_exits_non_zero(cli_env):
    with pytest.raises(SystemExit) as exc:
        cli_env(["--name", "core-corpus", "--size", "lots"])
    assert exc.value.code != 0


def test_cli_runs_while_the_app_holds_the_index_lock(cli_env, tmp_path):
    holder = QdrantClient(path=str(tmp_path / "cli-index" / "qdrant"))  # what the running app holds
    try:
        assert cli_env(["--name", "core-corpus", "--size", "1", "--yes"]) == 0
    finally:
        holder.close()


# ---------- review regressions ----------


class TaggedGenerator(FakeTestsetGenerator):
    """Like the fake, but each instance writes different questions (so concurrent runs get different ids)."""

    def __init__(self, tag):
        super().__init__()
        self.tag = tag

    def generate(self, documents, size, **kwargs):
        result = super().generate(documents, size, **kwargs)
        samples = [replace(s, user_input=f"{s.user_input} [{self.tag}]") for s in result.samples]
        return replace(result, samples=samples)


class RacingRepo:
    """A FileDatasetRepository whose first next_version() waits for the other writer, forcing both to read v1."""

    def __init__(self, root, barrier):
        self._repo = FileDatasetRepository(root)
        self._barrier = barrier
        self._first = True

    def __getattr__(self, name):
        return getattr(self._repo, name)

    def next_version(self, name):
        version = self._repo.next_version(name)
        if self._first:
            self._first = False
            self._barrier.wait(TIMEOUT)
        return version


def test_concurrent_generations_of_one_name_get_distinct_versions(tmp_path, data_dir):
    root, barrier, results, errors = tmp_path / ".datasets", threading.Barrier(2), [], []

    def run(tag):
        service = make_service(RacingRepo(root, barrier), TaggedGenerator(tag), data_dir)
        try:
            results.append(service.generate(req(size=2)))
        except Exception as e:  # surfaced below
            errors.append(e)

    threads = [threading.Thread(target=run, args=(tag,)) for tag in ("app", "cli")]
    for t in threads:
        t.start()
    for t in threads:
        t.join(TIMEOUT)

    assert errors == []
    assert sorted(r.version for r in results) == [1, 2]
    saved = FileDatasetRepository(root).list()
    assert sorted(s.version for s in saved) == [1, 2]
    v2 = next(r for r in results if r.version == 2)
    v1 = next(r for r in results if r.version == 1)
    # parent_id is the newest version *saved* when v2 was saved. If v2 saved while v1 was still being
    # written, there was none yet; it must never point anywhere else.
    assert v2.parent_id in (None, v1.id)
    assert v1.parent_id is None


@pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf"), "2.0", True])
def test_invalid_prices_are_rejected(repo, generator, data_dir, bad):
    with pytest.raises(ValueError, match="finite, non-negative"):
        make_service(repo, generator, data_dir, price_input_per_mtok=bad, price_output_per_mtok=1.0)


def test_zero_prices_are_allowed(repo, generator, data_dir):
    est = make_service(repo, generator, data_dir, price_input_per_mtok=0, price_output_per_mtok=0).estimate(req())
    assert est.est_cost_usd == 0


@pytest.mark.parametrize("raw", ["-1", "nan", "inf", "-inf", "cheap"])
def test_config_rejects_invalid_price_env_values(monkeypatch, raw):
    from rag_chatbot.config import load_settings

    monkeypatch.setattr("rag_chatbot.config.load_dotenv", lambda *_a, **_k: None)
    monkeypatch.setenv("SDG_PRICE_INPUT_PER_MTOK", raw)
    with pytest.raises(ValueError, match="SDG_PRICE_INPUT_PER_MTOK"):
        load_settings()


def test_cli_reports_an_invalid_price_and_exits_non_zero(cli_env, monkeypatch, capsys):
    monkeypatch.setenv("SDG_PRICE_OUTPUT_PER_MTOK", "-3")
    assert cli_env(["--name", "core-corpus", "--size", "1", "--yes"]) == 1
    assert "SDG_PRICE_OUTPUT_PER_MTOK" in capsys.readouterr().err


def test_estimates_carry_the_corpus_digest_and_it_tracks_changes(service, data_dir):
    est = service.estimate(req())
    assert est.corpus_digest == corpus_fingerprint(data_dir).digest == service.corpus_digest()

    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "Changed corpus.")
    doc.save(str(data_dir / "changed.pdf"))
    assert service.corpus_digest() != est.corpus_digest
    assert service.estimate(req()).corpus_digest == service.corpus_digest()
