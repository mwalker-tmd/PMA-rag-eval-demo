"""Unit tests for test-set domain helpers and the corpus fingerprint, plus file-adapter integration tests."""

import hashlib
from datetime import UTC, datetime

import pytest
from factories import make_testset

from rag_chatbot.adapters.datasets.file_repository import FileDatasetRepository
from rag_chatbot.domain.datasets import CorpusFingerprint, TestsetNotFound, new_testset_id, samples_sha256
from rag_chatbot.ingestion.fingerprint import corpus_fingerprint

DAY = datetime(2026, 10, 12, 15, 30, tzinfo=UTC)


# ---------- id helper ----------


def test_id_is_stable_and_formatted():
    digest = samples_sha256(make_testset().samples)
    a = new_testset_id(DAY, "core-corpus", 1, digest)
    assert a == new_testset_id(DAY, "core-corpus", 1, digest)
    assert a.startswith("ts_20261012_") and len(a) == len("ts_20261012_") + 8


def test_id_changes_with_samples_name_or_version():
    d1 = samples_sha256(make_testset().samples)
    d2 = samples_sha256(make_testset(tag="x").samples)
    base = new_testset_id(DAY, "core-corpus", 1, d1)
    assert new_testset_id(DAY, "core-corpus", 1, d2) != base
    assert new_testset_id(DAY, "other", 1, d1) != base
    assert new_testset_id(DAY, "core-corpus", 2, d1) != base


# ---------- corpus fingerprint ----------


@pytest.fixture
def corpus(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    (d / "a.pdf").write_bytes(b"%PDF-1.4 alpha")
    (d / "b.pdf").write_bytes(b"%PDF-1.4 beta")
    (d / "notes.txt").write_text("ignored: not a PDF")
    return d


def test_fingerprint_hashes_only_pdfs(corpus):
    fp = corpus_fingerprint(corpus)
    assert fp.files == {
        "a.pdf": hashlib.sha256(b"%PDF-1.4 alpha").hexdigest(),
        "b.pdf": hashlib.sha256(b"%PDF-1.4 beta").hexdigest(),
    }


def test_fingerprint_digest_is_stable_for_an_unchanged_corpus(corpus):
    assert corpus_fingerprint(corpus).digest == corpus_fingerprint(corpus).digest
    # Order of the mapping doesn't matter.
    files = corpus_fingerprint(corpus).files
    assert CorpusFingerprint(dict(reversed(list(files.items())))).digest == CorpusFingerprint(files).digest


@pytest.mark.parametrize(
    "change",
    [
        lambda d: (d / "c.pdf").write_bytes(b"%PDF-1.4 gamma"),  # added
        lambda d: (d / "b.pdf").unlink(),  # removed
        lambda d: (d / "a.pdf").write_bytes(b"%PDF-1.4 alpha v2"),  # modified
    ],
    ids=["added", "removed", "modified"],
)
def test_fingerprint_digest_changes_with_the_corpus(corpus, change):
    before = corpus_fingerprint(corpus).digest
    change(corpus)
    assert corpus_fingerprint(corpus).digest != before


def test_index_manifest_still_hashes_pdfs_the_same_way(container, data_dir):
    pdfs = container.index.expected_manifest()["pdfs"]
    assert pdfs == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(data_dir.glob("*.pdf"))}


# ---------- file adapter integration ----------


def test_interrupted_save_leaves_nothing_visible(tmp_path, monkeypatch):
    repo = FileDatasetRepository(tmp_path / ".datasets")
    ts = make_testset()

    def boom(*_args, **_kwargs):
        raise RuntimeError("disk full")  # after samples.jsonl was written, before the manifest

    monkeypatch.setattr(FileDatasetRepository, "_write_manifest", staticmethod(boom))
    with pytest.raises(RuntimeError):
        repo.save(ts)

    assert repo.list() == []
    assert not (repo.root / ts.manifest.id).exists()
    monkeypatch.undo()
    repo.save(ts)  # the id is still free
    assert repo.get(ts.manifest.id) == ts


def test_new_instance_removes_leftovers_from_a_crashed_save(tmp_path):
    root = tmp_path / ".datasets"
    FileDatasetRepository(root)
    leftover = root / ".tmp-crashed"
    leftover.mkdir()
    (leftover / "samples.jsonl").write_text('{"partial": true}\n')

    repo = FileDatasetRepository(root)
    assert not leftover.exists()
    assert repo.list() == []


def test_non_ascii_text_round_trips(tmp_path):
    repo = FileDatasetRepository(tmp_path / ".datasets")
    ts = make_testset(tag=" — café, naïve, 日本語, 🙂")
    repo.save(ts)
    assert FileDatasetRepository(tmp_path / ".datasets").get(ts.manifest.id) == ts
    raw = (repo.root / ts.manifest.id / "samples.jsonl").read_text(encoding="utf-8")
    assert "日本語" in raw  # stored as readable UTF-8, not \u escapes


@pytest.mark.parametrize("bad_id", ["../escape", "a/b", "", ".tmp-x"])
def test_path_like_ids_are_not_found(tmp_path, bad_id):
    repo = FileDatasetRepository(tmp_path / ".datasets")
    with pytest.raises(TestsetNotFound):
        repo.get(bad_id)
