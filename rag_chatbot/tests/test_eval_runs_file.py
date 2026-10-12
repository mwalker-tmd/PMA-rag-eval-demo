"""Integration tests for FileEvalRunRepository: crash safety and persistence across instances."""

import json

import pytest
from factories import make_run_manifest, make_sample_results

from rag_chatbot.adapters.eval_runs import file_repository
from rag_chatbot.adapters.eval_runs.file_repository import FileEvalRunRepository
from rag_chatbot.domain.evals import RunNotFound, RunStatus


@pytest.fixture
def root(tmp_path):
    return tmp_path / ".evals"


def test_a_crash_during_a_manifest_update_leaves_the_previous_manifest_readable(root, monkeypatch):
    repo = FileEvalRunRepository(root)
    m = make_run_manifest()
    repo.create(m)
    repo.update_progress(m.run_id, 0.3, "Answering 3 of 10")

    def crash(src, dst):
        raise OSError("power cut during replace")

    monkeypatch.setattr(file_repository.os, "replace", crash)
    with pytest.raises(OSError):
        repo.update_progress(m.run_id, 0.6, "Answering 6 of 10")
    monkeypatch.undo()

    raw = json.loads((root / m.run_id / "manifest.json").read_text())  # still valid JSON
    assert raw["progress"] == 0.3
    assert FileEvalRunRepository(root).get(m.run_id).manifest.message == "Answering 3 of 10"
    assert not [p for p in (root / m.run_id).iterdir() if ".tmp-" in p.name]  # temp file cleaned up


def test_a_crash_between_samples_and_manifest_leaves_the_run_running_and_finishable(root, monkeypatch):
    repo = FileEvalRunRepository(root)
    m = make_run_manifest()
    repo.create(m)

    def crash(*_args, **_kwargs):
        raise OSError("killed before the manifest was written")

    monkeypatch.setattr(FileEvalRunRepository, "_write_manifest", staticmethod(crash))
    with pytest.raises(OSError):
        repo.complete(m.run_id, make_sample_results(2), {"faithfulness": 0.9}, judge_usage=None, answer_usage=None)
    monkeypatch.undo()

    reopened = FileEvalRunRepository(root)
    assert reopened.get(m.run_id).manifest.status is RunStatus.RUNNING  # the manifest is the commit point
    reopened.fail(m.run_id, "Interrupted (server restarted)")  # e.g. recovery at startup
    assert reopened.get(m.run_id).manifest.status is RunStatus.FAILED


def test_a_new_instance_sees_every_run_and_status(root):
    first = FileEvalRunRepository(root)
    running, done, failed = (make_run_manifest(f"run_{name}") for name in ("running", "done", "failed"))
    for m in (running, done, failed):
        first.create(m)
    first.update_progress("run_running", 0.5, "halfway")
    first.complete("run_done", make_sample_results(2), {"faithfulness": 0.8}, judge_usage=None, answer_usage=None)
    first.fail("run_failed", "boom", samples=make_sample_results(1))

    second = FileEvalRunRepository(root)
    assert {s.run_id: s.status for s in second.list()} == {
        "run_running": RunStatus.RUNNING,
        "run_done": RunStatus.COMPLETED,
        "run_failed": RunStatus.FAILED,
    }
    assert second.get("run_running").manifest.progress == 0.5
    assert second.get("run_done").samples == make_sample_results(2)
    assert len(second.get("run_failed").samples) == 1


def test_leftover_temp_files_are_removed_on_startup(root):
    repo = FileEvalRunRepository(root)
    m = make_run_manifest()
    repo.create(m)
    leftover = root / m.run_id / ".manifest.json.tmp-crashed"
    leftover.write_text("{partial")
    FileEvalRunRepository(root)
    assert not leftover.exists()


def test_a_failed_create_leaves_no_half_created_run(root, monkeypatch):
    repo = FileEvalRunRepository(root)

    def crash(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(FileEvalRunRepository, "_write_manifest", staticmethod(crash))
    with pytest.raises(OSError):
        repo.create(make_run_manifest())
    monkeypatch.undo()
    assert repo.list() == []
    repo.create(make_run_manifest())  # the id is free again


def test_samples_are_stored_as_readable_utf8(root):
    repo = FileEvalRunRepository(root)
    m = make_run_manifest()
    repo.create(m)
    repo.complete(m.run_id, make_sample_results(1), {}, judge_usage=None, answer_usage=None)
    assert "日本語" in (root / m.run_id / "samples.jsonl").read_text(encoding="utf-8")


@pytest.mark.parametrize("bad_id", ["../escape", "a/b", "", ".hidden", "run_x\n"])
def test_path_like_ids_are_not_found(root, bad_id):
    with pytest.raises(RunNotFound):
        FileEvalRunRepository(root).get(bad_id)


def test_readers_never_see_a_torn_manifest_while_a_job_writes(root):
    """The evaluation job writes progress on a worker thread while the UI polls the same run."""
    import threading

    repo = FileEvalRunRepository(root)
    m = make_run_manifest()
    repo.create(m)
    done, errors = threading.Event(), []

    def writer():
        for i in range(1, 301):
            repo.update_progress(m.run_id, i / 300, f"step {i}")
        done.set()

    reader_repo = FileEvalRunRepository(root)  # a separate instance, like another part of the app
    t = threading.Thread(target=writer)
    t.start()
    while not done.is_set():
        try:
            got = reader_repo.get(m.run_id).manifest
            if got.message != "Starting" and got.message != f"step {round(got.progress * 300)}":
                errors.append((got.progress, got.message))
            reader_repo.list()
        except Exception as e:  # e.g. JSONDecodeError from a half-written file
            errors.append(repr(e))
    t.join()
    assert errors == []


def test_the_container_stores_runs_under_evals_dir(container, settings):
    assert isinstance(container.eval_runs, FileEvalRunRepository)
    assert container.eval_runs.root == settings.evals_dir
