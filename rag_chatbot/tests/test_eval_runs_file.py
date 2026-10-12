"""Integration tests for FileEvalRunRepository: crash safety and persistence across instances."""

import json
import os

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
    assert (root / m.run_id / "samples.jsonl").exists()  # written, but not committed
    uncommitted = reopened.get(m.run_id)
    assert uncommitted.manifest.status is RunStatus.RUNNING  # the manifest is the commit point
    assert uncommitted.samples == []  # so the uncommitted samples aren't exposed
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
        try:
            for i in range(1, 301):
                repo.update_progress(m.run_id, i / 300, f"step {i}")
        except Exception as e:  # report it instead of leaving the reader spinning
            errors.append(f"writer: {e!r}")
        finally:
            done.set()

    reader_repo = FileEvalRunRepository(root)  # a separate instance, like another part of the app
    t = threading.Thread(target=writer)
    t.start()
    while not done.wait(0):
        try:
            got = reader_repo.get(m.run_id).manifest
            if got.message != "Starting" and got.message != f"step {round(got.progress * 300)}":
                errors.append((got.progress, got.message))
            reader_repo.list()
        except Exception as e:  # e.g. JSONDecodeError from a half-written file
            errors.append(repr(e))
    t.join(10)
    assert errors == []


def _finish_in_another_process(root, run_id, outcome, barrier, results):
    """Runs in a child process: finish the run with this process's own outcome."""
    from factories import make_sample_results

    from rag_chatbot.adapters.eval_runs.file_repository import FileEvalRunRepository
    from rag_chatbot.domain.evals import RunFinalized

    repo = FileEvalRunRepository(root)
    barrier.wait()
    try:
        if outcome == "complete":
            repo.complete(run_id, make_sample_results(2), {"faithfulness": 0.9}, judge_usage=None, answer_usage=None)
        else:
            repo.fail(run_id, "failed in the other process", samples=make_sample_results(1))
        results.put((outcome, "won"))
    except RunFinalized:
        results.put((outcome, "finalized"))


def _assert_single_winner(repo, run_id, outcomes):
    assert sorted(r for _, r in outcomes) == ["finalized", "won"]
    winner = next(o for o, r in outcomes if r == "won")
    got = repo.get(run_id)
    if winner == "complete":
        assert got.manifest.status is RunStatus.COMPLETED and len(got.samples) == 2
    else:
        assert got.manifest.status is RunStatus.FAILED and len(got.samples) == 1


def test_two_instances_cannot_both_finish_a_run(root):
    import queue
    import threading

    FileEvalRunRepository(root).create(make_run_manifest())
    barrier, results = threading.Barrier(2), queue.Queue()
    threads = [
        threading.Thread(target=_finish_in_another_process, args=(root, "run_20261012-090000_baseline_ab12", o, barrier, results))
        for o in ("complete", "fail")
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    outcomes = [results.get_nowait() for _ in range(2)]
    _assert_single_winner(FileEvalRunRepository(root), "run_20261012-090000_baseline_ab12", outcomes)


def test_two_processes_cannot_both_finish_a_run(root):
    import multiprocessing

    run_id = "run_20261012-090000_baseline_ab12"
    FileEvalRunRepository(root).create(make_run_manifest(run_id))
    ctx = multiprocessing.get_context("spawn")
    barrier, results = ctx.Barrier(2), ctx.Queue()
    procs = [ctx.Process(target=_finish_in_another_process, args=(root, run_id, o, barrier, results)) for o in ("complete", "fail")]
    for p in procs:
        p.start()
    for p in procs:
        p.join(60)
        assert p.exitcode == 0
    outcomes = [results.get(timeout=5) for _ in range(2)]
    _assert_single_winner(FileEvalRunRepository(root), run_id, outcomes)


def test_the_container_stores_runs_under_evals_dir(container, settings):
    assert isinstance(container.eval_runs, FileEvalRunRepository)
    assert container.eval_runs.root == settings.evals_dir


# ---------- follow-up review regressions ----------


def test_startup_cleanup_waits_for_in_flight_writes(root):
    """A new instance must not delete another writer's temp file while that writer holds the lock."""
    import threading

    writer = FileEvalRunRepository(root)
    m = make_run_manifest()
    writer.create(m)
    in_flight = root / m.run_id / ".manifest.json.tmp-inflight"
    holding, release, init_done = threading.Event(), threading.Event(), threading.Event()

    def write_slowly():
        with writer._writing():  # the same lock a real write holds while its temp file exists
            in_flight.write_text("{}")
            holding.set()
            release.wait(10)
            os.replace(in_flight, root / m.run_id / "manifest.json.bak")  # the write finishes normally

    def start_new_instance():
        FileEvalRunRepository(root)
        init_done.set()

    t1 = threading.Thread(target=write_slowly)
    t1.start()
    assert holding.wait(10)
    t2 = threading.Thread(target=start_new_instance)
    t2.start()
    assert not init_done.wait(0.3), "startup cleanup ran while a write was in flight"
    assert in_flight.exists()
    release.set()
    t1.join(10)
    t2.join(10)
    assert init_done.is_set()
    assert (root / m.run_id / "manifest.json.bak").exists()  # the in-flight write wasn't sabotaged


def test_folders_from_interrupted_creates_are_reclaimed(root):
    FileEvalRunRepository(root)
    abandoned = root / "run_20261012-090000_baseline_ab12"
    abandoned.mkdir()  # create() was killed after mkdir, before the manifest
    (root / ".not-a-run").mkdir()

    repo = FileEvalRunRepository(root)  # startup cleanup removes it
    assert not abandoned.exists()
    assert (root / ".not-a-run").exists()  # only run-shaped folders are touched
    repo.create(make_run_manifest())
    assert repo.get("run_20261012-090000_baseline_ab12").manifest.status is RunStatus.RUNNING


def test_create_reclaims_an_abandoned_folder_without_a_restart(root):
    repo = FileEvalRunRepository(root)
    (root / "run_20261012-090000_baseline_ab12").mkdir()
    repo.create(make_run_manifest())  # no RunAlreadyExists
    assert len(repo.list()) == 1


def test_failing_after_an_interrupted_complete_drops_its_uncommitted_samples(root, monkeypatch):
    repo = FileEvalRunRepository(root)
    m = make_run_manifest()
    repo.create(m)

    def crash(*_args, **_kwargs):
        raise OSError("killed before the manifest was written")

    monkeypatch.setattr(FileEvalRunRepository, "_write_manifest", staticmethod(crash))
    with pytest.raises(OSError):
        repo.complete(m.run_id, make_sample_results(3), {"faithfulness": 0.9}, judge_usage=None, answer_usage=None)
    monkeypatch.undo()
    assert (root / m.run_id / "samples.jsonl").exists()

    recovered = FileEvalRunRepository(root)
    recovered.fail(m.run_id, "Interrupted (server restarted)")  # no samples of its own
    got = recovered.get(m.run_id)
    assert got.manifest.status is RunStatus.FAILED
    assert got.samples == []  # not the crashed complete()'s samples
    assert not (root / m.run_id / "samples.jsonl").exists()


def test_progress_updates_leave_samples_alone(root, monkeypatch):
    repo = FileEvalRunRepository(root)
    m = make_run_manifest()
    repo.create(m)
    repo.update_progress(m.run_id, 0.5, "halfway")
    repo.complete(m.run_id, make_sample_results(2), {}, judge_usage=None, answer_usage=None)
    assert len(repo.get(m.run_id).samples) == 2


# ---------- containment and corrupt runs ----------


@pytest.fixture
def outside(tmp_path):
    """A folder outside the store holding things a symlink could point at."""
    d = tmp_path / "outside"
    d.mkdir()
    return d


def _make_external_run(outside, root):
    """A valid-looking run that lives outside the store."""
    elsewhere = FileEvalRunRepository(outside / "other-store")
    elsewhere.create(make_run_manifest())
    return outside / "other-store" / "run_20261012-090000_baseline_ab12"


def test_a_symlinked_run_folder_is_neither_listed_read_written_nor_cleaned(root, outside):
    from rag_chatbot.domain.evals import RunAlreadyExists

    target = _make_external_run(outside, root)
    bait = target / ".manifest.json.tmp-not-yours"
    bait.write_text("must survive")
    before = (target / "manifest.json").read_text()
    root.mkdir(parents=True)
    (root / target.name).symlink_to(target, target_is_directory=True)

    repo = FileEvalRunRepository(root)  # startup cleanup must not reach through the link
    assert bait.exists()
    assert repo.list() == []
    with pytest.raises(RunNotFound):
        repo.get(target.name)
    with pytest.raises(RunNotFound):
        repo.update_progress(target.name, 0.5, "through the link")
    with pytest.raises(RunNotFound):
        repo.fail(target.name, "through the link")
    with pytest.raises(RunAlreadyExists):
        repo.create(make_run_manifest(target.name))
    assert (target / "manifest.json").read_text() == before  # nothing outside was modified


def test_a_symlinked_manifest_is_refused(root, outside):
    from rag_chatbot.domain.evals import RunCorrupt

    secret = outside / "secret.json"
    secret.write_text('{"not": "a run"}')
    repo = FileEvalRunRepository(root)
    folder = root / "run_linked"
    folder.mkdir()
    (folder / "manifest.json").symlink_to(secret)

    assert repo.list() == []
    with pytest.raises(RunCorrupt, match="symbolic link"):
        repo.get("run_linked")
    with pytest.raises(RunCorrupt):
        repo.update_progress("run_linked", 0.1, "x")
    assert secret.read_text() == '{"not": "a run"}'


def test_a_symlinked_samples_file_is_refused(root, outside):
    from rag_chatbot.domain.evals import RunCorrupt

    secret = outside / "secret.jsonl"
    secret.write_text('{"user_input": "leaked"}\n')
    repo = FileEvalRunRepository(root)
    m = make_run_manifest()
    repo.create(m)
    repo.complete(m.run_id, make_sample_results(1), {}, judge_usage=None, answer_usage=None)
    samples = root / m.run_id / "samples.jsonl"
    samples.unlink()
    samples.symlink_to(secret)

    with pytest.raises(RunCorrupt, match="symbolic link"):
        repo.get(m.run_id)


@pytest.mark.parametrize(
    "corrupt",
    [
        lambda d: {**d, "status": "paused"},  # not a RunStatus
        lambda d: {k: v for k, v in d.items() if k != "testset_id"},  # missing field
        lambda d: {**d, "started_at": "yesterday"},  # unparseable date
        None,  # not JSON at all
    ],
    ids=["unknown-status", "missing-field", "bad-date", "not-json"],
)
def test_a_corrupt_manifest_is_skipped_by_list_and_reported_by_get(root, corrupt):
    from rag_chatbot.domain.evals import RunCorrupt

    repo = FileEvalRunRepository(root)
    good, bad = make_run_manifest("run_good"), make_run_manifest("run_bad")
    repo.create(good)
    repo.create(bad)
    path = root / "run_bad" / "manifest.json"
    path.write_text("{ truncated" if corrupt is None else json.dumps(corrupt(json.loads(path.read_text()))))

    assert [s.run_id for s in repo.list()] == ["run_good"]  # the good run is still listed
    with pytest.raises(RunCorrupt, match="run_bad"):
        repo.get("run_bad")
    with pytest.raises(RunCorrupt):
        repo.fail("run_bad", "x")
    assert repo.get("run_good").manifest.status is RunStatus.RUNNING
