import threading
import time

import pytest

from rag_chatbot.services.jobs import (
    HISTORY_LIMIT,
    JobAlreadyRunning,
    JobNotFound,
    JobRunner,
    JobState,
)

TIMEOUT = 5  # seconds; generous so slow CI machines don't flake


class GatedThread(threading.Thread):
    """A thread whose start() is held back until `release()`, so tests can observe QUEUED."""

    gate = None

    def start(self):
        threading.Thread(target=lambda: (GatedThread.gate.wait(TIMEOUT), super(GatedThread, self).start()), daemon=True).start()


@pytest.fixture
def runner():
    return JobRunner()


def blocking_job(started: threading.Event, release: threading.Event, result="result-1"):
    def fn(report):
        started.set()
        release.wait(TIMEOUT)
        return result

    return fn


def test_job_moves_through_queued_running_completed():
    GatedThread.gate = threading.Event()
    runner = JobRunner(thread_factory=GatedThread)
    started, release = threading.Event(), threading.Event()

    job_id = runner.submit("sdg", "Generate 'core' (2)", blocking_job(started, release, "ts_1"))
    queued = runner.status(job_id)
    assert queued.state is JobState.QUEUED and queued.started_at is None

    GatedThread.gate.set()
    assert started.wait(TIMEOUT)
    running = runner.status(job_id)
    assert running.state is JobState.RUNNING and running.started_at is not None

    release.set()
    done = runner.wait(job_id, timeout=TIMEOUT)
    assert done.state is JobState.COMPLETED
    assert done.result_id == "ts_1"
    assert done.error is None and done.progress == 1.0
    assert done.created_at <= done.started_at <= done.finished_at
    assert (done.kind, done.label) == ("sdg", "Generate 'core' (2)")


def test_failing_job_records_the_error_and_the_runner_keeps_working(runner):
    def boom(report):
        raise ValueError("bad input")

    failed = runner.wait(runner.submit("eval", "Fails", boom), timeout=TIMEOUT)
    assert failed.state is JobState.FAILED
    assert failed.error == "ValueError: bad input"
    assert failed.result_id is None and failed.finished_at is not None

    ok = runner.wait(runner.submit("eval", "Works", lambda report: "batch_1"), timeout=TIMEOUT)
    assert ok.state is JobState.COMPLETED and ok.result_id == "batch_1"


def test_a_job_returning_a_non_string_result_fails_clearly(runner):
    status = runner.wait(runner.submit("sdg", "Bad return", lambda report: 42), timeout=TIMEOUT)
    assert status.state is JobState.FAILED and "TypeError" in status.error


def test_a_job_may_return_no_result_id(runner):
    status = runner.wait(runner.submit("sdg", "No result", lambda report: None), timeout=TIMEOUT)
    assert status.state is JobState.COMPLETED and status.result_id is None


def test_submit_while_a_job_runs_raises_and_starts_nothing(runner):
    started, release = threading.Event(), threading.Event()
    first = runner.submit("sdg", "First", blocking_job(started, release))
    assert started.wait(TIMEOUT)

    second_ran = threading.Event()
    with pytest.raises(JobAlreadyRunning) as exc:
        runner.submit("eval", "Second", lambda report: second_ran.set())
    assert exc.value.active.job_id == first
    assert "First" in str(exc.value)

    release.set()
    runner.wait(first, timeout=TIMEOUT)
    assert not second_ran.is_set()
    assert [j.label for j in runner.recent()] == ["First"]


def test_progress_reports_show_up_in_status(runner):
    reported, release = threading.Event(), threading.Event()

    def fn(report):
        report(0.25, "Answering question 1 of 4")
        report(0.5, "Answering question 2 of 4")
        reported.set()
        release.wait(TIMEOUT)
        return "done"

    job_id = runner.submit("eval", "Progress", fn)
    assert reported.wait(TIMEOUT)
    status = runner.status(job_id)
    assert (status.progress, status.message) == (0.5, "Answering question 2 of 4")
    release.set()
    runner.wait(job_id, timeout=TIMEOUT)


def test_progress_is_clamped_and_late_reports_are_ignored(runner):
    saved = {}

    def fn(report):
        report(1.7, "over")
        saved["report"] = report
        return "done"

    job_id = runner.submit("sdg", "Clamp", fn)
    done = runner.wait(job_id, timeout=TIMEOUT)
    saved["report"](0.1, "too late")  # after completion: ignored
    assert runner.status(job_id) == done
    assert done.progress == 1.0


def test_active_and_recent(runner):
    assert runner.active() is None
    started, release = threading.Event(), threading.Event()
    job_id = runner.submit("sdg", "Active", blocking_job(started, release))
    assert started.wait(TIMEOUT)
    assert runner.active().job_id == job_id
    release.set()
    runner.wait(job_id, timeout=TIMEOUT)
    assert runner.active() is None

    for i in range(HISTORY_LIMIT + 5):
        runner.wait(runner.submit("sdg", f"Job {i}", lambda report: None), timeout=TIMEOUT)
    history = runner.recent(limit=1000)
    assert len(history) == HISTORY_LIMIT
    assert history[0].label == f"Job {HISTORY_LIMIT + 4}"  # newest first
    assert [j.created_at for j in history] == sorted((j.created_at for j in history), reverse=True)
    assert len(runner.recent()) == 20


def test_a_blocked_waiter_survives_its_job_being_pushed_out_of_history(runner, monkeypatch):
    """Regression: wait_for releases the lock, so a burst of submits could evict the finished job
    before the waiter woke up, and the waiter's lookup raised KeyError."""
    started, release = threading.Event(), threading.Event()
    target = runner.submit("sdg", "Target", blocking_job(started, release, "target-result"))
    assert started.wait(TIMEOUT)

    burst_done = threading.Event()

    def finish_target_then_burst():
        release.set()
        while runner.status(target).state.is_active:
            time.sleep(0.001)
        for i in range(HISTORY_LIMIT + 10):
            runner.wait(runner.submit("sdg", f"Filler {i}", lambda report: None), timeout=TIMEOUT)
        burst_done.set()
        with runner._changed:
            runner._changed.notify_all()

    real_wait_for = runner._changed.wait_for
    raced = []

    def wait_for_while_others_run(predicate, timeout=None):
        if raced:  # only the first waiter (the target's) gets the forced interleaving
            return real_wait_for(predicate, timeout=timeout)
        raced.append(True)
        threading.Thread(target=finish_target_then_burst, daemon=True).start()
        real_wait_for(burst_done.is_set, timeout=TIMEOUT)  # lock released while the burst runs
        return predicate()

    monkeypatch.setattr(runner._changed, "wait_for", wait_for_while_others_run)

    status = runner.wait(target, timeout=TIMEOUT)
    assert status.state is JobState.COMPLETED and status.result_id == "target-result"
    assert len(runner.recent(limit=1000)) == HISTORY_LIMIT  # trimmed again once the waiter left


def test_unknown_job_id_raises(runner):
    with pytest.raises(JobNotFound):
        runner.status("job_nope")
    with pytest.raises(JobNotFound):
        runner.wait("job_nope", timeout=0.01)


def test_wait_returns_current_status_on_timeout(runner):
    started, release = threading.Event(), threading.Event()
    job_id = runner.submit("sdg", "Slow", blocking_job(started, release))
    assert started.wait(TIMEOUT)
    assert runner.wait(job_id, timeout=0.05).state is JobState.RUNNING
    release.set()
    assert runner.wait(job_id, timeout=TIMEOUT).state is JobState.COMPLETED


def test_a_thread_that_fails_to_start_does_not_block_future_jobs():
    def broken_factory(**_kwargs):
        raise RuntimeError("can't start new thread")

    runner = JobRunner(thread_factory=broken_factory)
    with pytest.raises(RuntimeError):
        runner.submit("sdg", "Never starts", lambda report: None)
    assert runner.active() is None
    assert runner.recent()[0].state is JobState.FAILED


def test_snapshots_stay_consistent_under_concurrent_updates(runner):
    n = 1000

    def fn(report):
        for i in range(1, n + 1):
            report(i / n, f"step {i}")
        return "done"

    job_id = runner.submit("eval", "Busy", fn)
    errors = []
    while True:
        s = runner.status(job_id)
        if s.state is JobState.RUNNING and s.progress is not None:
            # message and progress always come from the same report() call
            if s.message != f"step {round(s.progress * n)}":
                errors.append((s.progress, s.message))
        if not s.state.is_active:
            break
    assert errors == []
    assert runner.wait(job_id, timeout=TIMEOUT).state is JobState.COMPLETED


def test_the_container_owns_one_runner(container):
    assert isinstance(container.jobs, JobRunner)
    assert container.jobs is container.jobs
