"""Evaluate pipelines against a saved test set, then print a comparison table.

    uv run python scripts/run_eval.py --testset ts_20261012_ab12cd34
    uv run python scripts/run_eval.py --testset ts_20261012_ab12cd34 --pipelines baseline,rerank --effort low

Answers every question with each pipeline (LLM_MODEL), scores the answers with the judge
(EVAL_JUDGE_MODEL), saves one run per pipeline under EVALS_DIR, and prints the metrics side by
side with deltas against --reference (default: the first pipeline).

Stop the app first: this needs the vector index, and the embedded index allows one process at a time.
"""

import argparse
import sys

from rag_chatbot.container import ConfigError, build_container
from rag_chatbot.domain.datasets import TestsetNotFound
from rag_chatbot.domain.errors import ModelIncompatible
from rag_chatbot.domain.evals import EvaluationError, RunStatus
from rag_chatbot.domain.models import ReasoningEffort
from rag_chatbot.services.evaluation_service import (
    PipelineUnavailable,
    RunComparison,
    StartEvaluationRequest,
)
from rag_chatbot.services.sdg_service import InvalidRequest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--testset", required=True, metavar="TESTSET_ID", help="the test set to evaluate against")
    parser.add_argument("--pipelines", default="all", help="comma-separated pipeline names, or 'all' (default)")
    parser.add_argument("--effort", choices=[e.value for e in ReasoningEffort], help="reasoning effort for LLM_MODEL")
    parser.add_argument("--reference", metavar="PIPELINE", help="pipeline to compare against (default: the first)")
    args = parser.parse_args(argv)

    pipelines = () if args.pipelines.strip().lower() == "all" else tuple(p.strip() for p in args.pipelines.split(",") if p.strip())
    request = StartEvaluationRequest(
        testset_id=args.testset,
        pipelines=pipelines,
        reasoning_effort=ReasoningEffort(args.effort) if args.effort else None,
    )

    try:
        container = build_container()
    except (ConfigError, ValueError, RuntimeError) as e:  # RuntimeError: the index is locked by the running app
        print(f"Error: {e}", file=sys.stderr)
        return 1
    try:
        return _evaluate(container.evaluation, request, args)
    finally:
        container.close()  # release the index lock


def _evaluate(service, request: StartEvaluationRequest, args) -> int:
    for warning in service.warnings():
        print(f"Warning: {warning}")

    def report(progress, message):
        pct = f"{progress:4.0%}" if progress is not None else "  … "
        print(f"[{pct}] {message}", flush=True)

    try:
        runs = service.run_evaluation(request, report)
    except (InvalidRequest, TestsetNotFound, PipelineUnavailable, ModelIncompatible, EvaluationError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print()
    for run in runs:
        status = run.status.value if run.status is RunStatus.COMPLETED else f"{run.status.value.upper()}"
        print(f"{run.pipeline:<14} {run.run_id}  {status}")
    failed = [r for r in runs if r.status is not RunStatus.COMPLETED]
    for run in failed:
        print(f"  {run.pipeline}: {service.get_run(run.run_id).manifest.error}", file=sys.stderr)

    by_pipeline = {r.pipeline: r.run_id for r in runs}
    reference = by_pipeline.get(args.reference) if args.reference else runs[0].run_id
    if args.reference and reference is None:
        print(f"Note: --reference {args.reference} wasn't part of this batch; comparing against {runs[0].pipeline}.")
        reference = runs[0].run_id
    print()
    print(format_comparison(service.compare_runs([r.run_id for r in runs], reference), {r.run_id: r.pipeline for r in runs}))
    return 0 if not failed else 2


def format_comparison(c: RunComparison, names: dict[str, str]) -> str:
    """Metrics as rows, one column per run, with a delta column for every run but the reference."""
    header = ["metric"]
    for r in c.run_ids:
        header.append(names.get(r, r))
        if r != c.reference_run_id:
            header.append("Δ")
    rows = [header]
    for m in c.metrics:
        row = [m]
        for r in c.run_ids:
            v = c.values[m][r]
            row.append("—" if v is None else f"{v:.3f}")
            if r != c.reference_run_id:
                d = c.deltas[m][r]
                row.append("" if d is None else f"{d:+.3f}")
        rows.append(row)
    widths = [max(len(row[i]) for row in rows) for i in range(len(header))]
    lines = ["  ".join(cell.ljust(w) for cell, w in zip(row, widths)).rstrip() for row in rows]
    lines.insert(1, "  ".join("-" * w for w in widths))
    if c.reference_run_id:
        lines.append(f"\nΔ is relative to {names.get(c.reference_run_id, c.reference_run_id)}.")
    if c.mixed_testsets:
        lines.append("Warning: these runs used different test sets, so their scores aren't directly comparable.")
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
