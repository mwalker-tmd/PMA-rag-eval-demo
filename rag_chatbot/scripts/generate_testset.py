"""Generate a synthetic test set from the PDFs in DATA_DIR.

    uv run python scripts/generate_testset.py --name core-corpus --size 10
    uv run python scripts/generate_testset.py --name core-corpus --size 20 --reuse-kg-from ts_20261012_ab12cd34 --yes

Shows a token/cost estimate and asks for confirmation first (skip with --yes). Safe to run
while the app is running: it doesn't open the vector index.
"""

import argparse
import sys

from rag_chatbot.container import ConfigError, build_sdg_service
from rag_chatbot.domain.datasets import GenerationError, TestsetNotFound
from rag_chatbot.domain.errors import ModelIncompatible
from rag_chatbot.services.sdg_service import GenerateTestsetRequest, InvalidRequest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True, help="test set name; versions count up per name")
    parser.add_argument("--size", type=int, default=10, help="number of samples to generate (1–100, default 10)")
    parser.add_argument("--reuse-kg-from", metavar="TESTSET_ID", help="reuse this test set's knowledge graph")
    parser.add_argument("--yes", action="store_true", help="don't ask for confirmation")
    args = parser.parse_args(argv)

    try:
        service = build_sdg_service()
        req = GenerateTestsetRequest(name=args.name, size=args.size, reuse_kg_from=args.reuse_kg_from)
        est = service.estimate(req)
    except (ConfigError, InvalidRequest, TestsetNotFound, ValueError) as e:  # ValueError: malformed env value
        print(f"Error: {e}", file=sys.stderr)
        return 1

    cost = f"~${est.est_cost_usd:.2f}" if est.est_cost_usd is not None else "cost unknown (set SDG_PRICE_* to show it)"
    print(f"Model:    {est.model}")
    print(f"Corpus:   ~{est.corpus_tokens:,} tokens")
    print(f"Estimate: ~{est.est_input_tokens:,} input + ~{est.est_output_tokens:,} output tokens, {cost}")
    notes = [n for n, on in (("reusing a saved knowledge graph", est.reuses_kg), ("calibrated from past usage", est.calibrated)) if on]
    if notes:
        print(f"          ({'; '.join(notes)})")

    if not args.yes and input("Proceed? [y/N] ").strip().lower() not in ("y", "yes"):
        print("Cancelled.")
        return 0

    def report(progress, message):
        pct = f"{progress:4.0%}" if progress is not None else "  … "
        print(f"[{pct}] {message}", flush=True)

    try:
        summary = service.generate(req, report)
    except (InvalidRequest, TestsetNotFound, GenerationError, ModelIncompatible) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print(f"Saved test set {summary.id}: '{summary.name}' v{summary.version}, {summary.size} samples")
    if summary.requested_size and summary.size < summary.requested_size:
        print(f"Note: requested {summary.requested_size}, generated {summary.size} (the corpus supported fewer distinct questions)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
