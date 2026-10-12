"""Build (or rebuild) the vector indexes for all three pipelines.

    uv run python scripts/ingest.py           # build only if missing or stale
    uv run python scripts/ingest.py --force   # always rebuild

Stop the Streamlit app first: embedded Qdrant allows one process at a time.
"""

import argparse
import sys

from rag_chatbot.config import load_settings
from rag_chatbot.container import ConfigError, build_index_manager


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="rebuild even if the index is current")
    args = parser.parse_args()

    try:
        index = build_index_manager(load_settings())
    except ConfigError as e:
        print(f"Config error: {e}", file=sys.stderr)
        return 1

    built = index.ensure(print, force=args.force)
    if not built:
        print("Index is current; nothing to do (use --force to rebuild).")
    for name, count in index.counts().items():
        print(f"  {name:<14} {count:>8,} vectors" if count is not None else f"  {name:<14} missing")
    return 0


if __name__ == "__main__":
    sys.exit(main())
