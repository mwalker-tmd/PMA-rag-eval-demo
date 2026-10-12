"""Content fingerprint of the PDF corpus, shared by the vector index and test sets."""

from __future__ import annotations

import hashlib
from pathlib import Path

from rag_chatbot.domain.datasets import CorpusFingerprint
from rag_chatbot.ingestion.loader import list_pdfs


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def corpus_fingerprint(data_dir: Path) -> CorpusFingerprint:
    return CorpusFingerprint(files={p.name: sha256_file(p) for p in list_pdfs(data_dir)})
