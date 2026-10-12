"""Builds and persists one Qdrant collection per pipeline, plus the Parent-Child docstore.

`.index/manifest.json` records the PDF hashes, embedding model and pipeline settings. If
anything changes, the index is considered stale and is rebuilt. The manifest is written
last, so an interrupted build is also treated as stale.
"""

from __future__ import annotations

import json
import shutil
import threading
from pathlib import Path
from typing import Callable

from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from rag_chatbot.ingestion.fingerprint import corpus_fingerprint
from rag_chatbot.ingestion.loader import list_pdfs, load_pdfs
from rag_chatbot.pipelines.base import IndexContext, PipelineSpec

MANIFEST_VERSION = 1
EMBEDDING_DIMS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}

Progress = Callable[[str], None]


class IndexManager:
    def __init__(
        self,
        *,
        client: QdrantClient,
        embeddings: Embeddings,
        embedding_model: str,
        data_dir: Path,
        index_dir: Path,
        specs: dict[str, PipelineSpec],
        embedding_dim: int | None = None,
    ) -> None:
        self.client = client
        self.embeddings = embeddings
        self.embedding_model = embedding_model
        self.data_dir = data_dir
        self.index_dir = index_dir
        self.specs = specs
        self.embedding_dim = embedding_dim or EMBEDDING_DIMS.get(embedding_model)
        if self.embedding_dim is None:
            raise ValueError(f"Unknown vector size for embedding model '{embedding_model}'; add it to EMBEDDING_DIMS")
        self.search_lock = threading.Lock()
        self._contexts: dict[str, IndexContext] = {}

    @property
    def manifest_path(self) -> Path:
        return self.index_dir / "manifest.json"

    def docstore_dir(self, name: str) -> Path:
        return self.index_dir / "docstores" / name

    def expected_manifest(self) -> dict:
        return {
            "version": MANIFEST_VERSION,
            "embedding_model": self.embedding_model,
            "pdfs": corpus_fingerprint(self.data_dir).files,
            "pipelines": {name: spec.settings for name, spec in self.specs.items()},
        }

    def _read_manifest(self) -> dict | None:
        try:
            return json.loads(self.manifest_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    def is_current(self) -> bool:
        manifest = self._read_manifest()
        if manifest is None:
            return False
        expected = self.expected_manifest()
        stored = {k: manifest.get(k) for k in expected}
        return stored == expected and all(self.client.collection_exists(n) for n in self.specs)

    def counts(self) -> dict[str, int | None]:
        return {
            name: self.client.count(name).count if self.client.collection_exists(name) else None for name in self.specs
        }

    def context(self, name: str) -> IndexContext:
        if name not in self._contexts:
            self._contexts[name] = IndexContext(
                vector_store=QdrantVectorStore(client=self.client, collection_name=name, embedding=self.embeddings),
                embeddings=self.embeddings,
                docstore_dir=self.docstore_dir(name),
                search_lock=self.search_lock,
            )
        return self._contexts[name]

    def build(self, progress: Progress = lambda _msg: None) -> dict[str, int]:
        if not list_pdfs(self.data_dir):
            raise FileNotFoundError(f"No PDFs found in {self.data_dir}")
        self.manifest_path.unlink(missing_ok=True)

        progress(f"Loading PDFs from {self.data_dir}")
        docs = load_pdfs(self.data_dir)

        counts: dict[str, int] = {}
        for name, spec in self.specs.items():
            progress(f"Indexing {spec.label}…")
            with self.search_lock:
                if self.client.collection_exists(name):
                    self.client.delete_collection(name)
                self.client.create_collection(
                    collection_name=name,
                    vectors_config=VectorParams(size=self.embedding_dim, distance=Distance.COSINE),
                )
                shutil.rmtree(self.docstore_dir(name), ignore_errors=True)
                self.docstore_dir(name).mkdir(parents=True, exist_ok=True)
                counts[name] = spec.index(docs, self.context(name))
            progress(f"{spec.label}: {counts[name]:,} vectors")

        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(json.dumps({**self.expected_manifest(), "counts": counts}, indent=2))
        return counts

    def ensure(self, progress: Progress = lambda _msg: None, *, force: bool = False) -> bool:
        """Build the index if it's missing or stale. Returns True if a build ran."""
        if not force and self.is_current():
            return False
        self.build(progress)
        return True
