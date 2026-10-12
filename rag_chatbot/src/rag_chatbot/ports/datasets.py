from typing import Protocol, runtime_checkable

from rag_chatbot.domain.datasets import Testset, TestsetSummary


@runtime_checkable
class DatasetRepository(Protocol):
    """Outbound port for saving and loading test sets.

    Test sets are immutable once saved. Every adapter must pass
    tests/test_dataset_repository_contract.py.
    """

    def save(self, testset: Testset, knowledge_graph: bytes | None = None) -> None:
        """Raises TestsetAlreadyExists if the id is taken, ValueError if the test set is inconsistent."""
        ...

    def get(self, testset_id: str) -> Testset:
        """Raises TestsetNotFound."""
        ...

    def list(self) -> list[TestsetSummary]:
        """Newest first. Does not load samples."""
        ...

    def next_version(self, name: str) -> int: ...

    def latest(self, name: str) -> TestsetSummary | None: ...

    def load_knowledge_graph(self, testset_id: str) -> bytes | None:
        """The opaque bytes given to save(), or None if none were saved. Raises TestsetNotFound."""
        ...
