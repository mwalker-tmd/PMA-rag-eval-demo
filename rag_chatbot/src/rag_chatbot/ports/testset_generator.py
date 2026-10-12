from typing import Callable, Protocol, runtime_checkable

from rag_chatbot.domain.datasets import GenerationProgress, GenerationResult, GeneratorInfo, SourceDocument

ProgressCallback = Callable[[GenerationProgress], None]


@runtime_checkable
class TestsetGeneratorPort(Protocol):
    """Outbound port that turns a corpus into synthetic test samples.

    Tests should import this module rather than the class name: pytest would otherwise try to
    collect a module-level name starting with "Test" (and a Protocol can't carry `__test__ = False`
    without making it a required member).
    """

    @property
    def info(self) -> GeneratorInfo: ...

    def preflight(self) -> None:
        """Cheap check that the configured models work for generation. Raises ModelIncompatible."""
        ...

    def generate(
        self,
        documents: list[SourceDocument],
        size: int,
        *,
        knowledge_graph: bytes | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> GenerationResult:
        """Generate up to `size` samples; never more. Passing a saved `knowledge_graph` skips building one.

        May return fewer than `size` when the corpus can't support more distinct questions.
        Raises GenerationError (or ModelIncompatible).
        """
        ...
