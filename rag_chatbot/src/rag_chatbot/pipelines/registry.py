from rag_chatbot.pipelines import baseline, parent_child, rerank
from rag_chatbot.pipelines.base import PipelineSpec

PIPELINES: dict[str, PipelineSpec] = {spec.name: spec for spec in (baseline.SPEC, rerank.SPEC, parent_child.SPEC)}
