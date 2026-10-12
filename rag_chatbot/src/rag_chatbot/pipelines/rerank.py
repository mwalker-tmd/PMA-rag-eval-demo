"""Rerank pipeline: wide top-k similarity search, narrowed by a Cohere reranker."""

from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_cohere import CohereRerank
from langchain_core.documents import Document
from langgraph.graph import START, StateGraph

from rag_chatbot.pipelines.base import IndexContext, PipelineSpec, State, make_generate
from rag_chatbot.ports.llm import LLMPort

CHUNK_SIZE = 500
CHUNK_OVERLAP = 30
K = 20
RERANK_MODEL = "rerank-v3.5"


def _splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, add_start_index=True)


def index(docs: list[Document], ctx: IndexContext) -> int:
    split_documents = _splitter().split_documents(docs)
    _ = ctx.vector_store.add_documents(documents=split_documents)
    return len(split_documents)


def build_graph(ctx: IndexContext, llm: LLMPort, *, cohere_api_key: str | None = None, **_):
    adjusted_example_retriever = ctx.vector_store.as_retriever(search_kwargs={"k": K})
    compressor = CohereRerank(model=RERANK_MODEL, cohere_api_key=cohere_api_key)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=adjusted_example_retriever, search_kwargs={"k": 5}
    )

    def retrieve_adjusted(state):
        with ctx.search_lock:
            retrieved_docs = compression_retriever.invoke(state["question"])
        return {"context": retrieved_docs}

    adjusted_graph_builder = StateGraph(State).add_sequence([retrieve_adjusted, make_generate(llm)])
    adjusted_graph_builder.add_edge(START, "retrieve_adjusted")
    return adjusted_graph_builder.compile()


SPEC = PipelineSpec(
    name="rerank",
    label="Rerank",
    description=f"{CHUNK_SIZE}/{CHUNK_OVERLAP} chunks, top-{K} similarity → Cohere {RERANK_MODEL}",
    settings={"chunk_size": CHUNK_SIZE, "chunk_overlap": CHUNK_OVERLAP, "k": K},
    index=index,
    build_graph=build_graph,
    requires_cohere=True,
)
