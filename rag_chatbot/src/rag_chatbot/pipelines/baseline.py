"""Baseline pipeline: small fixed-size chunks and plain top-k similarity search."""

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langgraph.graph import START, StateGraph

from rag_chatbot.pipelines.base import IndexContext, PipelineSpec, State, make_generate
from rag_chatbot.ports.llm import LLMPort

CHUNK_SIZE = 50
CHUNK_OVERLAP = 0
K = 3


def _splitter() -> RecursiveCharacterTextSplitter:
    # add_start_index only adds metadata (for page citations); chunk boundaries are unchanged.
    return RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, add_start_index=True)


def index(docs: list[Document], ctx: IndexContext) -> int:
    split_documents = _splitter().split_documents(docs)
    _ = ctx.vector_store.add_documents(documents=split_documents)
    return len(split_documents)


def build_graph(ctx: IndexContext, llm: LLMPort, **_):
    retriever = ctx.vector_store.as_retriever(search_kwargs={"k": K})

    def retrieve(state):
        with ctx.search_lock:
            retrieved_docs = retriever.invoke(state["question"])
        return {"context": retrieved_docs}

    graph_builder = StateGraph(State).add_sequence([retrieve, make_generate(llm)])
    graph_builder.add_edge(START, "retrieve")
    return graph_builder.compile()


SPEC = PipelineSpec(
    name="baseline",
    label="Baseline",
    description=f"{CHUNK_SIZE}-char chunks, overlap {CHUNK_OVERLAP}, top-{K} similarity",
    settings={"chunk_size": CHUNK_SIZE, "chunk_overlap": CHUNK_OVERLAP, "k": K},
    index=index,
    build_graph=build_graph,
)
