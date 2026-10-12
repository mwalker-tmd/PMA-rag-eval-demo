"""Parent-Child pipeline.

Search small child chunks for precision, then return the larger parent documents so the
generator has enough surrounding context. Parents live in a file-backed docstore so
they persist alongside the Qdrant index.
"""

from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import LocalFileStore, create_kv_docstore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langgraph.graph import START, StateGraph

from rag_chatbot.pipelines.base import IndexContext, PipelineSpec, State, make_generate
from rag_chatbot.ports.llm import LLMPort

PARENT_CHUNK_SIZE = 1000
PARENT_CHUNK_OVERLAP = 100
CHILD_CHUNK_SIZE = 200
CHILD_CHUNK_OVERLAP = 40
K = 8


def _retriever(ctx: IndexContext) -> ParentDocumentRetriever:
    # Parents carry start_index for page citations; children inherit it from their parent.
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=PARENT_CHUNK_SIZE, chunk_overlap=PARENT_CHUNK_OVERLAP, add_start_index=True
    )
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=CHILD_CHUNK_SIZE, chunk_overlap=CHILD_CHUNK_OVERLAP)
    return ParentDocumentRetriever(
        vectorstore=ctx.vector_store,
        docstore=create_kv_docstore(LocalFileStore(str(ctx.docstore_dir))),
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
        search_kwargs={"k": K},
    )


def index(docs: list[Document], ctx: IndexContext) -> int:
    _retriever(ctx).add_documents(docs)
    return ctx.vector_store.client.count(ctx.vector_store.collection_name).count


def build_graph(ctx: IndexContext, llm: LLMPort, **_):
    parent_child_retriever = _retriever(ctx)

    def retrieve_custom(state):
        with ctx.search_lock:
            retrieved_docs = parent_child_retriever.invoke(state["question"])
        return {"context": retrieved_docs}

    custom_graph_builder = StateGraph(State).add_sequence([retrieve_custom, make_generate(llm)])
    custom_graph_builder.add_edge(START, "retrieve_custom")
    return custom_graph_builder.compile()


SPEC = PipelineSpec(
    name="parent_child",
    label="Parent-Child",
    description=(
        f"search {CHILD_CHUNK_SIZE}/{CHILD_CHUNK_OVERLAP} child chunks (top-{K}), "
        f"return {PARENT_CHUNK_SIZE}/{PARENT_CHUNK_OVERLAP} parents"
    ),
    settings={
        "parent_chunk_size": PARENT_CHUNK_SIZE,
        "parent_chunk_overlap": PARENT_CHUNK_OVERLAP,
        "child_chunk_size": CHILD_CHUNK_SIZE,
        "child_chunk_overlap": CHILD_CHUNK_OVERLAP,
        "k": K,
    },
    index=index,
    build_graph=build_graph,
)
