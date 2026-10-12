"""Load PDFs as one LangChain Document per file.

Merging each PDF's pages keeps chunks from being cut at page boundaries. `page_starts`
records where each page begins in the merged text so a chunk's `start_index` can be
mapped back to a page number.
"""

from __future__ import annotations

from bisect import bisect_right
from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document

PAGE_SEPARATOR = "\n\n"


def list_pdfs(data_dir: Path) -> list[Path]:
    return sorted(p for p in data_dir.iterdir() if p.suffix.lower() == ".pdf")


def load_pdf(path: Path) -> Document:
    pages = PyMuPDFLoader(str(path)).load()
    texts, page_starts, offset = [], [], 0
    for page in pages:
        page_starts.append(offset)
        texts.append(page.page_content)
        offset += len(page.page_content) + len(PAGE_SEPARATOR)
    return Document(
        page_content=PAGE_SEPARATOR.join(texts),
        metadata={"source": path.name, "page_starts": page_starts},
    )


def load_pdfs(data_dir: Path) -> list[Document]:
    return [load_pdf(p) for p in list_pdfs(data_dir)]


def page_for(metadata: dict) -> int | None:
    """1-based page number for a chunk split with add_start_index=True, else None."""
    starts, start_index = metadata.get("page_starts"), metadata.get("start_index")
    if not starts or start_index is None:
        return None
    return bisect_right(starts, start_index)
