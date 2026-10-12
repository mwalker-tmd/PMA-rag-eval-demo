import json

from rag_chatbot.ingestion.loader import load_pdfs, page_for


def test_load_pdfs_merges_pages(data_dir):
    docs = load_pdfs(data_dir)
    assert [d.metadata["source"] for d in docs] == ["leadership.pdf", "quality.pdf"]
    quality = docs[1]
    assert len(quality.metadata["page_starts"]) == 2
    assert "Quality is a habit" in quality.page_content and "Testing pyramids" in quality.page_content


def test_page_for():
    meta = {"page_starts": [0, 100, 250]}
    assert page_for({**meta, "start_index": 0}) == 1
    assert page_for({**meta, "start_index": 99}) == 1
    assert page_for({**meta, "start_index": 100}) == 2
    assert page_for({**meta, "start_index": 400}) == 3
    assert page_for({"start_index": 5}) is None


def test_build_writes_all_collections_and_manifest(container):
    assert not container.index.is_current()
    counts = container.index.build()
    assert set(counts) == {"baseline", "rerank", "parent_child"}
    assert all(c > 0 for c in counts.values())
    # 50-char chunks should produce far more vectors than 500-char chunks
    assert counts["baseline"] > counts["rerank"]
    assert container.index.is_current()
    manifest = json.loads(container.index.manifest_path.read_text())
    assert manifest["pipelines"]["baseline"] == {"chunk_size": 50, "chunk_overlap": 0, "k": 3}


def test_index_goes_stale_when_a_pdf_changes(container, data_dir):
    container.index.build()
    (data_dir / "quality.pdf").write_bytes((data_dir / "leadership.pdf").read_bytes())
    assert not container.index.is_current()
    assert container.index.ensure() is True
    assert container.index.is_current()
    assert container.index.ensure() is False
