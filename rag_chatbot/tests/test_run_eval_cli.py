"""scripts/run_eval.py end to end with fake providers, configured only through env vars."""

import importlib.util
from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from rag_chatbot.container import build_sdg_service
from rag_chatbot.services.sdg_service import GenerateTestsetRequest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_eval.py"


def load_cli():
    spec = importlib.util.spec_from_file_location("run_eval_cli", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main


@pytest.fixture
def cli(monkeypatch, tmp_path, data_dir):
    """Fake everything, isolate every folder, and save one test set. Returns (main, testset_id, tmp_path)."""
    monkeypatch.setattr("rag_chatbot.config.load_dotenv", lambda *_a, **_k: None)
    env = {
        "LLM_PROVIDER": "fake",
        "LLM_MODEL": "fake-model",
        "SDG_PROVIDER": "fake",
        "EVAL_PROVIDER": "fake",
        "EMBEDDING_PROVIDER": "fake",
        "EVAL_JUDGE_MODEL": "gpt-4.1-mini",
        "EVAL_ROW_DELAY_SECONDS": "0",
        "DATA_DIR": str(data_dir),
        "INDEX_DIR": str(tmp_path / "index"),
        "DATASETS_DIR": str(tmp_path / "datasets"),
        "EVALS_DIR": str(tmp_path / "evals"),
    }
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    for key in ("OPENAI_API_KEY", "COHERE_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    testset = build_sdg_service().generate(GenerateTestsetRequest(name="core-corpus", size=2))
    return load_cli(), testset.id, tmp_path


def test_cli_runs_every_pipeline_and_prints_a_comparison_table(cli, capsys):
    main, testset_id, tmp_path = cli
    assert main(["--testset", testset_id]) == 0
    out = capsys.readouterr().out
    table = out[out.index("metric"):]
    header = table.splitlines()[0].split()
    assert header == ["metric", "baseline", "parent_child", "Δ"]  # one column per pipeline (+ delta vs the first)
    assert "faithfulness" in table and "Δ is relative to baseline" in table
    assert "same as the answering model" not in out
    assert len(list((tmp_path / "evals").glob("run_*"))) == 2


def test_cli_prints_the_judge_warning_when_the_models_match(cli, capsys, monkeypatch):
    main, testset_id, _ = cli
    monkeypatch.setenv("EVAL_JUDGE_MODEL", "fake-model")
    assert main(["--testset", testset_id, "--pipelines", "baseline"]) == 0
    assert "Warning: The judge model (fake-model) is the same as the answering model" in capsys.readouterr().out


def test_cli_reference_option_and_unknown_testset(cli, capsys):
    main, testset_id, _ = cli
    assert main(["--testset", testset_id, "--reference", "parent_child"]) == 0
    assert "Δ is relative to parent_child" in capsys.readouterr().out
    assert main(["--testset", "ts_20260101_deadbeef"]) == 1
    assert "ts_20260101_deadbeef" in capsys.readouterr().err


def test_cli_refuses_an_unavailable_pipeline(cli, capsys):
    main, testset_id, _ = cli
    assert main(["--testset", testset_id, "--pipelines", "rerank"]) == 1
    assert "COHERE_API_KEY" in capsys.readouterr().err


def test_cli_explains_when_the_app_holds_the_index(cli, capsys):
    main, testset_id, tmp_path = cli
    (tmp_path / "index").mkdir(exist_ok=True)
    holder = QdrantClient(path=str(tmp_path / "index" / "qdrant"))  # the running app
    try:
        assert main(["--testset", testset_id]) == 1
        assert "already accessed" in capsys.readouterr().err
    finally:
        holder.close()



def test_cli_prints_the_stale_test_set_warning(cli, capsys, data_dir):
    import pymupdf

    main, testset_id, _ = cli
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "Changed corpus.")
    doc.save(str(data_dir / "changed.pdf"))
    assert main(["--testset", testset_id, "--pipelines", "baseline"]) == 0
    out = capsys.readouterr().out
    assert "Warning: Test set core-corpus v1 is stale" in out
    assert out.count("same as the answering model") == 0


def test_cli_reports_a_missing_corpus_instead_of_a_traceback(cli, capsys, data_dir):
    import shutil

    main, testset_id, _ = cli
    shutil.rmtree(data_dir)
    assert main(["--testset", testset_id]) == 1
    assert "No PDFs found" in capsys.readouterr().err


def test_cli_prints_the_judge_warning_once(cli, capsys, monkeypatch):
    main, testset_id, _ = cli
    monkeypatch.setenv("EVAL_JUDGE_MODEL", "fake-model")
    assert main(["--testset", testset_id, "--pipelines", "baseline"]) == 0
    assert capsys.readouterr().out.count("same as the answering model") == 1
