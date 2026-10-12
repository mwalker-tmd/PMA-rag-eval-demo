"""Live checks of the Ragas judge's behavior. RUN_LIVE_TESTS=1 uv run pytest -m live"""

import os
from dataclasses import replace

import pytest
from factories import make_eval_samples

from rag_chatbot.config import load_settings
from rag_chatbot.container import build_evaluator
from rag_chatbot.domain.evals import DEFAULT_METRICS

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(os.getenv("RUN_LIVE_TESTS") != "1", reason="set RUN_LIVE_TESTS=1 to call the real API"),
]


def test_all_six_metrics_score_and_a_grounded_answer_beats_a_made_up_one():
    evaluator = build_evaluator(replace(load_settings(), eval_provider="ragas"))
    grounded, made_up = make_eval_samples()
    result = evaluator.evaluate([grounded, made_up])
    print(f"\ngrounded: {result.scores[0]}\nmade up:  {result.scores[1]}\nusage: {result.usage}")

    assert list(result.scores[0]) == list(DEFAULT_METRICS)
    scored = [m for m in DEFAULT_METRICS if result.scores[0][m] is not None]
    assert scored == list(DEFAULT_METRICS), f"metrics without a score: {set(DEFAULT_METRICS) - set(scored)}"
    assert result.scores[0]["faithfulness"] > result.scores[1]["faithfulness"]
    assert result.usage is not None and result.usage.total_tokens > 0
