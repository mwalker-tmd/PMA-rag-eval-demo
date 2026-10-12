"""Live tests for RagasTestsetGenerator. They call OpenAI and cost a few cents: RUN_LIVE_TESTS=1 uv run pytest -m live"""

import os
import threading

import pytest

from rag_chatbot.config import load_settings
from rag_chatbot.container import build_testset_generator
from rag_chatbot.domain.datasets import SourceDocument

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(os.getenv("RUN_LIVE_TESTS") != "1", reason="set RUN_LIVE_TESTS=1 to call the real API"),
]

# Ragas rejects documents of 100 tokens or fewer, and multi-hop questions need entities shared across documents.
DOC_A = """\
Northwind Labs runs every release through a three-stage quality process. First, engineers pair on
code review within a day of opening a pull request, because Northwind's data shows that defects found
in review cost about a tenth as much to fix as defects found in production. Second, the platform team
maintains a testing pyramid: thousands of fast unit tests, a few hundred integration tests against
real databases, and a small set of end-to-end tests that exercise the checkout flow. Third, a release
captain, a rotating role held by a senior engineer for one week, signs off only when error budgets
are healthy. Priya Raman, Northwind's head of quality, introduced the release captain role in 2023
after two outages traced back to rushed Friday deployments. Since then, change failure rate has
dropped from 18 percent to 6 percent. Priya argues that quality is a habit built by many small,
repeated practices rather than a phase at the end of a project, and she measures it with four
indicators: deployment frequency, lead time for changes, change failure rate and time to restore.
"""

DOC_B = """\
Northwind Labs treats engineering leadership as the work of setting direction, removing blockers and
growing people. Engineering managers hold weekly one-on-ones and spend at least a third of their time
on coaching. Priya Raman, who leads quality, partners with the engineering managers to make the
release captain role a growth opportunity: engineers who serve as release captain learn incident
response, risk assessment and stakeholder communication. Leadership reviews the four delivery
indicators, deployment frequency, lead time for changes, change failure rate and time to restore, in a
monthly operations review, but managers are told never to use the indicators to rank individuals.
Instead, the indicators guide investment: when lead time rose in early 2024, leadership funded a
dedicated platform team to speed up the integration test suite. Northwind's leadership principles say
that a team's long-term velocity depends on psychological safety, clear ownership and a sustainable
pace, and that heroics during outages are a symptom of missing process rather than something to reward.
"""

DOCS = [
    SourceDocument(DOC_A, {"source": "northwind-quality.pdf"}),
    SourceDocument(DOC_B, {"source": "northwind-leadership.pdf"}),
]


@pytest.fixture(scope="module")
def generator():
    settings = load_settings()
    return build_testset_generator(settings.__class__(**{**settings.__dict__, "sdg_provider": "ragas"}))


def test_preflight_passes_for_the_default_model(generator):
    generator.preflight()


def test_generates_samples_and_reuses_the_graph_on_a_worker_thread(generator):
    first = generator.generate(DOCS, 2)
    assert len(first.samples) >= 2
    assert all(s.user_input and s.reference for s in first.samples)
    assert first.knowledge_graph
    assert first.usage is None or first.usage.total_tokens > 0

    # Reuse the saved graph from a background thread, as the job runner will.
    out: dict = {}

    def worker():
        try:
            out["result"] = generator.generate(DOCS, 2, knowledge_graph=first.knowledge_graph)
        except BaseException as e:  # surface in the main thread
            out["error"] = e

    t = threading.Thread(target=worker)
    t.start()
    t.join(timeout=600)
    assert "error" not in out, out.get("error")
    second = out["result"]
    assert len(second.samples) >= 2
    assert second.knowledge_graph == first.knowledge_graph
    print(f"\nfirst usage: {first.usage}\nsecond usage: {second.usage}")
    print("sample:", first.samples[0])
