"""Contract every DatasetRepository adapter must satisfy.

To add an adapter, add it to the `repo` fixture's params; every test below then runs against it.
"""

from dataclasses import replace
from datetime import timedelta, timezone

import pytest
from factories import T0, make_testset

from rag_chatbot.adapters.datasets.file_repository import FileDatasetRepository
from rag_chatbot.adapters.datasets.memory_repository import InMemoryDatasetRepository
from rag_chatbot.domain.datasets import TestsetAlreadyExists, TestsetNotFound
from rag_chatbot.ports.datasets import DatasetRepository


@pytest.fixture(params=["memory", "file"])
def repo(request, tmp_path) -> DatasetRepository:
    if request.param == "memory":
        return InMemoryDatasetRepository()
    return FileDatasetRepository(tmp_path / ".datasets")


def test_satisfies_protocol(repo):
    assert isinstance(repo, DatasetRepository)


def test_save_then_get_round_trips(repo):
    ts = make_testset()
    repo.save(ts)
    assert repo.get(ts.manifest.id) == ts


def test_saving_an_existing_id_raises_and_keeps_the_original(repo):
    ts = make_testset()
    repo.save(ts)
    clash = replace(ts, manifest=replace(ts.manifest, name="something-else"))
    with pytest.raises(TestsetAlreadyExists):
        repo.save(clash)
    assert repo.get(ts.manifest.id) == ts


def test_unknown_id_raises_not_found(repo):
    with pytest.raises(TestsetNotFound):
        repo.get("ts_20260101_deadbeef")
    with pytest.raises(TestsetNotFound):
        repo.load_knowledge_graph("ts_20260101_deadbeef")


def test_knowledge_graph_round_trips_as_bytes(repo):
    with_kg, without_kg = make_testset(with_kg=True), make_testset(version=2)
    kg = b'{"nodes": [], "relationships": [], "note": "caf\xc3\xa9"}'
    repo.save(with_kg, kg)
    repo.save(without_kg)
    assert repo.load_knowledge_graph(with_kg.manifest.id) == kg
    assert repo.load_knowledge_graph(without_kg.manifest.id) is None


def test_list_is_newest_first_and_has_no_samples(repo):
    old = make_testset("a", created_at=T0)
    new = make_testset("b", created_at=T0 + timedelta(hours=1))
    repo.save(old)
    repo.save(new)
    summaries = repo.list()
    assert [s.id for s in summaries] == [new.manifest.id, old.manifest.id]
    assert not hasattr(summaries[0], "samples")
    assert summaries[0].size == 3
    assert summaries[0].corpus_digest == new.manifest.corpus.digest


def test_versions_per_name(repo):
    assert repo.next_version("core-corpus") == 1
    assert repo.latest("core-corpus") is None

    v1 = make_testset(version=1)
    repo.save(v1)
    v2 = make_testset(version=2, created_at=T0 + timedelta(minutes=5), parent_id=v1.manifest.id)
    repo.save(v2)
    repo.save(make_testset("other", version=1))

    assert repo.next_version("core-corpus") == 3
    assert repo.next_version("other") == 2
    assert repo.latest("core-corpus").id == v2.manifest.id
    assert repo.latest("core-corpus").parent_id == v1.manifest.id


def test_identical_samples_in_two_versions_get_distinct_ids(repo):
    v1, v2 = make_testset(version=1), make_testset(version=2)
    assert v1.samples == v2.samples
    repo.save(v1)
    repo.save(v2)  # must not raise TestsetAlreadyExists
    assert v1.manifest.id != v2.manifest.id


@pytest.mark.parametrize(
    "corrupt",
    [
        lambda m: replace(m, size=m.size + 1),
        lambda m: replace(m, samples_sha256="0" * 64),
        lambda m: replace(m, has_knowledge_graph=True),  # but no graph given
        lambda m: replace(m, created_at=m.created_at.replace(tzinfo=None)),
        lambda m: replace(m, created_at=m.created_at.astimezone(timezone(timedelta(hours=5)))),
    ],
    ids=["size", "samples_sha256", "has_knowledge_graph", "naive_datetime", "non_utc_datetime"],
)
def test_inconsistent_test_sets_are_rejected(repo, corrupt):
    ts = make_testset()
    bad = replace(ts, manifest=corrupt(ts.manifest))
    with pytest.raises(ValueError):
        repo.save(bad)
    assert repo.list() == []


def test_mutating_the_saved_object_does_not_change_storage(repo):
    ts = make_testset()
    repo.save(ts)
    ts.samples[0].reference_contexts.append("tampered")
    assert "tampered" not in repo.get(ts.manifest.id).samples[0].reference_contexts
