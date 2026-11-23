import numpy as np

from suppmatch.candidates import build_candidates
from suppmatch.embed import CachedEmbedder, FakeEmbedder, cache_key


def test_fake_embedder_deterministic_and_normalized():
    a = FakeEmbedder().embed(["APEX TOOLS | 1 MAIN ST | SPRINGFIELD | US"])
    b = FakeEmbedder().embed(["APEX TOOLS | 1 MAIN ST | SPRINGFIELD | US"])
    assert np.array_equal(a, b)
    assert abs(float(np.linalg.norm(a[0])) - 1.0) < 1e-5


def test_fake_embedder_similar_strings_score_close():
    v = FakeEmbedder().embed(["APEX INDUSTRIES", "APEKS INDUSTRIES", "BLUEWATER FOODS"])
    assert v[0] @ v[1] > 0.7
    assert v[0] @ v[1] > v[0] @ v[2] + 0.3


def test_cache_key_includes_model_name():
    assert cache_key("model-a", "X") != cache_key("model-b", "X")
    assert cache_key("model-a", "X") == cache_key("model-a", "X")


def test_same_text_embedded_once_in_a_run(tmp_path):
    e = CachedEmbedder(FakeEmbedder(), tmp_path)
    e.embed(["A", "B", "A"])
    e.embed(["B", "A"])
    assert e.computed == 2


def test_second_run_computes_zero(tmp_path, records):
    first = CachedEmbedder(FakeEmbedder(), tmp_path)
    c1 = build_candidates(records.sap, records.fusion, first)
    assert first.computed > 0

    second = CachedEmbedder(FakeEmbedder(), tmp_path)  # fresh process, same cache folder
    c2 = build_candidates(records.sap, records.fusion, second)
    assert second.computed == 0
    assert second.hits >= first.computed
    assert [(c.vendor_id, c.full_cosine) for c in c1["0000100001"]] == \
           [(c.vendor_id, c.full_cosine) for c in c2["0000100001"]]


def test_no_cache_writes_nothing(tmp_path):
    e = CachedEmbedder(FakeEmbedder(), tmp_path / "c", use_cache=False)
    e.embed(["A", "B"])
    assert not (tmp_path / "c").exists()
