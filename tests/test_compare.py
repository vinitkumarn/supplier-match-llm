"""compare reads the eval JSON files (never the md) and shows "not run" for missing methods."""

import json

from suppmatch import evaluate as ev
from suppmatch.compare import NOT_RUN, compare


def _cells(text: str, label: str) -> list[str]:
    line = next(l for l in text.splitlines() if l.startswith(f"| {label} |"))
    return [c.strip() for c in line.strip("|").split("|")][1:]


def test_eval_writes_json_next_to_md(built, records, fake_embedder, tmp_path):
    m, path = ev.evaluate("embed", "dev", records, built["plan"], fake_embedder, "fake", limit=10,
                          reports_dir=tmp_path, data_dir=tmp_path)
    data = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    assert data["n"] == 10 and data["wrong_merges"] == len(m.wrong_merges)
    assert data["dollars_exposed_found"] == str(m.dollars_found)
    assert data["dollars_under_review"] == str(m.dollars_under_review)
    assert set(data["flags"]) == {"FRAUD_FLAG", "TAX_CONFLICT", "BANK_CONFLICT", "AMBIGUOUS"}
    assert data["embedder"] == "fake" and data["embeddings_computed"] is not None


def test_missing_runs_show_not_run(built, records, tmp_path):
    ev.evaluate("exact", "heldout", records, built["plan"], None, "fake", reports_dir=tmp_path, data_dir=tmp_path)
    text = compare("heldout", tmp_path).read_text(encoding="utf-8")
    header = _cells(text, "Metric")
    assert header == list(ev.METHODS)
    wrong = _cells(text, "Wrong merges (target 0)")
    assert wrong[0] == "0"
    assert wrong[1:] == [NOT_RUN] * 4
    # nothing at all for another split -> every cell "not run"
    empty = compare("redteam", tmp_path).read_text(encoding="utf-8")
    assert _cells(empty, "Duplicate recall") == [NOT_RUN] * 5


def test_compare_reads_json_not_md(built, records, tmp_path):
    _, md = ev.evaluate("fuzzy", "dev", records, built["plan"], None, "fake", reports_dir=tmp_path, data_dir=tmp_path)
    js = md.with_suffix(".json")
    data = json.loads(js.read_text(encoding="utf-8"))
    data.update(wrong_merges=77, dollars_under_review="12345.67", dollars_exposed_found="1.00",
                dollars_exposed_planted="2.00")
    js.write_text(json.dumps(data), encoding="utf-8")
    md.write_text("# garbage\n- Wrong merges (target 0): 0\n", encoding="utf-8")
    text = compare("dev", tmp_path).read_text(encoding="utf-8")
    fuzzy_col = ev.METHODS.index("fuzzy")
    assert _cells(text, "Wrong merges (target 0)")[fuzzy_col] == "77"
    assert _cells(text, "Dollars under review (NEEDS_HUMAN)")[fuzzy_col] == "$12,345.67"
    assert _cells(text, "Dollars exposed: found / planted")[fuzzy_col] == "$1.00 / $2.00"
    # an md alone (no JSON) does not count as a run
    js.unlink()
    assert _cells(compare("dev", tmp_path).read_text(encoding="utf-8"), "Wrong merges (target 0)")[fuzzy_col] == NOT_RUN
