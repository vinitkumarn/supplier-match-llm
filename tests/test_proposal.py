"""The merge proposal: no tax IDs or bank numbers, two separate dollar amounts, honest totals."""

import csv
from decimal import Decimal

from fakes import ScriptedClient, quoting
from test_prompt_safety import leaks, secrets

from suppmatch import evaluate as ev
from suppmatch.decide import MATCH, NEEDS_HUMAN, NEW, TAX_CONFLICT
from suppmatch.judge import Judge, ResponseCache
from suppmatch.proposal import CALL_BACK, build_rows, totals, write_proposal


def _routed_rows(built, records, fake_embedder, tmp_path, split=None):
    """Routed run with a model that always says SAME and a wide middle band (most rows reach it)."""
    plan = built["plan"]
    splits = [split] if split else ev.SPLITS
    lifnrs = [k for s in splits for k in ev.select_rows(plan, s)]
    judge = Judge(ScriptedClient(quoting("SAME")), ResponseCache(tmp_path / "resp", use_cache=False))
    decisions = ev.run_method("routed", records, lifnrs, fake_embedder, 0.0, 0.99, judge)
    return build_rows(decisions, records)


def test_proposal_never_contains_tax_id_or_bank_number(built, records, fake_embedder, tmp_path):
    values = secrets(built["tables"])
    rows = _routed_rows(built, records, fake_embedder, tmp_path)
    assert {MATCH, NEEDS_HUMAN} <= {r.outcome for r in rows}   # NEW rows: the baselines below
    csv_path, md_path = write_proposal(rows, "routed", "all", tmp_path)
    texts = [csv_path.read_text(encoding="utf-8"), md_path.read_text(encoding="utf-8")]
    assert leaks(texts + [f"x {sorted(values)[0]} y"], values)  # the scan does catch a leak
    assert leaks(texts, values) == []
    # also the exact / fuzzy / embed proposals
    for method in ("exact", "fuzzy", "embed"):
        decisions = ev.run_method(method, records, sorted(records.sap), fake_embedder, 0.6, 0.9)
        prows = build_rows(decisions, records)
        assert {r.outcome for r in prows} == {MATCH, NEW, NEEDS_HUMAN}
        c, m = write_proposal(prows, method, "all", tmp_path)
        assert leaks([c.read_text(encoding="utf-8"), m.read_text(encoding="utf-8")], values) == []


def test_totals_equal_row_sums_for_each_amount(built, records, fake_embedder, tmp_path):
    rows = _routed_rows(built, records, fake_embedder, tmp_path, "dev")
    t = totals(rows)
    exposed = sum((r.dollars_exposed for r in rows if r.dollars_exposed is not None), Decimal("0"))
    review = sum((r.dollars_under_review for r in rows if r.dollars_under_review is not None), Decimal("0"))
    assert isinstance(t["dollars_exposed"], Decimal) and isinstance(t["dollars_under_review"], Decimal)
    assert t["dollars_exposed"] == exposed > 0
    assert t["dollars_under_review"] == review > 0
    assert t[MATCH] + t[NEW] + t[NEEDS_HUMAN] == t["rows"] == len(rows)

    for r in rows:
        assert not (r.dollars_exposed is not None and r.dollars_under_review is not None)  # never both
        if r.outcome == MATCH:
            assert r.dollars_exposed == r.sap_open + r.fusion_open and r.dollars_under_review is None
            assert r.survivor.startswith(f"Fusion {r.fusion_vendor_id}")
        elif r.outcome == NEEDS_HUMAN and r.fusion_vendor_id is not None:
            assert r.dollars_under_review == r.sap_open + r.fusion_open and r.dollars_exposed is None
        else:
            assert r.dollars_exposed is None and r.dollars_under_review is None and r.survivor == ""

    # the CSV TOTAL row carries the same two numbers, separately
    csv_path, md_path = write_proposal(rows, "routed", "dev", tmp_path)
    with csv_path.open(encoding="utf-8", newline="") as fh:
        data = list(csv.DictReader(fh))
    total = data[-1]
    assert total["sap_lifnr"] == "TOTAL"
    assert Decimal(total["dollars_exposed"]) == sum(Decimal(r["dollars_exposed"]) for r in data[:-1] if r["dollars_exposed"])
    assert Decimal(total["dollars_under_review"]) == sum(
        Decimal(r["dollars_under_review"]) for r in data[:-1] if r["dollars_under_review"])
    assert "never added together" in md_path.read_text(encoding="utf-8")


def test_tax_conflict_lookalike_is_under_review_not_exposed(built, records, fake_embedder, tmp_path):
    plan = built["plan"]
    rows = {r.sap_lifnr: r for r in _routed_rows(built, records, fake_embedder, tmp_path)}
    lookalikes = [k for k, v in plan["vendors"].items()
                  if v["group"] == "LOOKALIKE" and TAX_CONFLICT in v["expected_flags"]]
    hit = [rows[k] for k in lookalikes if rows[k].outcome == NEEDS_HUMAN and TAX_CONFLICT in rows[k].flags]
    assert hit, "the model said SAME, so Python must have blocked it as a TAX_CONFLICT"
    for r in hit:
        assert r.dollars_exposed is None
        assert r.dollars_under_review == records.pair_open(r.sap_lifnr, r.fusion_vendor_id)
        assert r.survivor == ""
    t = totals(list(rows.values()))
    assert t["dollars_exposed"] == sum(r.dollars_exposed for r in rows.values() if r.outcome == MATCH)


def test_bank_note_never_copies_sap_bank(built, records, fake_embedder, tmp_path):
    plan = built["plan"]
    rows = {r.sap_lifnr: r for r in _routed_rows(built, records, fake_embedder, tmp_path)}
    bank_conflicts = [k for k, v in plan["vendors"].items() if v["group"] == "BANKCONFLICT"]
    assert bank_conflicts
    for k in bank_conflicts:
        assert CALL_BACK in rows[k].bank_note and "not copied" in rows[k].bank_note
    for r in rows.values():
        if r.outcome == MATCH:
            assert "same bank already on Fusion" in r.bank_note or CALL_BACK in r.bank_note \
                or "no SAP bank" in r.bank_note


def test_markdown_escapes_untrusted_text(built, records, fake_embedder, tmp_path):
    rows = _routed_rows(built, records, fake_embedder, tmp_path, "redteam")
    _, md_path = write_proposal(rows, "routed", "redteam", tmp_path)
    text = md_path.read_text(encoding="utf-8")
    assert "</UNTRUSTED>" not in text and "<UNTRUSTED>" not in text
    table = [l for l in text.splitlines() if l.startswith("| 0")]
    assert len(table) == len(rows)
    assert all(l.replace("\\|", "").count("|") == len(rows[0].cells()) + 1 for l in table)
