from conftest import vendor
from fakes import ScriptedClient, quoting

from suppmatch import evaluate as ev
from suppmatch.baselines import embed_only, exact, fuzzy
from suppmatch.candidates import build_candidates
from suppmatch.decide import MATCH, NEEDS_HUMAN, NEW, TAX_CONFLICT, Decision
from suppmatch.judge import Judge, ResponseCache


def test_lookalike_with_tax_different_never_match_crafted(fake_embedder):
    sap = vendor("S", "Apex Tools Inc", tax="111")
    fusion = {1: vendor(1, "Apex Tools Inc", tax="222")}  # same name, address; other company
    assert exact(sap, fusion).outcome == NEEDS_HUMAN
    assert TAX_CONFLICT in fuzzy(sap, fusion).flags
    cands = build_candidates({"S": sap}, fusion, fake_embedder)["S"]
    for t_low, t_high in [(0.0, 0.5), (0.6, 0.9), (0.9, 0.99)]:
        assert embed_only(cands, t_low, t_high).outcome != MATCH


def test_planted_never_match_rows_never_match_in_any_method(built, records, fake_embedder, tmp_path):
    plan = built["plan"]
    never = sorted(k for k, v in plan["vendors"].items() if MATCH not in v["allowed_outcomes"])
    lookalikes = [k for k in never if plan["vendors"][k]["group"] == "LOOKALIKE"]
    assert len(lookalikes) == 15
    for method in ev.METHODS:
        for t_low, t_high in [(0.0, 0.5), (0.6, 0.9)]:
            # the model always says SAME (with valid quotes): Python must still block every MATCH
            judge = Judge(ScriptedClient(quoting("SAME")), ResponseCache(tmp_path, use_cache=False))
            decisions = ev.run_method(method, records, never, fake_embedder, t_low, t_high, judge)
            assert not [k for k, d in decisions.items() if d.outcome == MATCH], method


def test_labels_come_from_plan_only(built, records):
    """Changing the plan changes the score; the decisions themselves do not look at the plan."""
    plan = built["plan"]
    rows = ev.select_rows(plan, "dev")
    decisions = ev.run_method("exact", records, rows, None, 0, 0)
    m1 = ev.score(decisions, plan, records)
    flipped = {"vendors": {k: {**v, "allowed_outcomes": ["NEW"], "expected_outcome": "NEW",
                               "expected_fusion_vendor_id": None} for k, v in plan["vendors"].items()}}
    m2 = ev.score(decisions, flipped, records)
    assert m1.matches_made == m2.matches_made
    assert len(m2.wrong_merges) == m2.matches_made > 0
    assert len(m1.wrong_merges) == 0


def test_evaluate_writes_report(built, records, fake_embedder, tmp_path):
    m, path = ev.evaluate("embed", "dev", records, built["plan"], fake_embedder, "fake", limit=10,
                          reports_dir=tmp_path, data_dir=tmp_path)
    assert m.n == 10
    assert path == tmp_path / "eval_embed_dev.md"
    text = path.read_text(encoding="utf-8")
    assert "Wrong merges" in text and "placeholder" in text


def test_flag_on_allowed_new_counts_as_caught(built, records):
    plan = built["plan"]
    k = next(k for k, v in plan["vendors"].items()
             if v["expected_outcome"] == NEEDS_HUMAN and v["expected_flags"] == [TAX_CONFLICT]
             and NEW in v["allowed_outcomes"])
    one = {"vendors": {k: plan["vendors"][k]}}
    with_flag = ev.score({k: Decision(NEW, "JUDGE_DIFFERENT", None, [TAX_CONFLICT])}, one, records)
    assert (with_flag.flag_caught, with_flag.flag_rows, with_flag.flag_caught_by[TAX_CONFLICT]) == (1, 1, 1)
    without = ev.score({k: Decision(NEW, "SURE_NEW")}, one, records)
    assert (without.flag_caught, without.flag_caught_by[TAX_CONFLICT]) == (0, 0)
    # NEW where the plan does not allow NEW is never "caught", flag or not
    strict = {"vendors": {k: {**plan["vendors"][k], "allowed_outcomes": [NEEDS_HUMAN]}}}
    assert ev.score({k: Decision(NEW, "JUDGE_DIFFERENT", None, [TAX_CONFLICT])}, strict, records).flag_caught == 0


def test_split_selection_and_limit(built):
    plan = built["plan"]
    for split in ev.SPLITS:
        rows = ev.select_rows(plan, split)
        assert rows and all(plan["vendors"][k]["split"] == split for k in rows)
    assert len(ev.select_rows(plan, "heldout", 5)) == 5


def test_routed_and_sonnet_only_reports(built, records, fake_embedder, tmp_path):
    out = {}
    for method in ("routed", "sonnet-only"):
        client = ScriptedClient(quoting("UNSURE"))
        judge = Judge(client, ResponseCache(tmp_path / method))
        m, path = ev.evaluate(method, "dev", records, built["plan"], fake_embedder, "fake",
                              reports_dir=tmp_path, data_dir=tmp_path, judge=judge)
        text = path.read_text(encoding="utf-8")
        assert "Routing share" in text and "Model calls per tier" in text and "per SAP vendor" in text
        assert sum(m.routing.values()) == m.n and m.routing["haiku"] == m.routing["sonnet"] == 0
        assert m.cost > 0 and m.cost is not None
        out[method] = (m, client)
    routed, sonnet = out["routed"][0], out["sonnet-only"][0]
    assert routed.calls["haiku"] == routed.calls["sonnet"] == sonnet.calls["sonnet"] > 0
    assert sonnet.calls["haiku"] == 0
    assert routed.routing["human"] == sonnet.routing["human"]
