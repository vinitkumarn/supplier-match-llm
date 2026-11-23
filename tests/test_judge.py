import json
from decimal import Decimal

import pytest
from conftest import vendor
from fakes import ScriptedClient, answer, by_model, quoting

from suppmatch import config, judge as judge_mod
from suppmatch.audit import AuditLog
from suppmatch.candidates import DIFFERENT, MISSING, SAME, Candidate, build_candidates
from suppmatch.decide import BANK_CONFLICT, MATCH, NEEDS_HUMAN, NEW, TAX_CONFLICT, decide
from suppmatch.judge import ROUTED, SONNET_ONLY, Judge, ResponseCache, response_cache_key
from suppmatch.packet import build_packet

T_LOW, T_HIGH = 0.6, 0.9
HAIKU, SONNET = config.MODEL_CHEAP, config.MODEL_STRONG


def middle_pair(tax=MISSING, bank=MISSING, sap_name="Apex Tools Inc", fus_name="Apex Tool Inc"):
    sap = vendor("S1", sap_name)
    fus = vendor(1, fus_name)
    return sap, [Candidate(fus, 0.8, 0.75, tax, bank, True)]


def make_judge(responder, tmp_path, use_cache=True, **kw):
    client = ScriptedClient(responder)
    return Judge(client, ResponseCache(tmp_path / "resp", use_cache), **kw), client


# --- routing -----------------------------------------------------------------

def test_haiku_same_does_not_call_sonnet(tmp_path):
    j, client = make_judge(quoting("SAME"), tmp_path)
    d = j.decide(*middle_pair(), T_LOW, T_HIGH)
    assert (d.outcome, d.route, d.rule) == (MATCH, "haiku", "JUDGE_SAME")
    assert client.calls_for(HAIKU) == 1 and client.calls_for(SONNET) == 0


def test_haiku_unsure_calls_sonnet(tmp_path):
    j, client = make_judge(by_model({HAIKU: quoting("UNSURE"), SONNET: quoting("DIFFERENT")}), tmp_path)
    d = j.decide(*middle_pair(), T_LOW, T_HIGH)
    assert (d.outcome, d.route) == (NEW, "sonnet")
    assert client.calls_for(HAIKU) == 1 and client.calls_for(SONNET) == 1


def test_sonnet_unsure_is_needs_human(tmp_path):
    j, client = make_judge(quoting("UNSURE"), tmp_path)
    d = j.decide(*middle_pair(), T_LOW, T_HIGH)
    assert (d.outcome, d.route, d.rule) == (NEEDS_HUMAN, "human", "JUDGE_UNSURE")
    assert len(client.requests) == 2


def test_sonnet_only_skips_haiku(tmp_path):
    j, client = make_judge(quoting("SAME"), tmp_path)
    d = j.decide(*middle_pair(), T_LOW, T_HIGH, SONNET_ONLY)
    assert (d.outcome, d.route) == (MATCH, "sonnet")
    assert client.calls_for(HAIKU) == 0 and client.calls_for(SONNET) == 1


def test_same_prompt_both_tiers_and_request_shape(tmp_path):
    j, client = make_judge(quoting("UNSURE"), tmp_path)
    j.decide(*middle_pair(), T_LOW, T_HIGH)
    a, b = client.requests
    assert a["model"] == HAIKU and b["model"] == SONNET
    assert a["system"] == b["system"] and a["messages"] == b["messages"]
    assert a["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert a["max_tokens"] == config.MAX_TOKENS == 4000
    assert "tools" not in a and "tool_choice" not in a


def test_rule_decided_rows_make_no_calls(built, records, fake_embedder, tmp_path):
    j, client = make_judge(quoting("SAME"), tmp_path)
    cands = build_candidates(records.sap, records.fusion, fake_embedder)
    judged = 0
    for k, s in records.sap.items():
        before = len(client.requests)
        d = j.decide(s, cands[k], T_LOW, T_HIGH)
        if decide(cands[k], T_LOW, T_HIGH).needs_judge:
            judged += 1
            assert d.route != "rules"
        else:
            assert len(client.requests) == before and d.route == "rules" and d.trace is None
    assert 0 < judged < len(records.sap)


# --- Python keeps the hard rules ---------------------------------------------

@pytest.mark.parametrize("tax,bank,flag", [(DIFFERENT, MISSING, TAX_CONFLICT), (MISSING, DIFFERENT, BANK_CONFLICT)])
def test_same_on_conflict_pair_is_needs_human(tmp_path, tax, bank, flag):
    j, _ = make_judge(quoting("SAME"), tmp_path)
    d = j.decide(*middle_pair(tax=tax, bank=bank), T_LOW, T_HIGH)
    assert d.outcome == NEEDS_HUMAN and flag in d.flags


@pytest.mark.parametrize("tax,bank,flag,words", [(DIFFERENT, MISSING, TAX_CONFLICT, "different tax ID"),
                                                 (MISSING, DIFFERENT, BANK_CONFLICT, "different bank account")])
def test_different_on_conflict_pair_is_new_and_keeps_flag(tmp_path, tax, bank, flag, words):
    """Lookalike with a conflict, judge says DIFFERENT: NEW, flag kept as information only."""
    j, _ = make_judge(quoting("DIFFERENT"), tmp_path)
    d = j.decide(*middle_pair(tax=tax, bank=bank), T_LOW, T_HIGH)
    assert (d.outcome, d.rule, d.fusion_vendor_id) == (NEW, "JUDGE_DIFFERENT", None)
    assert d.flags == [flag]
    assert any(f"closest Fusion supplier has a {words}" in r for r in d.reasons)


def test_different_without_conflict_has_no_flag(tmp_path):
    j, _ = make_judge(quoting("DIFFERENT"), tmp_path)
    d = j.decide(*middle_pair(), T_LOW, T_HIGH)
    assert d.outcome == NEW and d.flags == [] and not any("closest Fusion" in r for r in d.reasons)


# --- guards ------------------------------------------------------------------

def _fixed(text):
    return lambda model, user_text, n: text


@pytest.mark.parametrize("ev", [
    [{"record": "SAP", "field": "NAME1", "quote": "Apex"}, {"record": "FUSION", "field": "vendor_name", "quote": "Acme"}],
    [{"record": "SAP", "field": "NAME1", "quote": "Apex"}, {"record": "FUSION", "field": "tax_id", "quote": "Apex"}],
    [{"record": "SAP", "field": "NAME1", "quote": "Apex"}, {"record": "FUSION", "field": "vendor_name", "quote": "  "}],
    [],                                                       # SAME with no evidence
    [{"record": "SAP", "field": "NAME1", "quote": "Apex"}],   # SAME with only SAP quotes
])
def test_guard_failures_are_needs_human_and_not_escalated(tmp_path, ev):
    j, client = make_judge(_fixed(answer("SAME", ev)), tmp_path)
    d = j.decide(*middle_pair(), T_LOW, T_HIGH)
    assert (d.outcome, d.rule, d.route) == (NEEDS_HUMAN, "JUDGE_INVALID", "human")
    assert "guard failed" in " ".join(d.reasons)
    assert client.calls_for(SONNET) == 0
    assert not list((tmp_path / "resp").rglob("*.json"))     # never cached


def test_different_needs_a_quote(tmp_path):
    j, _ = make_judge(_fixed(answer("DIFFERENT", [])), tmp_path)
    assert j.decide(*middle_pair(), T_LOW, T_HIGH).outcome == NEEDS_HUMAN


def test_escaped_quote_is_accepted(tmp_path):
    sap, cands = middle_pair(sap_name="Apex <Tools> Inc")
    ev = [{"record": "SAP", "field": "NAME1", "quote": "Apex &lt;Tools&gt;"},
          {"record": "FUSION", "field": "vendor_name", "quote": "Apex Tool"}]
    j, _ = make_judge(_fixed(answer("SAME", ev)), tmp_path)
    assert j.decide(sap, cands, T_LOW, T_HIGH).outcome == MATCH


# --- schema + retry ------------------------------------------------------------

def test_invalid_then_valid_retries_once_with_full_error(tmp_path):
    good = quoting("SAME")
    j, client = make_judge(lambda m, t, n: answer("SAME", outcome="MATCH") if n == 1 else good(m, t, n), tmp_path)
    d = j.decide(*middle_pair(), T_LOW, T_HIGH)
    assert d.outcome == MATCH and len(client.requests) == 2
    retry = client.requests[1]["messages"]
    assert [m["role"] for m in retry] == ["user", "assistant", "user"]
    assert "outcome" in retry[2]["content"] and "Extra inputs are not permitted" in retry[2]["content"]


def test_invalid_twice_is_needs_human_without_sonnet(tmp_path):
    j, client = make_judge(_fixed("not json"), tmp_path)
    d = j.decide(*middle_pair(), T_LOW, T_HIGH)
    assert (d.outcome, d.rule) == (NEEDS_HUMAN, "JUDGE_INVALID")
    assert client.calls_for(HAIKU) == 2 and client.calls_for(SONNET) == 0


def test_max_tokens_stop_is_invalid_and_counted(tmp_path):
    text = answer("SAME")
    j, client = make_judge(lambda m, t, n: (text, "max_tokens"), tmp_path)
    d = j.decide(*middle_pair(), T_LOW, T_HIGH)
    assert d.outcome == NEEDS_HUMAN
    assert [c.stop_reason for c in d.trace.calls] == ["max_tokens", "max_tokens"]


def test_api_error_is_needs_human(tmp_path):
    j, client = make_judge(lambda m, t, n: RuntimeError("overloaded"), tmp_path)
    d = j.decide(*middle_pair(), T_LOW, T_HIGH)
    assert d.outcome == NEEDS_HUMAN and "overloaded" in " ".join(d.reasons)
    assert len(client.requests) == 1


# --- response cache ------------------------------------------------------------

def test_same_packet_is_sent_once(tmp_path):
    j, client = make_judge(quoting("SAME"), tmp_path)
    d1 = j.decide(*middle_pair(), T_LOW, T_HIGH)
    d2 = j.decide(*middle_pair(), T_LOW, T_HIGH)
    assert len(client.requests) == 1 and d1.outcome == d2.outcome == MATCH
    assert d2.trace.calls[0].cached and d2.trace.cost == 0


def test_model_or_prompt_change_means_new_call(tmp_path, monkeypatch):
    j, client = make_judge(quoting("SAME"), tmp_path)
    j.decide(*middle_pair(), T_LOW, T_HIGH)
    j2 = Judge(client, j.cache, cheap="claude-other-model")
    j2.decide(*middle_pair(), T_LOW, T_HIGH)
    assert len(client.requests) == 2
    monkeypatch.setattr(judge_mod, "PROMPT_VERSION", "judge-v2")
    j.decide(*middle_pair(), T_LOW, T_HIGH)
    assert len(client.requests) == 3


def test_errors_and_invalid_answers_are_not_cached(tmp_path):
    for responder in (lambda m, t, n: RuntimeError("boom"), _fixed("garbage")):
        j, client = make_judge(responder, tmp_path / str(id(responder)))
        j.decide(*middle_pair(), T_LOW, T_HIGH)
        n = len(client.requests)
        j.decide(*middle_pair(), T_LOW, T_HIGH)
        assert len(client.requests) == 2 * n


def test_guards_run_on_cache_hits(tmp_path):
    j, client = make_judge(quoting("SAME"), tmp_path)
    sap, cands = middle_pair()
    packet = build_packet(sap, cands[0])
    key = response_cache_key(HAIKU, judge_mod.PROMPT_VERSION, packet)
    path = tmp_path / "resp" / key[:2] / f"{key}.json"
    path.parent.mkdir(parents=True)
    path.write_text(answer("SAME", [{"record": "SAP", "field": "NAME1", "quote": "NOT THERE"},
                                    {"record": "FUSION", "field": "vendor_name", "quote": "Apex"}]))
    d = j.decide(sap, cands, T_LOW, T_HIGH)
    assert d.outcome == NEEDS_HUMAN and "guard failed" in " ".join(d.reasons)
    assert len(client.requests) == 0


def test_no_cache_calls_every_time_and_writes_nothing(tmp_path):
    j, client = make_judge(quoting("SAME"), tmp_path, use_cache=False)
    j.decide(*middle_pair(), T_LOW, T_HIGH)
    j.decide(*middle_pair(), T_LOW, T_HIGH)
    assert len(client.requests) == 2 and not (tmp_path / "resp").exists()


# --- cost + audit --------------------------------------------------------------

def test_cost_per_mtok():
    assert config.cost_usd("claude-haiku-5-5", 1_000_000, 1_000_000, 1_000_000, 1_000_000) == Decimal("0.735")
    assert config.cost_usd("claude-sonnet-5-5", 1000, 200, 500, 0) == Decimal("0.00405")
    assert config.cost_usd("claude-unknown", 1, 1) is None


def test_unknown_model_cost_is_none(tmp_path):
    j, _ = make_judge(quoting("SAME"), tmp_path, cheap="claude-unknown")
    assert j.decide(*middle_pair(), T_LOW, T_HIGH).trace.cost is None


def test_audit_has_ids_and_cost_but_no_names(tmp_path):
    audit = AuditLog(tmp_path / "audit.jsonl")
    j, _ = make_judge(by_model({HAIKU: quoting("UNSURE"), SONNET: quoting("SAME")}), tmp_path, audit=audit)
    d = j.decide(*middle_pair(), T_LOW, T_HIGH)
    text = (tmp_path / "audit.jsonl").read_text()
    row = json.loads(text)
    assert row["lifnr"] == "S1" and row["fusion_vendor_id"] == 1 and row["tier_used"] == "sonnet"
    assert row["outcome"] == d.outcome == MATCH and row["packet_sha256"] == d.trace.packet_sha256
    assert [c["tier"] for c in row["calls"]] == ["haiku", "sonnet"]
    assert Decimal(row["cost_usd"]) == d.trace.cost > 0
    assert "Apex" not in text and "APEX" not in text and "Main St" not in text
