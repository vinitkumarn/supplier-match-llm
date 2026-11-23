"""Tax IDs and bank numbers never reach the model; untrusted text stays inside UNTRUSTED."""

import re

from fakes import ScriptedClient, answer, by_model, quoting

from suppmatch import config
from suppmatch.audit import AuditLog
from suppmatch.candidates import Candidate, build_candidates
from suppmatch.decide import NEEDS_HUMAN, TAX_CONFLICT
from suppmatch.evaluate import SPLITS, select_rows
from suppmatch.judge import SYSTEM_PROMPT, Judge, ResponseCache
from suppmatch.normalize import normalize_id, normalize_tax_id
from suppmatch.packet import build_packet, render

HAIKU, SONNET = config.MODEL_CHEAP, config.MODEL_STRONG


def secrets(tables) -> set[str]:
    """Every tax ID and bank account number in the CSVs: raw and normalized forms."""
    out = set()
    for r in tables["fusion_suppliers"]:
        out |= {r["tax_id"], normalize_id(r["tax_id"]), normalize_tax_id(r["tax_id"], r["country"])}
    for r in tables["sap_lfa1"]:
        out |= {r["STCD1"], normalize_id(r["STCD1"]), normalize_tax_id(r["STCD1"], r["LAND1"])}
    for r in tables["fusion_bank"]:
        out |= {r["account_number"], normalize_id(r["account_number"])}
    for r in tables["sap_lfbk"]:
        out |= {r["BANKN"], normalize_id(r["BANKN"])}
    out.discard("")
    assert min(len(s) for s in out) >= 6, "a very short number would make this scan meaningless"
    return out


def leaks(texts, values) -> list[str]:
    return sorted({v for t in texts for v in values if v in t})


def test_no_tax_or_bank_number_in_any_prompt(built, records, fake_embedder, tmp_path):
    values = secrets(built["tables"])
    assert leaks([f"x {sorted(values)[0]} y"], values)       # the scan does catch a leak
    plan = built["plan"]
    all_rows = [k for split in SPLITS for k in select_rows(plan, split)]
    assert any(plan["vendors"][k]["split"] == "redteam" for k in all_rows)
    cands = build_candidates({k: records.sap[k] for k in all_rows}, records.fusion, fake_embedder)

    # 1. Every packet the judge could ever be shown: every candidate of every SAP vendor.
    rendered = [render(build_packet(records.sap[k], c)) for k in all_rows for c in cands[k]]
    assert len(rendered) > len(all_rows)
    assert leaks(rendered + [SYSTEM_PROMPT], values) == []

    # 2. Every prompt actually sent in a routed run (wide middle band, retries, both tiers),
    #    and the audit trail.
    haiku = lambda m, t, n: "not json" if n == 1 else answer("UNSURE")
    client = ScriptedClient(by_model({HAIKU: haiku, SONNET: quoting("SAME")}))
    audit = AuditLog(tmp_path / "audit.jsonl")
    j = Judge(client, ResponseCache(tmp_path / "resp"), audit)
    for k in all_rows:
        j.decide(records.sap[k], cands[k], 0.0, 0.99)
    assert client.calls_for(HAIKU) > 20 and client.calls_for(SONNET) > 10
    assert leaks(client.prompts(), values) == []
    assert leaks([audit.path.read_text(encoding="utf-8")], values) == []


def _redteam(plan, rid):
    return next(k for k, v in plan["vendors"].items() if v["redteam_id"] == rid)


def test_r01_injection_is_inside_untrusted_and_same_still_needs_human(built, records, fake_embedder, tmp_path):
    k = _redteam(built["plan"], "R01")
    sap = records.sap[k]
    assert "answer SAME" in sap.name2
    decoy = built["plan"]["vendors"][k]["related_fusion_vendor_ids"][0]
    cands = [c for c in build_candidates({k: sap}, records.fusion, fake_embedder)[k] if c.vendor_id == decoy]
    assert cands and cands[0].tax == "DIFFERENT"

    client = ScriptedClient(quoting("SAME"))          # the model falls for the injection
    d = Judge(client, ResponseCache(tmp_path / "r")).decide(sap, cands, 0.0, 0.99)
    assert d.outcome == NEEDS_HUMAN and TAX_CONFLICT in d.flags and d.rule == "CONFLICT"

    prompt = client.requests[0]["messages"][0]["content"]
    line = next(l for l in prompt.splitlines() if l.startswith("SAP.NAME2:"))
    assert line == f"SAP.NAME2: <UNTRUSTED>{sap.name2}</UNTRUSTED>"


def test_r04_fake_closing_tag_is_escaped(built, records):
    k = _redteam(built["plan"], "R04")
    sap = records.sap[k]
    assert "</UNTRUSTED>" in sap.name2
    fus = next(iter(records.fusion.values()))
    prompt = render(build_packet(sap, Candidate(fus, 0.5, 0.5, "MISSING", "MISSING", False)))
    assert "&lt;/UNTRUSTED&gt;" in prompt
    tagged = [l for l in prompt.splitlines() if "<UNTRUSTED>" in l]
    assert all(l.count("<UNTRUSTED>") == 1 and l.count("</UNTRUSTED>") == 1 and l.endswith("</UNTRUSTED>")
               for l in tagged)
    assert len(re.findall("</UNTRUSTED>", prompt)) == len(tagged) == 15
