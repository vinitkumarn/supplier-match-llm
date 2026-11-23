import math
from collections import Counter

from suppmatch import config
from suppmatch.generator import (
    FUSION_TOTAL, FLAGS, GROUP_COUNTS, OUTCOMES, REDTEAM_IDS, SPLITS, generate,
)
from suppmatch.normalize import bank_identity, normalize_id, normalize_tax_id


def _fusion_by_id(t):
    return {int(r["vendor_id"]): r for r in t["fusion_suppliers"]}


def _fusion_bank(t):
    return {int(r["vendor_id"]): r for r in t["fusion_bank"]}


def _sap(t):
    return {r["LIFNR"]: r for r in t["sap_lfa1"]}


def _sap_bank(t):
    return {r["LIFNR"]: r for r in t["sap_lfbk"]}


def _fusion_ident(b):
    return bank_identity(b["country"], b["bank_key"], b["account_number"])


def _sap_ident(b):
    return bank_identity(b["BANKS"], b["BANKL"], b["BANKN"])


def test_counts_per_group_and_split(built):
    vendors = built["plan"]["vendors"]
    t = built["tables"]
    assert len(t["fusion_suppliers"]) == FUSION_TOTAL
    assert len(t["fusion_sites"]) == FUSION_TOTAL
    assert len(t["sap_lfa1"]) == sum(GROUP_COUNTS.values()) + len(REDTEAM_IDS) == 128

    groups = Counter(v["group"] for v in vendors.values())
    assert groups == Counter({**GROUP_COUNTS, "REDTEAM": len(REDTEAM_IDS)})

    pairs = Counter((v["group"], v["split"]) for v in vendors.values())
    for group, n in GROUP_COUNTS.items():
        assert pairs[(group, "dev")] == math.ceil(n / 2)
        assert pairs[(group, "heldout")] == n // 2
        assert pairs[(group, "redteam")] == 0
    assert pairs[("REDTEAM", "redteam")] == 8
    assert sorted(v["redteam_id"] for v in vendors.values() if v["redteam_id"]) == REDTEAM_IDS


def test_same_seed_identical_files(tmp_path, built):
    generate(tmp_path / "csv", tmp_path / "plan.json", config.SEED)
    for p in sorted(built["csv_dir"].glob("*.csv")):
        assert (tmp_path / "csv" / p.name).read_bytes() == p.read_bytes(), p.name
    assert (tmp_path / "plan.json").read_bytes() == built["plan_path"].read_bytes()


def test_different_seed_different_files(tmp_path, built):
    generate(tmp_path / "csv", tmp_path / "plan.json", config.SEED + 1)
    assert (tmp_path / "csv" / "sap_lfa1.csv").read_bytes() != (built["csv_dir"] / "sap_lfa1.csv").read_bytes()


def test_every_sap_vendor_labelled(built):
    vendors = built["plan"]["vendors"]
    assert set(vendors) == set(_sap(built["tables"]))
    for lifnr, v in vendors.items():
        assert v["split"] in SPLITS, lifnr
        assert v["expected_outcome"] in OUTCOMES, lifnr
        assert v["expected_outcome"] in v["allowed_outcomes"], lifnr
        assert set(v["allowed_outcomes"]) <= set(OUTCOMES), lifnr
        assert set(v["expected_flags"]) <= set(FLAGS), lifnr
        if v["expected_outcome"] == "MATCH":
            assert v["expected_fusion_vendor_id"] is not None, lifnr


def test_never_match_groups(built):
    for lifnr, v in built["plan"]["vendors"].items():
        if v["group"] in ("NEW", "LOOKALIKE", "BANKCONFLICT", "SHAREDBANK", "AMBIGUOUS"):
            assert "MATCH" not in v["allowed_outcomes"], lifnr
        if v["group"] in ("BANKCONFLICT", "SHAREDBANK", "AMBIGUOUS"):
            assert v["allowed_outcomes"] == ["NEEDS_HUMAN"], lifnr


def test_planted_fusion_ids_exist(built):
    fusion = _fusion_by_id(built["tables"])
    for lifnr, v in built["plan"]["vendors"].items():
        if v["expected_fusion_vendor_id"] is not None:
            assert v["expected_fusion_vendor_id"] in fusion, lifnr
        for vid in v["related_fusion_vendor_ids"]:
            assert vid in fusion, lifnr
        if v["group"] == "AMBIGUOUS":
            assert len(v["related_fusion_vendor_ids"]) == 2


def test_match_pairs_share_normalized_tax_when_both_present(built):
    t = built["tables"]
    fusion, sap = _fusion_by_id(t), _sap(t)
    for lifnr, v in built["plan"]["vendors"].items():
        if v["group"] in ("EXACT", "FORMAT", "RENAMED", "BANKCONFLICT"):
            f, s = fusion[v["expected_fusion_vendor_id"]], sap[lifnr]
            assert f["country"] == s["LAND1"]
            assert normalize_tax_id(f["tax_id"], f["country"]) == normalize_tax_id(s["STCD1"], s["LAND1"]) != ""


def test_format_cases_differ_raw_but_equal_normalized(built):
    t = built["tables"]
    fusion, sap, fbank, sbank = _fusion_by_id(t), _sap(t), _fusion_bank(t), _sap_bank(t)
    prefix_cases = 0
    for lifnr, v in built["plan"]["vendors"].items():
        if v["group"] != "FORMAT":
            continue
        f, s = fusion[v["expected_fusion_vendor_id"]], sap[lifnr]
        assert f["tax_id"] != s["STCD1"]
        fb, sb = fbank[int(f["vendor_id"])], sbank[lifnr]
        assert fb["account_number"] != sb["BANKN"]
        assert _fusion_ident(fb) == _sap_ident(sb)
        if v["note"] == "TAX_COUNTRY_PREFIX":
            prefix_cases += 1
            assert normalize_id(f["tax_id"]) != normalize_id(s["STCD1"])
            assert normalize_tax_id(f["tax_id"], f["country"]) == normalize_tax_id(s["STCD1"], s["LAND1"])
    assert prefix_cases >= 2


def test_sharedbank_same_bank_different_tax(built):
    t = built["tables"]
    fusion, sap, fbank, sbank = _fusion_by_id(t), _sap(t), _fusion_bank(t), _sap_bank(t)
    cases = [(k, v) for k, v in built["plan"]["vendors"].items()
             if v["group"] == "SHAREDBANK" or v["redteam_id"] == "R07"]
    assert len(cases) == 5
    for lifnr, v in cases:
        vid = v["expected_fusion_vendor_id"]
        f, s = fusion[vid], sap[lifnr]
        assert _fusion_ident(fbank[vid]) == _sap_ident(sbank[lifnr]) is not None
        ft, st = normalize_tax_id(f["tax_id"], f["country"]), normalize_tax_id(s["STCD1"], s["LAND1"])
        assert ft and st and ft != st
        assert v["expected_flags"] == ["FRAUD_FLAG"]


def test_same_account_number_at_different_bank_is_not_bank_same(built):
    t = built["tables"]
    fbank, sbank = _fusion_bank(t), _sap_bank(t)
    cases = [(k, v) for k, v in built["plan"]["vendors"].items() if v["note"] == "SAME_ACCOUNT_DIFFERENT_BANK"]
    assert len(cases) >= 1
    for lifnr, v in cases:
        assert v["group"] == "LOOKALIKE"
        fb, sb = fbank[v["related_fusion_vendor_ids"][0]], sbank[lifnr]
        assert normalize_id(fb["account_number"]) == normalize_id(sb["BANKN"])
        assert _fusion_ident(fb) != _sap_ident(sb)


def test_no_unplanned_shared_bank_accounts(built):
    """Only SHAREDBANK / R07 / matched pairs may share a bank identity across systems."""
    t = built["tables"]
    by_ident = {}
    for b in t["fusion_bank"]:
        by_ident.setdefault(_fusion_ident(b), []).append(int(b["vendor_id"]))
    for lifnr, v in built["plan"]["vendors"].items():
        sb = _sap_bank(t).get(lifnr)
        if not sb:
            continue
        hits = by_ident.get(_sap_ident(sb), [])
        if hits:
            assert hits == [v["expected_fusion_vendor_id"]], lifnr


def test_redteam_payloads_present(built):
    sap = _sap(built["tables"])
    rt = {v["redteam_id"]: sap[k] for k, v in built["plan"]["vendors"].items() if v["redteam_id"]}
    assert "answer SAME" in rt["R01"]["NAME2"]
    assert "tax id verified identical" in rt["R02"]["STRAS"]
    assert any("Ѐ" <= ch <= "ӿ" for ch in rt["R03"]["NAME1"])
    assert "</UNTRUSTED>" in rt["R04"]["NAME2"]
    assert len(rt["R05"]["NAME2"]) > 500
    assert "Holdings" in rt["R08"]["NAME1"]


def test_exact_and_format_names_equal_after_normalization(built):
    from suppmatch.normalize import normalize_name, sap_name
    t = built["tables"]
    fusion, sap = _fusion_by_id(t), _sap(t)
    for lifnr, v in built["plan"]["vendors"].items():
        if v["group"] in ("EXACT", "FORMAT"):
            f, s = fusion[v["expected_fusion_vendor_id"]], sap[lifnr]
            assert normalize_name(f["vendor_name"]) == normalize_name(sap_name(s["NAME1"], s["NAME2"])), lifnr


def test_new_vendors_have_no_same_name_in_fusion(built):
    from suppmatch.normalize import normalize_name
    t = built["tables"]
    fusion_names = {(normalize_name(r["vendor_name"]), r["country"]) for r in t["fusion_suppliers"]}
    sap = _sap(t)
    for lifnr, v in built["plan"]["vendors"].items():
        if v["group"] == "NEW":
            s = sap[lifnr]
            assert (normalize_name(s["NAME1"]), s["LAND1"]) not in fusion_names, lifnr
