from conftest import vendor

from suppmatch.candidates import DIFFERENT, MISSING, SAME, bank_flag, build_candidates, tax_flag
from suppmatch.normalize import bank_identity, normalize_tax_id


def test_blocking_never_crosses_countries(records, fake_embedder):
    cands = build_candidates(records.sap, records.fusion, fake_embedder)
    assert set(cands) == set(records.sap)
    for lifnr, cs in cands.items():
        assert cs, lifnr
        assert all(c.fusion.country == records.sap[lifnr].country for c in cs), lifnr
        assert len(cs) >= min(5, sum(f.country == records.sap[lifnr].country for f in records.fusion.values()))


def test_identical_name_in_other_country_is_not_a_candidate(fake_embedder):
    sap = {"S1": vendor("S1", "Apex Tools", "US", tax="111")}
    fusion = {1: vendor(1, "Apex Tools", "CA", tax="111"),
              2: vendor(2, "Bluewater Foods", "US", tax="222")}
    cands = build_candidates(sap, fusion, fake_embedder)
    assert [c.vendor_id for c in cands["S1"]] == [2]


def test_same_account_different_bank_is_not_bank_same():
    a = vendor("S", "X", bank=bank_identity("US", "021000021", "12345678"))
    b = vendor(1, "X", bank=bank_identity("US", "026009593", "12345678"))
    c = vendor(2, "X", bank=bank_identity("US", "021-000-021", "0012 3456 78"))
    assert bank_flag(a, b) == DIFFERENT
    assert bank_flag(a, c) == SAME
    assert bank_flag(a, vendor(3, "X")) == MISSING


def test_planted_same_account_different_bank_row(built, records, fake_embedder):
    plan = built["plan"]["vendors"]
    rows = [k for k, v in plan.items() if v["note"] == "SAME_ACCOUNT_DIFFERENT_BANK"]
    assert len(rows) == 1
    lifnr = rows[0]
    decoy = plan[lifnr]["related_fusion_vendor_ids"][0]
    sap, fus = records.sap[lifnr], records.fusion[decoy]
    assert sap.bank[2] == fus.bank[2] and sap.bank[1] != fus.bank[1]  # same account, other bank
    assert bank_flag(sap, fus) == DIFFERENT
    for c in build_candidates({lifnr: sap}, records.fusion, fake_embedder)[lifnr]:
        assert c.bank != SAME


def test_tax_flag_uses_vendor_country_prefix():
    a = vendor("S", "X", "GB", tax=normalize_tax_id("123456789", "GB"))
    b = vendor(1, "X", "GB", tax=normalize_tax_id("GB 123 456 789", "GB"))
    assert tax_flag(a, b) == SAME
    assert tax_flag(a, vendor(2, "X", "GB", tax="999")) == DIFFERENT
    assert tax_flag(a, vendor(3, "X", "GB")) == MISSING


def test_tax_or_bank_twin_is_pulled_in_even_outside_top5(fake_embedder):
    bank = bank_identity("US", "1", "999")
    sap = {"S": vendor("S", "Zephyr Holdings", tax="555", bank=bank)}
    fusion = {i: vendor(i, f"Zephyr Holding {i}") for i in range(1, 8)}
    fusion[99] = vendor(99, "Completely Other Name", tax="777", bank=bank, address="9 Far Rd", city="Elsewhere")
    ids = [c.vendor_id for c in build_candidates(sap, fusion, fake_embedder)["S"]]
    assert 99 in ids and len(ids) == 6
