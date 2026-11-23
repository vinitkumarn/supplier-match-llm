import pytest
from conftest import vendor

from suppmatch.candidates import DIFFERENT, MISSING, SAME, Candidate
from suppmatch.decide import (
    AMBIGUOUS, BANK_CONFLICT, FRAUD_FLAG, MATCH, NEEDS_HUMAN, NEW, TAX_CONFLICT, decide, guard_match,
)

T_LOW, T_HIGH = 0.6, 0.9


def cand(vid, full, tax=MISSING, bank=MISSING, name=None):
    return Candidate(vendor(vid, f"V{vid}"), name if name is not None else full, full, tax, bank, True)


def test_fraud_same_bank_different_tax():
    d = decide([cand(1, 0.99, tax=DIFFERENT, bank=SAME)], T_LOW, T_HIGH)
    assert (d.outcome, d.rule, d.flags) == (NEEDS_HUMAN, "FRAUD", [FRAUD_FLAG])


def test_fraud_wins_even_when_another_candidate_is_a_sure_match():
    d = decide([cand(1, 0.99, tax=SAME, bank=MISSING), cand(2, 0.3, tax=DIFFERENT, bank=SAME)], T_LOW, T_HIGH)
    assert d.rule == "FRAUD" and d.fusion_vendor_id == 2


@pytest.mark.parametrize("tax,bank,flag", [
    (DIFFERENT, MISSING, TAX_CONFLICT), (DIFFERENT, DIFFERENT, TAX_CONFLICT),
    (MISSING, DIFFERENT, BANK_CONFLICT), (SAME, DIFFERENT, BANK_CONFLICT),
])
@pytest.mark.parametrize("full", [1.0, 0.95, 0.75, 0.2])
def test_conflicts_never_match(tax, bank, flag, full):
    d = decide([cand(1, full, tax=tax, bank=bank)], T_LOW, T_HIGH)
    assert d.outcome != MATCH
    if d.outcome == NEEDS_HUMAN or d.needs_judge:
        assert flag in d.flags


def test_high_band_tax_conflict_is_needs_human_without_model():
    d = decide([cand(1, 1.0, tax=DIFFERENT)], T_LOW, T_HIGH)
    assert (d.outcome, d.rule, d.flags) == (NEEDS_HUMAN, "CONFLICT", [TAX_CONFLICT])


def test_same_tax_different_bank_is_bank_conflict():
    d = decide([cand(1, 0.5, tax=SAME, bank=DIFFERENT)], T_LOW, T_HIGH)
    assert (d.outcome, d.flags) == (NEEDS_HUMAN, [BANK_CONFLICT])


def test_guard_match_blocks_conflicts():
    assert guard_match(cand(1, 1.0, tax=SAME, bank=SAME), "X", "r").outcome == MATCH
    assert guard_match(cand(1, 1.0, tax=DIFFERENT, bank=SAME), "X", "r").flags == [FRAUD_FLAG]
    assert guard_match(cand(1, 1.0, tax=DIFFERENT), "X", "r").outcome == NEEDS_HUMAN
    assert guard_match(cand(1, 1.0, bank=DIFFERENT), "X", "r").flags == [BANK_CONFLICT]


def test_ambiguous_two_high_no_tie_break():
    d = decide([cand(1, 0.97), cand(2, 0.95)], T_LOW, T_HIGH)
    assert (d.outcome, d.rule, d.flags) == (NEEDS_HUMAN, "AMBIGUOUS", [AMBIGUOUS])


def test_two_high_with_tax_tie_break_is_match():
    d = decide([cand(1, 0.97), cand(2, 0.95, tax=SAME)], T_LOW, T_HIGH)
    assert (d.outcome, d.fusion_vendor_id) == (MATCH, 2)


def test_two_tax_same_candidates_are_ambiguous():
    d = decide([cand(1, 0.97, tax=SAME, bank=SAME), cand(2, 0.95, tax=SAME)], T_LOW, T_HIGH)
    assert (d.outcome, d.rule) == (NEEDS_HUMAN, "AMBIGUOUS")


def test_sure_match():
    assert decide([cand(1, 0.95, tax=SAME)], T_LOW, T_HIGH).outcome == MATCH
    assert decide([cand(1, 0.30, tax=SAME, bank=SAME)], T_LOW, T_HIGH).outcome == MATCH  # renamed


def test_sure_new():
    assert decide([cand(1, 0.4)], T_LOW, T_HIGH).outcome == NEW
    assert decide([], T_LOW, T_HIGH).outcome == NEW
    # below T_LOW but tax SAME: not sure NEW, goes to the judge
    assert decide([cand(1, 0.4, tax=SAME)], T_LOW, T_HIGH).needs_judge


def test_middle_band_needs_judge():
    d = decide([cand(1, 0.75, tax=DIFFERENT)], T_LOW, T_HIGH)
    assert d.needs_judge and d.outcome is None and d.flags == [TAX_CONFLICT] and d.fusion_vendor_id == 1
    assert decide([cand(1, 0.95)], T_LOW, T_HIGH).needs_judge  # high but tax MISSING
