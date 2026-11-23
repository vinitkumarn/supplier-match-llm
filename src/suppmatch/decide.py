"""Decision rules (docs/design.md, "Decision rules"). Python owns every outcome and flag.

Order: FRAUD -> SURE MATCH -> AMBIGUOUS -> CONFLICT -> SURE NEW -> needs judge (middle band).
The model (a later PR) only ever sees the "needs judge" rows and returns a verdict; Python
turns that verdict into an outcome, and a hard conflict still blocks MATCH.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .candidates import DIFFERENT, SAME, Candidate

MATCH, NEW, NEEDS_HUMAN = "MATCH", "NEW", "NEEDS_HUMAN"
FRAUD_FLAG, TAX_CONFLICT, BANK_CONFLICT, AMBIGUOUS = "FRAUD_FLAG", "TAX_CONFLICT", "BANK_CONFLICT", "AMBIGUOUS"


@dataclass
class Decision:
    outcome: str | None                 # MATCH / NEW / NEEDS_HUMAN; None = needs judge
    rule: str                           # FRAUD / SURE_MATCH / AMBIGUOUS / CONFLICT / SURE_NEW / NEEDS_JUDGE / ...
    fusion_vendor_id: int | None = None
    flags: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    candidate: Candidate | None = None  # the pair the decision is about (or the one to judge)

    @property
    def needs_judge(self) -> bool:
        return self.outcome is None


def conflict_flags(c: Candidate) -> list[str]:
    flags = []
    if c.tax == DIFFERENT:
        flags.append(TAX_CONFLICT)
    if c.bank == DIFFERENT:
        flags.append(BANK_CONFLICT)
    return flags


def guard_match(c: Candidate, rule: str, reason: str) -> Decision:
    """Every proposed MATCH goes through here: a hard conflict is never a MATCH."""
    if c.bank == SAME and c.tax == DIFFERENT:
        return Decision(NEEDS_HUMAN, "FRAUD", c.vendor_id, [FRAUD_FLAG],
                        [reason, "same bank account but different tax ID"], c)
    flags = conflict_flags(c)
    if flags:
        return Decision(NEEDS_HUMAN, "CONFLICT", c.vendor_id, flags,
                        [reason, "blocked: " + ", ".join(flags)], c)
    return Decision(MATCH, rule, c.vendor_id, [], [reason], c)


def decide(cands: list[Candidate], t_low: float, t_high: float) -> Decision:
    """cands: best full_cosine first (from build_candidates)."""
    if not cands:
        return Decision(NEW, "SURE_NEW", reasons=["no Fusion supplier in the same country"])

    # 1. FRAUD: bank SAME and tax_id DIFFERENT (both present). No model.
    for c in cands:
        if c.bank == SAME and c.tax == DIFFERENT:
            return Decision(NEEDS_HUMAN, "FRAUD", c.vendor_id, [FRAUD_FLAG],
                            ["same bank account but different tax ID"], c)

    # 2. SURE MATCH: tax SAME and (bank SAME or full_cosine >= T_HIGH), no bank conflict.
    sure = [c for c in cands if c.tax == SAME and c.bank != DIFFERENT
            and (c.bank == SAME or c.full_cosine >= t_high)]
    tax_same_bank_diff = [c for c in cands if c.tax == SAME and c.bank == DIFFERENT]
    if len(sure) == 1 and not tax_same_bank_diff:
        c = sure[0]
        why = "tax ID and bank SAME" if c.bank == SAME else f"tax ID SAME and full_cosine {c.full_cosine:.3f} >= T_HIGH"
        return guard_match(c, "SURE_MATCH", why)
    if len(sure) + len(tax_same_bank_diff) > 1:
        both = sure + tax_same_bank_diff
        flags = [AMBIGUOUS] + ([BANK_CONFLICT] if tax_same_bank_diff else [])
        return Decision(NEEDS_HUMAN, "AMBIGUOUS", both[0].vendor_id, flags,
                        [f"{len(both)} Fusion suppliers share the tax ID"], both[0])

    # 3. AMBIGUOUS: two or more above T_HIGH with no tax / bank tie-break.
    high = [c for c in cands if c.full_cosine >= t_high]
    if len(high) >= 2 and not any(c.tax == SAME or c.bank == SAME for c in high):
        return Decision(NEEDS_HUMAN, "AMBIGUOUS", high[0].vendor_id, [AMBIGUOUS],
                        [f"{len(high)} Fusion suppliers above T_HIGH, no tax/bank tie-break"], high[0])

    # 4. CONFLICT: same tax ID but different bank, or a high-band best with a hard conflict.
    if tax_same_bank_diff:
        c = tax_same_bank_diff[0]
        return Decision(NEEDS_HUMAN, "CONFLICT", c.vendor_id, [BANK_CONFLICT],
                        ["same tax ID but different bank account"], c)
    best = cands[0]
    if best.full_cosine >= t_high and conflict_flags(best):
        return Decision(NEEDS_HUMAN, "CONFLICT", best.vendor_id, conflict_flags(best),
                        [f"full_cosine {best.full_cosine:.3f} >= T_HIGH but " + ", ".join(conflict_flags(best))], best)

    # 5. SURE NEW: best below T_LOW and no tax SAME anywhere.
    if best.full_cosine < t_low and not any(c.tax == SAME for c in cands):
        return Decision(NEW, "SURE_NEW", reasons=[f"best full_cosine {best.full_cosine:.3f} < T_LOW"])

    # 6. The unclear middle: the judge decides (later PR). Conflict flags travel with it.
    return Decision(None, "NEEDS_JUDGE", best.vendor_id, conflict_flags(best),
                    [f"middle band: best full_cosine {best.full_cosine:.3f}"], best)
