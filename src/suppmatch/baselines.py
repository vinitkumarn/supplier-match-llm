"""Baseline methods to compare against (docs/design.md, "Eval").

- exact: normalized name + country equal.
- fuzzy: difflib ratio of normalized names >= 0.90, same country.
- embed: the decision rules with embedding bands; the middle band goes to NEEDS_HUMAN.
Every proposed MATCH goes through guard_match: a tax or bank conflict is never a MATCH.
"""

from __future__ import annotations

import difflib

from .candidates import Candidate, pair_flags
from .decide import AMBIGUOUS, NEEDS_HUMAN, NEW, Decision, decide, guard_match
from .records import Vendor

FUZZY_MIN = 0.90


def _pair(sap: Vendor, fus: Vendor) -> Candidate:
    tax, bank, postal = pair_flags(sap, fus)
    return Candidate(fus, 0.0, 0.0, tax, bank, postal)


def _pick(sap: Vendor, hits: list[Vendor], rule: str, reason: str) -> Decision:
    if not hits:
        return Decision(NEW, rule, reasons=["no name match in the same country"])
    if len(hits) > 1:
        return Decision(NEEDS_HUMAN, "AMBIGUOUS", hits[0].key, [AMBIGUOUS],
                        [f"{len(hits)} Fusion suppliers match the name"], _pair(sap, hits[0]))
    return guard_match(_pair(sap, hits[0]), rule, reason)


def exact(sap: Vendor, fusion: dict[int, Vendor]) -> Decision:
    hits = [f for f in fusion.values() if f.country == sap.country and f.norm_name == sap.norm_name]
    return _pick(sap, hits, "EXACT", "same normalized name and country")


def fuzzy(sap: Vendor, fusion: dict[int, Vendor]) -> Decision:
    best, hits = 0.0, []
    for f in fusion.values():
        if f.country != sap.country:
            continue
        ratio = difflib.SequenceMatcher(None, sap.norm_name, f.norm_name).ratio()
        if ratio > best:
            best, hits = ratio, [f]
        elif ratio == best:
            hits.append(f)
    if best < FUZZY_MIN:
        hits = []
    return _pick(sap, hits, "FUZZY", f"name ratio {best:.3f} >= {FUZZY_MIN}")


def embed_only(cands: list[Candidate], t_low: float, t_high: float) -> Decision:
    d = decide(cands, t_low, t_high)
    if d.needs_judge:
        d.outcome = NEEDS_HUMAN
        d.reasons.append("embed-only: middle band goes to a person")
    return d
