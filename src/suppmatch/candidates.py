"""Candidates and features (docs/design.md, "Candidates and features").

- Blocking: an SAP vendor is only ever compared with Fusion suppliers of the same country.
- Top 5 Fusion candidates by cosine similarity of the full embedding text (numpy dot product).
- Also kept: any same-country Fusion supplier with the same tax ID or the same bank identity,
  so the FRAUD / CONFLICT / RENAMED checks never depend on the name landing in the top 5.
- Features: name_cosine, full_cosine, tax_id flag, bank flag, postal_equal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import config
from .embed import Embedder
from .normalize import clean_text, embedding_text
from .records import Vendor

SAME, DIFFERENT, MISSING = "SAME", "DIFFERENT", "MISSING"


@dataclass(frozen=True)
class Candidate:
    fusion: Vendor
    name_cosine: float
    full_cosine: float
    tax: str           # SAME / DIFFERENT / MISSING
    bank: str          # SAME / DIFFERENT / MISSING
    postal_equal: bool

    @property
    def vendor_id(self) -> int:
        return self.fusion.key


def tax_flag(a: Vendor, b: Vendor) -> str:
    """Tax IDs are already normalized with each vendor's own country (prefix rule)."""
    if not a.tax or not b.tax:
        return MISSING
    return SAME if a.tax == b.tax else DIFFERENT


def bank_flag(a: Vendor, b: Vendor) -> str:
    """Bank identity = (country, bank key, account). Same account at another bank is DIFFERENT."""
    if a.bank is None or b.bank is None:
        return MISSING
    return SAME if a.bank == b.bank else DIFFERENT


def pair_flags(sap: Vendor, fus: Vendor) -> tuple[str, str, bool]:
    postal = clean_text(sap.postal).replace(" ", "")
    return tax_flag(sap, fus), bank_flag(sap, fus), bool(postal) and postal == clean_text(fus.postal).replace(" ", "")


def full_text(v: Vendor) -> str:
    return embedding_text(v.name, v.address, v.city, v.country)


def build_candidates(sap: dict[str, Vendor], fusion: dict[int, Vendor], embedder: Embedder,
                     top_k: int = config.TOP_K) -> dict[str, list[Candidate]]:
    """For each SAP vendor: its candidates, best full_cosine first. Never crosses countries."""
    fus_list = list(fusion.values())
    sap_list = list(sap.values())
    fus_full = embedder.embed([full_text(f) for f in fus_list])
    fus_name = embedder.embed([f.norm_name for f in fus_list])
    sap_full = embedder.embed([full_text(s) for s in sap_list])
    sap_name = embedder.embed([s.norm_name for s in sap_list])

    by_country: dict[str, list[int]] = {}
    for i, f in enumerate(fus_list):
        by_country.setdefault(f.country, []).append(i)

    out: dict[str, list[Candidate]] = {}
    for si, s in enumerate(sap_list):
        idx = by_country.get(s.country, [])
        if not idx:
            out[s.key] = []
            continue
        idx_arr = np.array(idx)
        full = fus_full[idx_arr] @ sap_full[si]
        order = np.argsort(-full, kind="stable")
        keep = [int(idx_arr[o]) for o in order[:top_k]]
        for i in idx:  # exact tax / bank key pull-in
            f = fus_list[i]
            if i not in keep and ((s.tax and s.tax == f.tax) or (s.bank is not None and s.bank == f.bank)):
                keep.append(i)
        cands = []
        for i in keep:
            f = fus_list[i]
            assert f.country == s.country  # blocking
            tax, bank, postal = pair_flags(s, f)
            cands.append(Candidate(f, float(fus_name[i] @ sap_name[si]), float(fus_full[i] @ sap_full[si]),
                                   tax, bank, postal))
        cands.sort(key=lambda c: -c.full_cosine)
        out[s.key] = cands
    return out
