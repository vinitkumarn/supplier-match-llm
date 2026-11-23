"""Merge proposal for the Supplier Master team (docs/design.md, "Proposal and dollars").

One row per SAP vendor: outcome, Fusion vendor, flags, who decided, why, the survivor, field
differences, a bank note and two dollar amounts that are never added together:
- dollars_exposed: SAP open items + Fusion open AP for a suggested MATCH (a likely duplicate:
  the same invoice could be paid twice).
- dollars_under_review: the same sum for a NEEDS_HUMAN pair with a Fusion vendor (unresolved;
  a person decides).
Nothing is written to any system. The file never contains a tax ID or a bank account number
(only bank last4).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from . import config
from .candidates import SAME, bank_flag
from .decide import MATCH, NEEDS_HUMAN, NEW, Decision
from .normalize import clean_text, normalize_address, normalize_name
from .records import Records, Vendor

ROUTE_NAMES = {"rules": "rules", "haiku": "Haiku", "sonnet": "Sonnet", "human": "human"}
CALL_BACK = "verify by call-back to the number on file before adding"

COLUMNS = ["sap_lifnr", "outcome", "fusion_vendor_id", "flags", "route", "reason", "survivor",
           "name_diff", "address_diff", "postal_diff", "bank_note", "sap_open", "fusion_open",
           "dollars_exposed", "dollars_under_review"]


@dataclass
class ProposalRow:
    sap_lifnr: str
    outcome: str
    fusion_vendor_id: int | None
    flags: list[str]
    route: str
    reason: str
    survivor: str
    name_diff: str
    address_diff: str
    postal_diff: str
    bank_note: str
    sap_open: Decimal
    fusion_open: Decimal | None
    dollars_exposed: Decimal | None        # MATCH rows only
    dollars_under_review: Decimal | None   # NEEDS_HUMAN rows with a Fusion vendor only

    def cells(self) -> list[str]:
        def s(x) -> str:
            return "" if x is None else str(x)
        return [self.sap_lifnr, self.outcome, s(self.fusion_vendor_id), " ".join(self.flags), self.route,
                self.reason, self.survivor, self.name_diff, self.address_diff, self.postal_diff,
                self.bank_note, s(self.sap_open), s(self.fusion_open), s(self.dollars_exposed),
                s(self.dollars_under_review)]


def _diff(sap_text: str, fus_text: str, same: bool) -> str:
    return "same" if same else f"SAP: {sap_text} / Fusion: {fus_text}"


def bank_note(sap: Vendor, fus: Vendor) -> str:
    """Never copy a SAP bank account to Fusion. Only last4 is ever shown."""
    if sap.bank is None:
        return "no SAP bank account; nothing to add"
    flag = bank_flag(sap, fus)
    if flag == SAME:
        return f"same bank already on Fusion (last4 {fus.last4}); nothing to add"
    on_fusion = f"Fusion has a different bank (last4 {fus.last4}). " if fus.bank is not None else ""
    return f"{on_fusion}SAP bank (last4 {sap.last4}) is not copied to Fusion; {CALL_BACK}"


def build_row(lifnr: str, d: Decision, records: Records) -> ProposalRow:
    sap = records.sap[lifnr]
    vid = d.fusion_vendor_id if d.outcome in (MATCH, NEEDS_HUMAN) else None
    name_diff = address_diff = postal_diff = note = survivor = ""
    fus_open = exposed = review = None
    if vid is not None:
        fus = records.fusion[vid]
        name_diff = _diff(sap.name, fus.name, normalize_name(sap.name) == normalize_name(fus.name))
        sap_addr, fus_addr = f"{sap.address}, {sap.city}", f"{fus.address}, {fus.city}"
        address_diff = _diff(sap_addr, fus_addr,
                             (normalize_address(sap.address), clean_text(sap.city))
                             == (normalize_address(fus.address), clean_text(fus.city)))
        postal_diff = _diff(sap.postal, fus.postal,
                            clean_text(sap.postal).replace(" ", "") == clean_text(fus.postal).replace(" ", ""))
        note = bank_note(sap, fus)
        fus_open = records.fusion_open.get(vid, Decimal("0.00"))
        amount = records.pair_open(lifnr, vid)
        if d.outcome == MATCH:
            survivor = f"Fusion {vid} (system of record)"
            exposed = amount
        else:
            review = amount
    return ProposalRow(lifnr, d.outcome or NEEDS_HUMAN, vid, list(d.flags), ROUTE_NAMES.get(d.route, d.route),
                       "; ".join(d.reasons), survivor, name_diff, address_diff, postal_diff, note,
                       records.sap_open.get(lifnr, Decimal("0.00")), fus_open, exposed, review)


def build_rows(decisions: dict[str, Decision], records: Records) -> list[ProposalRow]:
    return [build_row(k, d, records) for k, d in sorted(decisions.items())]


def totals(rows: list[ProposalRow]) -> dict:
    """Row counts per outcome and the two dollar amounts, each summed on its own (Decimal)."""
    return {
        "rows": len(rows),
        **{o: sum(r.outcome == o for r in rows) for o in (MATCH, NEW, NEEDS_HUMAN)},
        "dollars_exposed": sum((r.dollars_exposed for r in rows if r.dollars_exposed is not None), Decimal("0")),
        "dollars_under_review": sum((r.dollars_under_review for r in rows if r.dollars_under_review is not None),
                                    Decimal("0")),
    }


def _md(text: str) -> str:
    """Names and addresses are untrusted text: keep them from breaking the table or acting as HTML."""
    return (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace("|", "\\|").replace("\r", " ").replace("\n", " "))


def write_proposal(rows: list[ProposalRow], method: str, split: str,
                   reports_dir: Path | None = None) -> tuple[Path, Path]:
    out_dir = Path(reports_dir or config.REPORTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"proposal_{method}_{split}.csv"
    md_path = csv_path.with_suffix(".md")
    t = totals(rows)

    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(COLUMNS)
        for r in rows:
            w.writerow(r.cells())
        w.writerow(["TOTAL"] + [""] * (len(COLUMNS) - 3) + [str(t["dollars_exposed"]), str(t["dollars_under_review"])])

    lines = [
        f"# Merge proposal: {method} on {split}", "",
        "Suggestions only. Nothing is merged, created, updated or paid. The Supplier Master team decides.", "",
        "## Totals", "",
        f"- SAP vendors: {t['rows']} (MATCH {t[MATCH]}, NEW {t[NEW]}, NEEDS_HUMAN {t[NEEDS_HUMAN]})",
        f"- Dollars exposed (suggested MATCH, likely duplicates): ${t['dollars_exposed']:,.2f}",
        f"- Dollars under review (NEEDS_HUMAN pairs, a person decides): ${t['dollars_under_review']:,.2f}",
        "- The two amounts are kept apart and never added together.", "",
        "## Rules for the reviewer", "",
        "- Survivor is always the Fusion record (system of record). Field differences are for review.",
        f"- A SAP bank account is never copied to Fusion: {CALL_BACK}.",
        "- Tax IDs and bank account numbers are not in this file (bank last4 only).", "",
        "## Rows", "",
        "| " + " | ".join(COLUMNS) + " |",
        "|" + "---|" * len(COLUMNS),
    ]
    for r in rows:
        lines.append("| " + " | ".join(_md(c) for c in r.cells()) + " |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, md_path
