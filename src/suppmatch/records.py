"""Read the DuckDB tables into small Python records (normalized once, used everywhere)."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from .db import connect
from .normalize import bank_identity, normalize_name, normalize_tax_id, sap_name


@dataclass(frozen=True)
class Vendor:
    """One supplier from either system. `key` is the SAP LIFNR or the Fusion vendor_id.

    tax and bank are normalized for comparing only; they never reach a model.
    """

    key: str | int
    name: str
    address: str
    city: str
    postal: str
    country: str
    tax: str                               # normalized, "" = missing
    bank: tuple[str, str, str] | None      # (country, bank key, account), None = missing
    last4: str
    name1: str = ""                        # raw SAP NAME1 / NAME2 (shown to the judge as UNTRUSTED)
    name2: str = ""

    @property
    def norm_name(self) -> str:
        return normalize_name(self.name)


@dataclass
class Records:
    sap: dict[str, Vendor]
    fusion: dict[int, Vendor]
    sap_open: dict[str, Decimal]       # LIFNR -> sum of open items
    fusion_open: dict[int, Decimal]    # vendor_id -> sum of open AP

    def pair_open(self, lifnr: str, vendor_id: int | None) -> Decimal:
        """SAP open items + Fusion open AP for one SAP vendor and one Fusion supplier."""
        zero = Decimal("0.00")
        fus = self.fusion_open.get(vendor_id, zero) if vendor_id is not None else zero
        return self.sap_open.get(lifnr, zero) + fus


def _s(value) -> str:
    return "" if value is None else str(value)


def load_records(db_path: Path) -> Records:
    with connect(db_path) as con:
        sap_bank = {r[0]: r[1:] for r in con.execute(
            "SELECT LIFNR, BANKS, BANKL, BANKN FROM sap_lfbk ORDER BY LIFNR").fetchall()}
        sap: dict[str, Vendor] = {}
        for lifnr, n1, n2, stras, ort, pstlz, land, stcd in con.execute(
                "SELECT LIFNR, NAME1, NAME2, STRAS, ORT01, PSTLZ, LAND1, STCD1 FROM sap_lfa1 ORDER BY LIFNR").fetchall():
            b = sap_bank.get(lifnr)
            ident = bank_identity(*b) if b else None
            sap[lifnr] = Vendor(lifnr, sap_name(n1, n2), _s(stras), _s(ort), _s(pstlz), _s(land),
                                normalize_tax_id(stcd, land), ident, ident[2][-4:] if ident else "",
                                _s(n1), _s(n2))

        # One site and at most one bank account per Fusion vendor in this data; take the first.
        sites = {}
        for vid, line1, city, postal in con.execute(
                "SELECT vendor_id, address_line1, city, postal_code FROM fusion_sites ORDER BY rowid").fetchall():
            sites.setdefault(vid, (line1, city, postal))
        banks = {}
        for vid, country, key, acct, last4 in con.execute(
                "SELECT vendor_id, country, bank_key, account_number, last4 FROM fusion_bank ORDER BY rowid").fetchall():
            banks.setdefault(vid, (bank_identity(country, key, acct), _s(last4)))
        fusion: dict[int, Vendor] = {}
        for vid, name, tax, country in con.execute(
                "SELECT vendor_id, vendor_name, tax_id, country FROM fusion_suppliers ORDER BY vendor_id").fetchall():
            line1, city, postal = sites.get(vid, ("", "", ""))
            ident, last4 = banks.get(vid, (None, ""))
            fusion[vid] = Vendor(vid, _s(name), _s(line1), _s(city), _s(postal), _s(country),
                                 normalize_tax_id(tax, country), ident, last4)

        sap_open = {k: Decimal(v) for k, v in con.execute(
            "SELECT LIFNR, sum(DMBTR) FROM sap_open_items GROUP BY LIFNR").fetchall()}
        fusion_open = {k: Decimal(v) for k, v in con.execute(
            "SELECT vendor_id, sum(amount_remaining) FROM fusion_open_ap GROUP BY vendor_id").fetchall()}
    return Records(sap, fusion, sap_open, fusion_open)
