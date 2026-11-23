"""The packet: everything the judge model may see about one SAP / Fusion pair.

docs/design.md, "The model call": both records (original + normalized name and address,
wrapped as UNTRUSTED), the features, the tax_id flag, the bank flag and bank last4 only.
Never a tax ID, a bank key or a full bank account number: those stay in Python.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

from .candidates import Candidate
from .normalize import normalize_address, normalize_name
from .records import Vendor

SAP, FUSION = "SAP", "FUSION"


@dataclass(frozen=True)
class Packet:
    sap_key: str
    fusion_id: int
    sap_fields: dict[str, str]
    fusion_fields: dict[str, str]
    features: dict[str, object]     # name_cosine, full_cosine, postal_equal
    tax_id: str                     # SAME / DIFFERENT / MISSING
    bank: str                       # SAME / DIFFERENT / MISSING
    sap_last4: str
    fusion_last4: str

    def fields(self, record: str) -> dict[str, str] | None:
        return {SAP: self.sap_fields, FUSION: self.fusion_fields}.get(record)

    def to_json(self) -> str:
        """Canonical text of the packet (same packet -> same text -> same hash)."""
        return json.dumps(asdict(self), sort_keys=True, ensure_ascii=False, separators=(",", ":"))

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()


def build_packet(sap: Vendor, cand: Candidate) -> Packet:
    fus = cand.fusion
    sap_fields = {
        "NAME1": sap.name1 or sap.name, "NAME2": sap.name2, "STRAS": sap.address,
        "ORT01": sap.city, "PSTLZ": sap.postal, "LAND1": sap.country,
        "norm_name": sap.norm_name, "norm_address": normalize_address(sap.address),
    }
    fusion_fields = {
        "vendor_name": fus.name, "address_line1": fus.address, "city": fus.city,
        "postal_code": fus.postal, "country": fus.country,
        "norm_name": normalize_name(fus.name), "norm_address": normalize_address(fus.address),
    }
    features = {"name_cosine": round(cand.name_cosine, 3), "full_cosine": round(cand.full_cosine, 3),
                "postal_equal": cand.postal_equal}
    return Packet(str(sap.key), int(fus.key), sap_fields, fusion_fields, features,
                  cand.tax, cand.bank, sap.last4, fus.last4)


def escape(text: str) -> str:
    """< and > become &lt; and &gt;, so untrusted text can never open or close a tag."""
    return text.replace("<", "&lt;").replace(">", "&gt;")


def untrusted(text: str) -> str:
    return f"<UNTRUSTED>{escape(text)}</UNTRUSTED>"


def render(p: Packet) -> str:
    """The user message. Record fields are untrusted; the features block is computed by Python."""
    lines = ["Compare these two supplier records. Are they the same company?", "", "SAP record:"]
    lines += [f"SAP.{k}: {untrusted(v)}" for k, v in p.sap_fields.items()]
    lines += ["", "FUSION record:"]
    lines += [f"FUSION.{k}: {untrusted(v)}" for k, v in p.fusion_fields.items()]
    f = p.features
    lines += [
        "", "Computed by the system (trusted):",
        f"name_cosine: {f['name_cosine']:.3f}",
        f"full_cosine: {f['full_cosine']:.3f}",
        f"postal_equal: {'true' if f['postal_equal'] else 'false'}",
        f"tax_id: {p.tax_id}",
        f"bank: {p.bank}",
        f"bank_last4: SAP {p.sap_last4 or 'none'}, FUSION {p.fusion_last4 or 'none'}",
    ]
    return "\n".join(lines)
