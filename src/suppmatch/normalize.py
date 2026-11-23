"""Deterministic normalization rules (docs/design.md, "Normalization").

Plain Python, no model. The same input always gives the same output.
"""

from __future__ import annotations

import re

LEGAL_SUFFIXES = (
    "INC", "INCORPORATED", "LLC", "LTD", "LIMITED", "CORP", "CORPORATION",
    "CO", "GMBH", "AG", "PVT", "PLC",
)

STREET_WORDS = {
    "STREET": "ST",
    "AVENUE": "AVE",
    "ROAD": "RD",
    "SUITE": "STE",
}

_DROP = re.compile(r"[.']")            # "Inc." -> "INC", "O'Neil" -> "ONEIL"
_PUNCT = re.compile(r"[^\w\s]|_")      # any other punctuation -> space
_SPACES = re.compile(r"\s+")
_NOT_ALNUM = re.compile(r"[^0-9A-Za-z]")


def clean_text(text: str | None) -> str:
    """Uppercase, strip punctuation, collapse spaces.

    Letters from any alphabet are kept (so a Cyrillic look-alike letter stays
    different from the Latin one: that is a red-team case, not something to hide).
    """
    if not text:
        return ""
    text = _DROP.sub("", text.upper())
    text = _PUNCT.sub(" ", text)
    return _SPACES.sub(" ", text).strip()


def normalize_name(name: str | None) -> str:
    """Clean a supplier name and remove legal suffixes from the end.

    Suffixes are removed repeatedly ("PVT LTD" -> both go) but never the
    last remaining word.
    """
    tokens = clean_text(name).split()
    while len(tokens) > 1 and tokens[-1] in LEGAL_SUFFIXES:
        tokens.pop()
    return " ".join(tokens)


def normalize_address(text: str | None) -> str:
    """Clean an address line and use standard street words (STREET -> ST ...)."""
    return " ".join(STREET_WORDS.get(t, t) for t in clean_text(text).split())


def normalize_id(value: str | None) -> str:
    """Letters and digits only, uppercase, leading zeros removed.

    Used for tax IDs, bank keys and bank account numbers. "" means missing.
    """
    if not value:
        return ""
    return _NOT_ALNUM.sub("", value).upper().lstrip("0")


def normalize_tax_id(value: str | None, country: str | None) -> str:
    """normalize_id, plus: drop the vendor's own 2-letter country prefix.

    "GB 123456789" and "123456789" for a GB vendor are the same tax ID.
    A prefix for a different country is kept (it is real information).
    """
    raw = _NOT_ALNUM.sub("", value or "").upper()
    cc = (country or "").strip().upper()
    if len(cc) == 2 and raw.startswith(cc):
        raw = raw[2:]
    return raw.lstrip("0")


def bank_identity(country: str | None, bank_key: str | None, account: str | None) -> tuple[str, str, str] | None:
    """A bank account is only unique within one bank: identity is
    (country, bank key, account number), all normalized. None when there is no account."""
    acct = normalize_id(account)
    if not acct:
        return None
    return ((country or "").strip().upper(), normalize_id(bank_key), acct)


def sap_name(name1: str | None, name2: str | None) -> str:
    """SAP splits long names over NAME1 and NAME2; join the non-empty parts."""
    return " ".join(p.strip() for p in (name1, name2) if p and p.strip())


def embedding_text(name: str | None, address: str | None, city: str | None, country: str | None) -> str:
    """Text that gets embedded: normalized "name | address | city | country"."""
    return " | ".join(
        [normalize_name(name), normalize_address(address), clean_text(city), clean_text(country)]
    )
