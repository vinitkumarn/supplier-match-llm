"""Guards on every judge answer (fresh or from the cache). Any failure -> NEEDS_HUMAN.

- Each evidence item names a record (SAP / FUSION) and a field that exists in the packet.
- Each quote is an exact substring of that field: the raw text or the escaped text the model
  saw (< and > as &lt; and &gt;). A blank quote does not count.
- SAME needs at least one quote from the SAP record AND one from the FUSION record;
  DIFFERENT needs at least one quote. UNSURE needs none.
"""

from __future__ import annotations

from .packet import FUSION, SAP, Packet, escape
from .schema import Judgment


def check(j: Judgment, p: Packet) -> list[str]:
    """Empty list = the answer passes."""
    errors = []
    for i, ev in enumerate(j.evidence):
        fields = p.fields(ev.record)
        if fields is None or ev.field not in fields:
            errors.append(f"evidence {i}: field {ev.record}.{ev.field} does not exist")
            continue
        value = fields[ev.field]
        if not ev.quote.strip() or (ev.quote not in value and ev.quote not in escape(value)):
            errors.append(f"evidence {i}: quote is not an exact substring of {ev.record}.{ev.field}")
    records = {ev.record for ev in j.evidence}
    if j.verdict == "SAME" and not {SAP, FUSION} <= records:
        errors.append("SAME needs at least one quote from the SAP record and one from the FUSION record")
    if j.verdict == "DIFFERENT" and not j.evidence:
        errors.append("DIFFERENT needs at least one quote")
    return errors
