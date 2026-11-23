"""Audit trail: one JSON line per SAP vendor the judge looked at.

Ids, hashes, tier used, tokens, cost and outcome only. No names, no addresses, no tax IDs, no
bank numbers and no model text (full error messages go to the normal log instead).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .judge import PROMPT_VERSION, JudgeTrace


def _money(x) -> str | None:
    return None if x is None else str(x)


class AuditLog:
    def __init__(self, path: Path):
        self.path = Path(path)

    def write(self, lifnr: str, decision, trace: JudgeTrace) -> None:
        row = {
            "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "method": trace.method,
            "lifnr": str(lifnr),
            "fusion_vendor_id": decision.candidate.vendor_id if decision.candidate else None,
            "packet_sha256": trace.packet_sha256,
            "prompt_version": PROMPT_VERSION,
            "tiers": [{"tier": t.tier, "model": t.model, "cache_key": t.cache_key,
                       "verdict": t.judgment.verdict if t.judgment else None,
                       "ok": t.error is None} for t in trace.tiers],
            "calls": [{"tier": c.tier, "model": c.model, "attempt": c.attempt, "cached": c.cached,
                       "input_tokens": c.input_tokens, "output_tokens": c.output_tokens,
                       "cache_read_tokens": c.cache_read_tokens, "cache_write_tokens": c.cache_write_tokens,
                       "stop_reason": c.stop_reason, "cost_usd": _money(c.cost)} for c in trace.calls],
            "tier_used": decision.route,
            "cost_usd": _money(trace.cost),
            "outcome": decision.outcome,
            "rule": decision.rule,
            "flags": list(decision.flags),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
