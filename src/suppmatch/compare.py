"""Side-by-side comparison of the five methods on one split.

Reads reports/eval_<method>_<split>.json (written by evaluate.py), never the md reports.
A method with no JSON shows "not run" in every row.
"""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from typing import Callable

from . import config
from .evaluate import FLAG_TYPES, METHODS, ROUTES, TIERS

NOT_RUN = "not run"


def _pct(x) -> str:
    return "n/a" if x is None else f"{x:.1%}"


def _usd(x: str | None, places: int = 2) -> str:
    return "unknown" if x is None else f"${Decimal(x):,.{places}f}"


def _frac(a: int, b: int, rate) -> str:
    return f"{a}/{b} ({_pct(rate)})"


ROWS: list[tuple[str, Callable[[dict], str]]] = [
    ("Wrong merges (target 0)", lambda j: str(j["wrong_merges"])),
    ("Duplicate recall", lambda j: _frac(j["duplicates_found"], j["duplicates"], j["recall"])),
    ("Match precision", lambda j: _frac(j["correct_matches"], j["matches_made"], j["precision"])),
    ("NEW accuracy", lambda j: _frac(j["new_correct"], j["new_expected"], j["new_accuracy"])),
    ("NEEDS_HUMAN rate", lambda j: _frac(j["needs_human"], j["n"], j["needs_human_rate"])),
    *[(f"Flags caught: {f}", (lambda f: lambda j: f"{j['flags'][f]['caught']}/{j['flags'][f]['planted']}")(f))
      for f in FLAG_TYPES],
    ("Routing share (rules / Haiku / Sonnet / human)",
     lambda j: " / ".join(str(j["routing"][r]) for r in ROUTES)),
    ("Calls per tier (Haiku / Sonnet)", lambda j: " / ".join(str(j["calls"][t]) for t in TIERS)),
    ("Cost per SAP vendor", lambda j: _usd(j["cost_per_vendor"], 6)),
    ("Cost total", lambda j: _usd(j["cost_total"], 6)),
    ("max_tokens stops (Haiku / Sonnet)", lambda j: " / ".join(str(j["max_tokens_stops"][t]) for t in TIERS)),
    ("Embeddings computed / cached",
     lambda j: "n/a" if j["embeddings_computed"] is None else f"{j['embeddings_computed']} / {j['embeddings_cached']}"),
    ("Dollars exposed: found / planted",
     lambda j: f"{_usd(j['dollars_exposed_found'])} / {_usd(j['dollars_exposed_planted'])}"),
    ("Dollars under review (NEEDS_HUMAN)", lambda j: _usd(j["dollars_under_review"])),
    ("Rows", lambda j: str(j["n"]) + (f" (limit {j['limit']})" if j.get("limit") else "")),
    ("Embedder", lambda j: j["embedder"] or "none"),
]


def load_runs(split: str, reports_dir: Path | None = None) -> dict[str, dict | None]:
    out = {}
    for m in METHODS:
        path = Path(reports_dir or config.REPORTS_DIR) / f"eval_{m}_{split}.json"
        out[m] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    return out


def compare(split: str, reports_dir: Path | None = None) -> Path:
    runs = load_runs(split, reports_dir)
    lines = [f"# Compare methods on {split}", "",
             "Numbers come from reports/eval_<method>_<split>.json. \"not run\" = no eval for that method yet.",
             "Wrong merges first: a wrong merge pays the wrong company.", "",
             "| Metric | " + " | ".join(METHODS) + " |", "|---|" + "---|" * len(METHODS)]
    for label, fn in ROWS:
        lines.append(f"| {label} | " + " | ".join(NOT_RUN if runs[m] is None else fn(runs[m]) for m in METHODS) + " |")
    embedders = {r["embedder"] for r in runs.values() if r and r["embedder"]}
    if len(embedders) > 1:
        lines += ["", f"Warning: runs use different embedders ({', '.join(sorted(embedders))}); "
                      "re-run them with the same --embedder before comparing."]
    if "fake" in embedders:
        lines += ["", "FakeEmbedder runs are for tests only; real numbers come from --embedder local."]
    lines += ["", "Dollars exposed and dollars under review are separate amounts and are never added together.",
              "Suggestions only: nothing is merged."]
    path = Path(reports_dir or config.REPORTS_DIR) / f"compare_{split}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
