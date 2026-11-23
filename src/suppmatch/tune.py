"""Threshold tuning on the DEV split only (docs/design.md, "Threshold tuning").

Each threshold is tuned for its own job:
- T_HIGH: the most duplicates matched by rule while dev match precision stays >= 0.98.
- T_LOW: as high as possible (fewest rows left for the judge) while no dev duplicate's
  correct Fusion supplier scores below it (so no real duplicate is thrown out as NEW).
Held-out and red-team rows are never read here. Supplier Master owns the result.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from . import config
from .candidates import build_candidates, full_text
from .decide import MATCH, decide
from .embed import CachedEmbedder
from .records import Records

PRECISION_TARGET = 0.98
T_HIGH_GRID = [round(0.70 + i * 0.01, 2) for i in range(30)]   # 0.70 .. 0.99


@dataclass
class TuneResult:
    embedder: str
    embed_model: str
    t_low: float
    t_high: float
    target_met: bool
    dev_rows: int
    dev_duplicates: int
    rule_matches: int
    rule_correct: int
    precision: float | None
    recall: float
    duplicates_below_t_low: int
    lowest_duplicate_cosine: float | None
    middle_band: int
    tuned_on: str = "dev"


def tune(records: Records, plan: dict, embedder: CachedEmbedder, embedder_kind: str):
    """Returns (TuneResult, rows used, t_high table, t_low table)."""
    labels = plan["vendors"]
    rows = sorted(k for k, v in labels.items() if v["split"] == "dev")
    assert rows and all(labels[k]["split"] == "dev" for k in rows), "tuning must read dev rows only"
    dev = {k: labels[k] for k in rows}
    saps = {k: records.sap[k] for k in rows}
    cands = build_candidates(saps, records.fusion, embedder)
    dups = [k for k in rows if dev[k]["expected_outcome"] == MATCH]

    # T_HIGH: rule-made MATCHes do not depend on T_LOW.
    high_table = []
    for t_high in T_HIGH_GRID:
        made = correct = 0
        for k in rows:
            d = decide(cands[k], 0.0, t_high)
            if d.outcome == MATCH:
                made += 1
                correct += MATCH in dev[k]["allowed_outcomes"] and d.fusion_vendor_id == dev[k]["expected_fusion_vendor_id"]
        prec = correct / made if made else None
        high_table.append((t_high, made, correct, prec, correct / len(dups) if dups else 0.0))
    ok = [r for r in high_table if r[3] is not None and r[3] >= PRECISION_TARGET]
    target_met = bool(ok)
    if ok:
        best = max(ok, key=lambda r: (r[4], r[0]))
    else:
        best = max(high_table, key=lambda r: (r[3] or 0.0, r[4], r[0]))
    t_high, made, correct, prec, recall = best

    # T_LOW: cosine of each dev duplicate with its correct Fusion supplier.
    dup_cos = {}
    for k in dups:
        target = records.fusion[dev[k]["expected_fusion_vendor_id"]]
        a, b = embedder.embed([full_text(records.sap[k]), full_text(target)])
        dup_cos[k] = float(a @ b)
    lowest = min(dup_cos.values()) if dup_cos else None
    low_table = []
    for i in range(int(round(t_high * 100))):
        t_low = round(i * 0.01, 2)
        below = sum(c < t_low for c in dup_cos.values())
        middle = sum(decide(cands[k], t_low, t_high).needs_judge for k in rows)
        low_table.append((t_low, below, middle))
    zero = [r for r in low_table if r[1] == 0]
    t_low, below, middle = max(zero, key=lambda r: (-r[2], r[0]))

    result = TuneResult(embedder_kind, embedder.name, t_low, t_high, target_met, len(rows), len(dups),
                        made, correct, prec, recall, below, lowest, middle)
    return result, rows, high_table, low_table


def write_outputs(result: TuneResult, high_table, low_table, reports_dir: Path | None = None,
                  data_dir: Path | None = None) -> tuple[Path, Path]:
    json_path = config.thresholds_path(result.embedder, data_dir)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(asdict(result), indent=2) + "\n", encoding="utf-8")

    p = lambda x: "n/a" if x is None else f"{x:.1%}"  # noqa: E731
    lines = [
        f"# Thresholds for embedder `{result.embedder}` ({result.embed_model})", "",
        "Tuned on the **dev split only** (held-out and red-team rows are never read). "
        "Supplier Master owns these values.", "",
    ]
    if result.embedder == "fake":
        lines += ["> FakeEmbedder thresholds are for tests only. Run "
                  "`python -m suppmatch tune --embedder local` before any real eval.", ""]
    lines += [
        "## Chosen", "",
        f"- **T_HIGH = {result.t_high:.2f}**: rule-made MATCHes {result.rule_matches}, correct "
        f"{result.rule_correct}, dev precision {p(result.precision)} (target >= {PRECISION_TARGET:.0%}"
        f"{'' if result.target_met else ', NOT MET: highest precision chosen instead'}), "
        f"recall {result.rule_correct}/{result.dev_duplicates} = {p(result.recall)}.",
        f"- **T_LOW = {result.t_low:.2f}**: dev duplicates whose correct Fusion supplier scores below "
        f"T_LOW: {result.duplicates_below_t_low} (lowest duplicate cosine "
        f"{'n/a' if result.lowest_duplicate_cosine is None else f'{result.lowest_duplicate_cosine:.3f}'}); "
        f"middle band (rows left for the judge): {result.middle_band} of {result.dev_rows} dev rows.",
        "",
        "## T_HIGH grid (dev)", "", "| T_HIGH | Rule MATCHes | Correct | Precision | Recall |", "|---|---|---|---|---|",
    ]
    lines += [f"| {t:.2f} | {m} | {c} | {p(pr)} | {p(rc)} |" for t, m, c, pr, rc in high_table]
    lines += ["", f"## T_LOW grid (dev, with T_HIGH {result.t_high:.2f})", "",
              "| T_LOW | Duplicates below T_LOW | Middle band |", "|---|---|---|"]
    lines += [f"| {t:.2f} | {b} | {mid} |" for t, b, mid in low_table if round(t * 100) % 5 == 0 or t == result.t_low]
    reports = Path(reports_dir or config.REPORTS_DIR)
    reports.mkdir(parents=True, exist_ok=True)
    md_path = reports / f"thresholds_{result.embedder}.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path
