"""Eval harness (docs/design.md, "Eval").

Labels come ONLY from plan.json (the planted answers), never from rules code.
Metrics, wrong merges first: a wrong merge pays the wrong company; a missed duplicate
risks a double payment. Both matter, but a wrong merge is worse.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path

from . import config
from .baselines import embed_only, exact, fuzzy
from .candidates import build_candidates
from .decide import MATCH, NEEDS_HUMAN, NEW, Decision
from .embed import CachedEmbedder
from .records import Records

METHODS = ("exact", "fuzzy", "embed")
SPLITS = ("dev", "heldout", "redteam")


def select_rows(plan: dict, split: str, limit: int | None = None) -> list[str]:
    rows = sorted(k for k, v in plan["vendors"].items() if v["split"] == split)
    return rows[:limit] if limit else rows


def run_method(method: str, records: Records, lifnrs: list[str], embedder: CachedEmbedder | None,
               t_low: float, t_high: float) -> dict[str, Decision]:
    saps = {k: records.sap[k] for k in lifnrs}
    if method == "exact":
        return {k: exact(s, records.fusion) for k, s in saps.items()}
    if method == "fuzzy":
        return {k: fuzzy(s, records.fusion) for k, s in saps.items()}
    if method == "embed":
        if embedder is None:
            raise ValueError("embed method needs an embedder")
        cands = build_candidates(saps, records.fusion, embedder)
        return {k: embed_only(cands[k], t_low, t_high) for k in saps}
    raise ValueError(f"unknown method {method!r} (use {', '.join(METHODS)})")


def _ratio(a: int, b: int) -> float | None:
    return a / b if b else None


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.1%}"


def is_wrong_merge(d: Decision, label: dict) -> bool:
    if d.outcome != MATCH:
        return False
    return MATCH not in label["allowed_outcomes"] or d.fusion_vendor_id != label["expected_fusion_vendor_id"]


@dataclass
class Metrics:
    n: int
    wrong_merges: list[str]
    duplicates: int
    duplicates_found: int
    duplicates_sent_new: list[str]
    matches_made: int
    correct_matches: int
    new_expected: int
    new_correct: int
    allowed_ok: int
    needs_human: int
    flag_rows: int
    flag_caught: int
    routing: Counter
    dollars_planted: Decimal
    dollars_found: Decimal
    by_group: dict[str, Counter] = field(default_factory=dict)
    misses: list[str] = field(default_factory=list)

    @property
    def recall(self) -> float | None:
        return _ratio(self.duplicates_found, self.duplicates)

    @property
    def precision(self) -> float | None:
        return _ratio(self.correct_matches, self.matches_made)


def score(decisions: dict[str, Decision], plan: dict, records: Records) -> Metrics:
    labels = plan["vendors"]
    m = Metrics(len(decisions), [], 0, 0, [], 0, 0, 0, 0, 0, 0, 0, 0, Counter(), Decimal("0"), Decimal("0"))
    for k, d in sorted(decisions.items()):
        lab = labels[k]
        group = lab["group"]
        m.by_group.setdefault(group, Counter())[d.outcome or "NEEDS_JUDGE"] += 1
        m.routing["middle band -> human" if d.rule == "NEEDS_JUDGE" else "rules"] += 1
        ok = d.outcome in lab["allowed_outcomes"]
        m.allowed_ok += ok
        if not ok:
            m.misses.append(k)
        if is_wrong_merge(d, lab):
            m.wrong_merges.append(k)
        if d.outcome == MATCH:
            m.matches_made += 1
            m.correct_matches += not is_wrong_merge(d, lab)
        if d.outcome == NEEDS_HUMAN:
            m.needs_human += 1
        target = lab["expected_fusion_vendor_id"]
        if lab["expected_outcome"] == MATCH:
            m.duplicates += 1
            exposed = records.sap_open.get(k, Decimal("0")) + records.fusion_open.get(target, Decimal("0"))
            m.dollars_planted += exposed
            if d.outcome == MATCH and d.fusion_vendor_id == target:
                m.duplicates_found += 1
            if d.outcome in (MATCH, NEEDS_HUMAN) and d.fusion_vendor_id == target:
                m.dollars_found += exposed
            if d.outcome == NEW:
                m.duplicates_sent_new.append(k)
        if lab["expected_outcome"] == NEW:
            m.new_expected += 1
            m.new_correct += d.outcome == NEW
        # plan.json: expected_flags apply "when the row ends as NEEDS_HUMAN"; count the rows
        # planted as NEEDS_HUMAN (a LOOKALIKE planted as NEW needs no flag).
        if lab["expected_flags"] and lab["expected_outcome"] == NEEDS_HUMAN:
            m.flag_rows += 1
            m.flag_caught += d.outcome == NEEDS_HUMAN and set(lab["expected_flags"]) <= set(d.flags)
    return m


def summary_lines(m: Metrics) -> list[str]:
    return [
        f"Wrong merges (target 0): {len(m.wrong_merges)}" + (f"  {', '.join(m.wrong_merges)}" if m.wrong_merges else ""),
        f"Duplicate recall: {m.duplicates_found}/{m.duplicates} = {_pct(m.recall)}",
        f"Match precision: {m.correct_matches}/{m.matches_made} = {_pct(m.precision)}",
        f"Duplicates sent to NEW (missed, could be paid twice): {len(m.duplicates_sent_new)}",
        f"NEW accuracy: {m.new_correct}/{m.new_expected} = {_pct(_ratio(m.new_correct, m.new_expected))}",
        f"Outcome allowed by plan: {m.allowed_ok}/{m.n} = {_pct(_ratio(m.allowed_ok, m.n))}",
        f"NEEDS_HUMAN rate: {m.needs_human}/{m.n} = {_pct(_ratio(m.needs_human, m.n))}",
        f"Fraud / conflict / ambiguous flags caught (rows planted NEEDS_HUMAN): {m.flag_caught}/{m.flag_rows}",
        "Routing: " + ", ".join(f"{k} {v}" for k, v in sorted(m.routing.items())) + "; Haiku 0, Sonnet 0 (no model yet)",
        "Cost per SAP vendor: $0 (no model calls in this method)",
        f"Dollars exposed on duplicates: found ${m.dollars_found:,.2f} of planted ${m.dollars_planted:,.2f}",
    ]


def write_report(path: Path, method: str, split: str, header: list[str], m: Metrics,
                 decisions: dict[str, Decision], plan: dict) -> None:
    lines = [f"# Eval: {method} on {split}", ""]
    lines += [f"- {h}" for h in header] + ["", "## Metrics (wrong merges first)", ""]
    lines += [f"- {s}" for s in summary_lines(m)]
    lines += ["", "## By planted group", "", "| Group | Rows | MATCH | NEW | NEEDS_HUMAN |", "|---|---|---|---|---|"]
    for g, c in sorted(m.by_group.items()):
        lines.append(f"| {g} | {sum(c.values())} | {c[MATCH]} | {c[NEW]} | {c[NEEDS_HUMAN]} |")
    lines += ["", "## Rows whose outcome the plan does not allow", ""]
    if not m.misses:
        lines.append("None.")
    else:
        lines += ["| LIFNR | Group | Expected | Got | Fusion vendor | Flags | Rule | Reason |", "|---|---|---|---|---|---|---|---|"]
        for k in m.misses:
            d, lab = decisions[k], plan["vendors"][k]
            lines.append(f"| {k} | {lab['group']} | {'/'.join(lab['allowed_outcomes'])} | {d.outcome} | "
                         f"{d.fusion_vendor_id or ''} | {' '.join(d.flags)} | {d.rule} | {'; '.join(d.reasons)} |")
    lines += ["", "Labels come from data/plan.json only. Suggestions only: nothing is merged."]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate(method: str, split: str, records: Records, plan: dict, embedder: CachedEmbedder | None,
             embedder_kind: str, limit: int | None = None, reports_dir: Path | None = None,
             data_dir: Path | None = None) -> tuple[Metrics, Path]:
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}")
    rows = select_rows(plan, split, limit)
    t_low, t_high, source = config.thresholds(embedder_kind, data_dir)
    decisions = run_method(method, records, rows, embedder if method == "embed" else None, t_low, t_high)
    m = score(decisions, plan, records)
    header = [f"Rows: {len(rows)}" + (f" (limit {limit})" if limit else "")]
    if method == "embed":
        header += [f"Embedder: {embedder_kind} ({embedder.name})",
                   f"Thresholds: T_LOW {t_low:.2f}, T_HIGH {t_high:.2f} from {source}",
                   f"Embeddings computed {embedder.computed}, cache hits {embedder.hits}"]
        if embedder_kind == "fake":
            header.append("FakeEmbedder numbers are for tests only; real numbers come from --embedder local.")
    elif method == "fuzzy":
        header.append("difflib ratio >= 0.90 on normalized names, same country")
    path = Path(reports_dir or config.REPORTS_DIR) / f"eval_{method}_{split}.md"
    write_report(path, method, split, header, m, decisions, plan)
    return m, path
