"""Eval harness (docs/design.md, "Eval").

Labels come ONLY from plan.json (the planted answers), never from rules code.
Metrics, wrong merges first: a wrong merge pays the wrong company; a missed duplicate
risks a double payment. Both matter, but a wrong merge is worse.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path

from . import config
from .baselines import embed_only, exact, fuzzy
from .candidates import build_candidates
from .decide import AMBIGUOUS, BANK_CONFLICT, FRAUD_FLAG, MATCH, NEEDS_HUMAN, NEW, TAX_CONFLICT, Decision
from .embed import CachedEmbedder
from .judge import PROMPT_VERSION, ROUTED, SONNET_ONLY, Judge
from .records import Records

MODEL_METHODS = (ROUTED, SONNET_ONLY)
METHODS = ("exact", "fuzzy", "embed") + MODEL_METHODS
ROUTES = ("rules", "haiku", "sonnet", "human")
TIERS = ("haiku", "sonnet")
SPLITS = ("dev", "heldout", "redteam")
FLAG_TYPES = (FRAUD_FLAG, TAX_CONFLICT, BANK_CONFLICT, AMBIGUOUS)


def select_rows(plan: dict, split: str, limit: int | None = None) -> list[str]:
    rows = sorted(k for k, v in plan["vendors"].items() if v["split"] == split)
    return rows[:limit] if limit else rows


def run_method(method: str, records: Records, lifnrs: list[str], embedder: CachedEmbedder | None,
               t_low: float, t_high: float, judge: Judge | None = None) -> dict[str, Decision]:
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
    if method in MODEL_METHODS:
        if embedder is None or judge is None:
            raise ValueError(f"{method} method needs an embedder and a judge")
        cands = build_candidates(saps, records.fusion, embedder)
        return {k: judge.decide(s, cands[k], t_low, t_high, method) for k, s in saps.items()}
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
    dollars_planted: Decimal            # planted duplicates: SAP open + Fusion open of the real pair
    dollars_found: Decimal              # of those, the ones suggested MATCH to the right Fusion supplier
    by_group: dict[str, Counter] = field(default_factory=dict)
    misses: list[str] = field(default_factory=list)
    calls: Counter = field(default_factory=Counter)        # real API calls per tier (retries included)
    cache_hits: Counter = field(default_factory=Counter)   # response-cache answers per tier
    max_tokens_stops: Counter = field(default_factory=Counter)  # calls cut off at max_tokens, per tier
    cost: Decimal | None = Decimal("0")                    # None if a model has no price
    # NEEDS_HUMAN rows with a Fusion vendor: SAP open + Fusion open of the pair a person must
    # look at. Kept apart from dollars_found; the two are never added together.
    dollars_under_review: Decimal = Decimal("0")
    flag_rows_by: Counter = field(default_factory=Counter)    # rows planted NEEDS_HUMAN with that flag
    flag_caught_by: Counter = field(default_factory=Counter)  # of those, NEEDS_HUMAN (or allowed NEW) with that flag

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
        m.routing[d.route] += 1
        if d.trace is not None:
            for c in d.trace.calls:
                (m.cache_hits if c.cached else m.calls)[c.tier] += 1
                m.max_tokens_stops[c.tier] += c.stop_reason == "max_tokens"
            m.cost = None if m.cost is None or d.trace.cost is None else m.cost + d.trace.cost
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
            if d.fusion_vendor_id is not None:
                m.dollars_under_review += records.pair_open(k, d.fusion_vendor_id)
        target = lab["expected_fusion_vendor_id"]
        if lab["expected_outcome"] == MATCH:
            m.duplicates += 1
            exposed = records.pair_open(k, target)
            m.dollars_planted += exposed
            if d.outcome == MATCH and d.fusion_vendor_id == target:
                m.duplicates_found += 1
                m.dollars_found += exposed
            if d.outcome == NEW:
                m.duplicates_sent_new.append(k)
        if lab["expected_outcome"] == NEW:
            m.new_expected += 1
            m.new_correct += d.outcome == NEW
        # plan.json: count the rows planted as NEEDS_HUMAN with flags (a LOOKALIKE planted as NEW
        # needs no flag). The flag is caught if it is on the row and the row ends NEEDS_HUMAN, or
        # NEW where the plan allows NEW (the judge said DIFFERENT; the flag stays as information).
        if lab["expected_flags"] and lab["expected_outcome"] == NEEDS_HUMAN:
            m.flag_rows += 1
            shown = d.outcome == NEEDS_HUMAN or (d.outcome == NEW and NEW in lab["allowed_outcomes"])
            m.flag_caught += shown and set(lab["expected_flags"]) <= set(d.flags)
            for f in lab["expected_flags"]:
                m.flag_rows_by[f] += 1
                m.flag_caught_by[f] += shown and f in d.flags
    return m


def _cost_line(m: Metrics) -> str:
    if m.cost is None:
        return "Cost: unknown (a model has no price in config.PRICES)"
    per = m.cost / m.n if m.n else Decimal("0")
    return f"Cost: total ${m.cost:.6f}, per SAP vendor ${per:.6f}"


def summary_lines(m: Metrics) -> list[str]:
    return [
        f"Wrong merges (target 0): {len(m.wrong_merges)}" + (f"  {', '.join(m.wrong_merges)}" if m.wrong_merges else ""),
        f"Duplicate recall: {m.duplicates_found}/{m.duplicates} = {_pct(m.recall)}",
        f"Match precision: {m.correct_matches}/{m.matches_made} = {_pct(m.precision)}",
        f"Duplicates sent to NEW (missed, could be paid twice): {len(m.duplicates_sent_new)}",
        f"NEW accuracy: {m.new_correct}/{m.new_expected} = {_pct(_ratio(m.new_correct, m.new_expected))}",
        f"Outcome allowed by plan: {m.allowed_ok}/{m.n} = {_pct(_ratio(m.allowed_ok, m.n))}",
        f"NEEDS_HUMAN rate: {m.needs_human}/{m.n} = {_pct(_ratio(m.needs_human, m.n))}",
        f"Fraud / conflict / ambiguous flags caught (rows planted NEEDS_HUMAN; NEEDS_HUMAN or allowed NEW with the flag):{m.flag_caught}/{m.flag_rows}"
        + " (" + ", ".join(f"{f} {m.flag_caught_by[f]}/{m.flag_rows_by[f]}" for f in FLAG_TYPES) + ")",
        "Routing share (who decided): " + ", ".join(
            f"{r} {m.routing[r]} ({_pct(_ratio(m.routing[r], m.n))})" for r in ROUTES),
        "Model calls per tier: " + ", ".join(
            f"{t} {m.calls[t]} (cache hits {m.cache_hits[t]}, max_tokens stops {m.max_tokens_stops[t]})"
            for t in TIERS),
        _cost_line(m),
        f"Dollars exposed (duplicates suggested MATCH): found ${m.dollars_found:,.2f} of planted ${m.dollars_planted:,.2f}",
        f"Dollars under review (NEEDS_HUMAN pairs, a person decides; not added to the above): ${m.dollars_under_review:,.2f}",
    ]


def _money(x: Decimal | None) -> str | None:
    return None if x is None else str(x)


def metrics_dict(m: Metrics, method: str, split: str, extra: dict) -> dict:
    """The numbers behind the md report, as plain JSON (compare.py reads this, never the md).
    Money is written as a string so the Decimal comes back exactly."""
    per_vendor = None if m.cost is None else (m.cost / m.n if m.n else Decimal("0"))
    return {
        "method": method, "split": split, "n": m.n,
        "wrong_merges": len(m.wrong_merges), "wrong_merge_ids": m.wrong_merges,
        "duplicates": m.duplicates, "duplicates_found": m.duplicates_found, "recall": m.recall,
        "duplicates_sent_new": len(m.duplicates_sent_new),
        "matches_made": m.matches_made, "correct_matches": m.correct_matches, "precision": m.precision,
        "new_expected": m.new_expected, "new_correct": m.new_correct,
        "new_accuracy": _ratio(m.new_correct, m.new_expected),
        "allowed_ok": m.allowed_ok,
        "needs_human": m.needs_human, "needs_human_rate": _ratio(m.needs_human, m.n),
        "flags": {f: {"caught": m.flag_caught_by[f], "planted": m.flag_rows_by[f]} for f in FLAG_TYPES},
        "routing": {r: m.routing[r] for r in ROUTES},
        "calls": {t: m.calls[t] for t in TIERS},
        "cache_hits": {t: m.cache_hits[t] for t in TIERS},
        "max_tokens_stops": {t: m.max_tokens_stops[t] for t in TIERS},
        "cost_total": _money(m.cost), "cost_per_vendor": _money(per_vendor),
        "dollars_exposed_found": _money(m.dollars_found),
        "dollars_exposed_planted": _money(m.dollars_planted),
        "dollars_under_review": _money(m.dollars_under_review),
        **extra,
    }


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
             data_dir: Path | None = None, judge: Judge | None = None) -> tuple[Metrics, Path]:
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}")
    rows = select_rows(plan, split, limit)
    t_low, t_high, source = config.thresholds(embedder_kind, data_dir)
    uses_embedder = method == "embed" or method in MODEL_METHODS
    decisions = run_method(method, records, rows, embedder if uses_embedder else None, t_low, t_high, judge)
    m = score(decisions, plan, records)
    header = [f"Rows: {len(rows)}" + (f" (limit {limit})" if limit else "")]
    if uses_embedder:
        header += [f"Embedder: {embedder_kind} ({embedder.name})",
                   f"Thresholds: T_LOW {t_low:.2f}, T_HIGH {t_high:.2f} from {source}",
                   f"Embeddings computed {embedder.computed}, cache hits {embedder.hits}"]
        if embedder_kind == "fake":
            header.append("FakeEmbedder numbers are for tests only; real numbers come from --embedder local.")
    if method in MODEL_METHODS:
        header += [f"Models: cheap {judge.cheap}, strong {judge.strong}; prompt {PROMPT_VERSION}; "
                   f"max_tokens {config.MAX_TOKENS}",
                   "Response cache: " + ("on" if judge.cache.use_cache else "off (--no-cache)")]
    elif method == "fuzzy":
        header.append("difflib ratio >= 0.90 on normalized names, same country")
    path = Path(reports_dir or config.REPORTS_DIR) / f"eval_{method}_{split}.md"
    write_report(path, method, split, header, m, decisions, plan)
    extra = {"limit": limit, "embedder": embedder_kind if uses_embedder else None,
             "embedder_name": embedder.name if uses_embedder else None,
             "t_low": t_low if uses_embedder else None, "t_high": t_high if uses_embedder else None,
             "thresholds_source": source if uses_embedder else None,
             "embeddings_computed": embedder.computed if uses_embedder else None,
             "embeddings_cached": embedder.hits if uses_embedder else None}
    path.with_suffix(".json").write_text(
        json.dumps(metrics_dict(m, method, split, extra), indent=2) + "\n", encoding="utf-8")
    return m, path
