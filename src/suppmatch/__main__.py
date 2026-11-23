"""Command line: python -m suppmatch build | tune | eval"""

from __future__ import annotations

import argparse
import sys
from collections import Counter

from . import config
from .db import load_all
from .embed import get_embedder
from .evaluate import METHODS, SPLITS, evaluate, summary_lines
from .generator import generate, load_plan
from .records import load_records
from .tune import tune, write_outputs


def cmd_build(args: argparse.Namespace) -> None:
    csv_counts = generate(config.CSV_DIR, config.PLAN_PATH, args.seed)
    db_counts = load_all(config.CSV_DIR, config.DB_PATH)
    print(f"Wrote CSVs to {config.CSV_DIR} and plan to {config.PLAN_PATH} (seed {args.seed})")
    print(f"Loaded DuckDB {config.DB_PATH}")
    for table, n in db_counts.items():
        print(f"  {table:<18} {n:>5} rows (csv {csv_counts[table]})")
    vendors = load_plan(config.PLAN_PATH)["vendors"].values()
    print("SAP vendors by group / split:")
    pairs = Counter((v["group"], v["split"]) for v in vendors)
    for group in sorted({g for g, _ in pairs}):
        row = ", ".join(f"{s}={pairs[(group, s)]}" for s in ("dev", "heldout", "redteam") if pairs[(group, s)])
        print(f"  {group:<13} {row}")


def _inputs():
    if not config.DB_PATH.exists() or not config.PLAN_PATH.exists():
        sys.exit("No data yet. Run: python -m suppmatch build")
    return load_records(config.DB_PATH), load_plan(config.PLAN_PATH)


def cmd_tune(args: argparse.Namespace) -> None:
    records, plan = _inputs()
    embedder = get_embedder(args.embedder, use_cache=not args.no_cache)
    try:
        result, rows, high, low = tune(records, plan, embedder, args.embedder)
    except RuntimeError as exc:
        sys.exit(str(exc))
    json_path, md_path = write_outputs(result, high, low)
    print(f"Tuned on {len(rows)} dev rows with embedder {args.embedder} ({embedder.name})")
    if not result.target_met:
        print(f"WARNING: no T_HIGH reaches dev precision >= 0.98; chose the most precise one.")
    print(f"  T_HIGH {result.t_high:.2f}: rule MATCHes {result.rule_matches}, precision "
          f"{result.precision if result.precision is None else f'{result.precision:.1%}'}, "
          f"recall {result.recall:.1%}")
    print(f"  T_LOW  {result.t_low:.2f}: duplicates below T_LOW {result.duplicates_below_t_low}, "
          f"middle band {result.middle_band}/{result.dev_rows}")
    print(f"  embeddings computed {embedder.computed}, cache hits {embedder.hits}")
    print(f"Wrote {json_path} and {md_path}")
    if args.embedder == "fake":
        print("FakeEmbedder thresholds are for tests only. Run 'python -m suppmatch tune --embedder local' before a real eval.")


def cmd_eval(args: argparse.Namespace) -> None:
    records, plan = _inputs()
    embedder = get_embedder(args.embedder, use_cache=not args.no_cache) if args.method == "embed" else None
    try:
        m, path = evaluate(args.method, args.split, records, plan, embedder, args.embedder, args.limit)
    except RuntimeError as exc:
        sys.exit(str(exc))
    print(f"{args.method} on {args.split} ({m.n} rows)")
    if embedder is not None:
        t_low, t_high, source = config.thresholds(args.embedder)
        print(f"  embedder {args.embedder}; T_LOW {t_low:.2f} T_HIGH {t_high:.2f} from {source}")
        print(f"  embeddings computed {embedder.computed}, cache hits {embedder.hits}")
    for line in summary_lines(m):
        print("  " + line)
    print(f"Wrote {path}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m suppmatch")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="generate synthetic CSVs + plan.json and load DuckDB")
    build.add_argument("--seed", type=int, default=config.SEED)
    build.set_defaults(func=cmd_build)

    tune_p = sub.add_parser("tune", help="tune T_LOW / T_HIGH on the dev split only")
    tune_p.add_argument("--embedder", choices=["local", "fake"], default="local")
    tune_p.add_argument("--no-cache", action="store_true", help="do not read or write the embedding cache")
    tune_p.set_defaults(func=cmd_tune)

    ev = sub.add_parser("eval", help="score a method against the planted answers")
    ev.add_argument("--method", choices=METHODS, required=True)
    ev.add_argument("--split", choices=SPLITS, default="dev")
    ev.add_argument("--limit", type=int, default=None, help="only the first N SAP vendors of the split")
    ev.add_argument("--embedder", choices=["local", "fake"], default="local")
    ev.add_argument("--no-cache", action="store_true", help="do not read or write the embedding cache")
    ev.set_defaults(func=cmd_eval)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
