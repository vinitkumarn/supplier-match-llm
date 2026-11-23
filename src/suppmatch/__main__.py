"""Command line: python -m suppmatch build"""

from __future__ import annotations

import argparse
from collections import Counter

from . import config
from .db import load_all
from .generator import generate, load_plan


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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m suppmatch")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="generate synthetic CSVs + plan.json and load DuckDB")
    build.add_argument("--seed", type=int, default=config.SEED)
    build.set_defaults(func=cmd_build)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
