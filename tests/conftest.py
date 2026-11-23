import csv
from pathlib import Path

import pytest

from suppmatch import config
from suppmatch.db import load_all
from suppmatch.generator import generate, load_plan


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


@pytest.fixture(scope="session")
def built(tmp_path_factory):
    """Generate the data once per test run into a temp folder and load DuckDB."""
    root = tmp_path_factory.mktemp("build")
    csv_dir, plan_path, db_path = root / "csv", root / "plan.json", root / "test.duckdb"
    generate(csv_dir, plan_path, config.SEED)
    load_all(csv_dir, db_path)
    tables = {p.stem: read_csv(p) for p in sorted(csv_dir.glob("*.csv"))}
    return {"root": root, "csv_dir": csv_dir, "plan_path": plan_path, "db_path": db_path,
            "plan": load_plan(plan_path), "tables": tables}
