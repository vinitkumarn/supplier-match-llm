import csv
from pathlib import Path

import pytest

from suppmatch import config
from suppmatch.db import load_all
from suppmatch.embed import get_embedder
from suppmatch.generator import generate, load_plan
from suppmatch.records import Vendor, load_records


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


@pytest.fixture(scope="session")
def records(built):
    return load_records(built["db_path"])


@pytest.fixture
def fake_embedder(tmp_path):
    """FakeEmbedder with its own empty cache folder."""
    return get_embedder("fake", cache_dir=tmp_path / "emb_cache")


def vendor(key, name, country="US", tax="", bank=None, address="1 Main St", city="Springfield", postal="11111"):
    """A hand-made record for unit tests (tax / bank already normalized)."""
    return Vendor(key, name, address, city, postal, country, tax, bank, bank[2][-4:] if bank else "")
