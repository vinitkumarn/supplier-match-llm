"""Settings in one place. Supplier Master owns the thresholds."""

from __future__ import annotations

import json
import os
from decimal import Decimal
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Data generation -------------------------------------------------------
SEED = 42

# --- Matching thresholds (owned by Supplier Master) ------------------------
# Placeholders, used only until `python -m suppmatch tune --embedder <name>` has written
# data/thresholds_<name>.json (tuned on the dev split only; precision first: a wrong merge
# is worse than a missed duplicate). Each embedder has its own scale, so its own file.
T_LOW = 0.60   # best full_cosine below this (and tax_id not SAME) -> NEW
T_HIGH = 0.90  # tax_id SAME and full_cosine at or above this -> MATCH
TOP_K = 5      # Fusion candidates kept per SAP vendor

# --- Models ----------------------------------------------------------------
MODEL_CHEAP = os.getenv("SUPPMATCH_MODEL_CHEAP") or "claude-haiku-5-5"
MODEL_STRONG = os.getenv("SUPPMATCH_MODEL_STRONG") or "claude-sonnet-5-5"
EMBED_MODEL = os.getenv("SUPPMATCH_EMBED_MODEL") or "sentence-transformers/all-MiniLM-L6-v2"
MAX_TOKENS = 4000  # per call; thinking tokens count toward it

# USD per million tokens (Decimal, never float). A model not listed here -> cost None.
PRICES: dict[str, dict[str, Decimal]] = {
    "claude-haiku-5-5": {"input": Decimal("0.10"), "output": Decimal("0.50"),
                         "cache_read": Decimal("0.01"), "cache_write": Decimal("0.125")},
    "claude-sonnet-5-5": {"input": Decimal("2.00"), "output": Decimal("10.00"),
                          "cache_read": Decimal("0.10"), "cache_write": Decimal("2.50")},
}


def cost_usd(model: str, input_tokens: int, output_tokens: int, cache_read: int = 0,
             cache_write: int = 0) -> Decimal | None:
    """Dollar cost of one call, or None when the model has no price in PRICES."""
    p = PRICES.get(model)
    if p is None:
        return None
    total = (input_tokens * p["input"] + output_tokens * p["output"]
             + cache_read * p["cache_read"] + cache_write * p["cache_write"])
    return total / Decimal(1_000_000)

# --- Paths -----------------------------------------------------------------
BASE_DIR = Path(os.getenv("SUPPMATCH_HOME") or ".").resolve()
DATA_DIR = BASE_DIR / "data"
CSV_DIR = DATA_DIR / "csv"
PLAN_PATH = DATA_DIR / "plan.json"
DB_PATH = DATA_DIR / "suppmatch.duckdb"
CACHE_DIR = DATA_DIR / "cache"
REPORTS_DIR = BASE_DIR / "reports"
LOG_DIR = BASE_DIR / "logs"


def thresholds_path(embedder: str, data_dir: Path | None = None) -> Path:
    return Path(data_dir or DATA_DIR) / f"thresholds_{embedder}.json"


def thresholds(embedder: str, data_dir: Path | None = None) -> tuple[float, float, str]:
    """(T_LOW, T_HIGH, source) for this embedder: the tuned file if present, else placeholders."""
    path = thresholds_path(embedder, data_dir)
    if path.exists():
        with path.open(encoding="utf-8") as fh:
            data = json.load(fh)
        return float(data["t_low"]), float(data["t_high"]), str(path.name)
    return T_LOW, T_HIGH, "placeholder (config.py)"
