"""Settings in one place. Supplier Master owns the thresholds."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Data generation -------------------------------------------------------
SEED = 42

# --- Matching thresholds (owned by Supplier Master) ------------------------
# Placeholders. PR 2 tunes them on the dev split only (precision first:
# a wrong merge is worse than a missed duplicate) and reports held-out numbers.
T_LOW = 0.60   # best full_cosine below this (and tax_id not SAME) -> NEW
T_HIGH = 0.90  # tax_id SAME and full_cosine at or above this -> MATCH
TOP_K = 5      # Fusion candidates kept per SAP vendor

# --- Models ----------------------------------------------------------------
MODEL_CHEAP = os.getenv("SUPPMATCH_MODEL_CHEAP") or "claude-haiku-5-5"
MODEL_STRONG = os.getenv("SUPPMATCH_MODEL_STRONG") or "claude-sonnet-5-5"
EMBED_MODEL = os.getenv("SUPPMATCH_EMBED_MODEL") or "sentence-transformers/all-MiniLM-L6-v2"
MAX_TOKENS = 2000

# --- Paths -----------------------------------------------------------------
BASE_DIR = Path(os.getenv("SUPPMATCH_HOME") or ".").resolve()
DATA_DIR = BASE_DIR / "data"
CSV_DIR = DATA_DIR / "csv"
PLAN_PATH = DATA_DIR / "plan.json"
DB_PATH = DATA_DIR / "suppmatch.duckdb"
CACHE_DIR = DATA_DIR / "cache"
REPORTS_DIR = BASE_DIR / "reports"
