# Supplier Match (LLM + embeddings) — project rules

## What this is
A portfolio project by Vinit (Oracle ERP techno-functional, 12+ years AP/P2P; owner of Oracle
Fusion CDM and EDQ). A company is moving suppliers from SAP into Oracle Fusion. Many suppliers
exist in both systems with slightly different names, addresses and formats. For each SAP vendor
the tool suggests MATCH (to one Fusion supplier), NEW (no match) or NEEDS_HUMAN, with reasons,
flags and dollars exposed to double payment. It SUGGESTS a merge proposal; the Supplier Master
team decides. It never merges, creates, updates or pays anything. Synthetic data only. USD only.
Local DuckDB only.

The design is in docs/design.md. Follow it. If something in it seems wrong, say so in the PR
description instead of silently changing it.

## Keep it simple (on purpose)
- No data pipeline: CSVs load into DuckDB tables, Python does the rest in memory.
- Embeddings: a small local model (sentence-transformers all-MiniLM-L6-v2), loaded lazily.
  Tests use a deterministic FakeEmbedder. No vector database: numpy cosine on a few hundred rows.
- Routing: Python rules -> embedding score bands -> Claude Haiku -> Claude Sonnet only when
  Haiku says UNSURE -> NEEDS_HUMAN when Sonnet is also UNSURE. One call per step, no tools,
  no agent loop.
- Python owns every decision: MATCH / NEW / NEEDS_HUMAN, flags, survivorship notes, dollars.
- Prefer the simplest code that meets the design. Do not add features that are not in it.

## Non-negotiable design rules
- Never auto-merge. Output is a proposal file for review.
- Tax IDs and bank account numbers never reach the model. The model sees only flags:
  tax_id = SAME / DIFFERENT / MISSING, bank = SAME / DIFFERENT / MISSING (plus bank last4).
  A test scans every prompt and fails if any tax ID or bank number from the CSVs appears.
- Hard conflicts beat the model: bank DIFFERENT or tax_id DIFFERENT -> never MATCH; at most
  NEEDS_HUMAN with the flag. Bank SAME + tax_id DIFFERENT -> FRAUD_FLAG, NEEDS_HUMAN, no model.
- The model returns a verdict (SAME / DIFFERENT / UNSURE) plus evidence quotes only
  (pydantic, extra="forbid"). Every evidence quote must be an exact substring of one of the two
  records' text fields. Any failure -> NEEDS_HUMAN.
- Supplier name and address fields are UNTRUSTED (they can carry injected text): wrap them as
  <UNTRUSTED>...</UNTRUSTED> with < and > escaped.
- Thresholds (T_LOW, T_HIGH) live in config.py; Supplier Master owns them. They are tuned on the
  dev split only and reported on the held-out split. Precision first: a wrong merge is worse
  than a missed one.
- Eval labels come from the planted plan file, NEVER from rules code. The held-out split and
  the red-team set are never used for tuning.
- Money is Decimal, never float.
- Caching: (1) embedding cache keyed by sha256(embed model name + normalized text): the same
  text is never embedded twice; (2) response cache keyed by sha256(model + prompt version +
  packet); never cache errors or invalid answers; guards run on hits; --no-cache.
- LLM: Anthropic Claude. SUPPMATCH_MODEL_CHEAP (default claude-haiku-5-5),
  SUPPMATCH_MODEL_STRONG (default claude-sonnet-5-5). max_tokens 4000 (thinking counts toward it), system prompt with
  cache_control, one retry with the full validation error, full error messages in logs.

## Working rules
- Never commit .env or any secret. .gitignore covers .env, .venv/, *.db, *.duckdb, logs/,
  *.egg-info/, data/csv/, data/*.json, data/cache/, reports/, __pycache__/, .pytest_cache/.
- There is no API key in this sandbox and the embedding model may not download here. Never call
  the real Claude API here. Use the scripted fake client and the FakeEmbedder in tests.
  Vinit runs real evals on his laptop.
- One milestone per PR. All tests must pass before you stop.
- Every PR description must include, in simple everyday language:
  1. What this PR adds and why
  2. A short "what to review" checklist (the 3-5 things most worth checking)
  3. Controls/accounting notes where relevant (wrong merge vs missed duplicate, dollars exposed,
     bank and tax conflicts, who owns the thresholds, grey areas and the default)
  4. How to test it on Windows PowerShell: py launcher (Python 3.14), venv activated with
     Set-ExecutionPolicy -Scope Process Bypass then .venv\Scripts\Activate.ps1, python -m pip
  5. Any new Python concept explained in 2-3 plain sentences
