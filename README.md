# supplier-match-llm

A company is moving suppliers from SAP into Oracle Fusion. Many SAP vendors already exist in
Fusion with slightly different names, addresses and formats. For each SAP vendor this tool
**suggests** MATCH (to one Fusion supplier), NEW, or NEEDS_HUMAN, with reasons, flags and the
dollars exposed to double payment. The Supplier Master team decides. The tool never merges,
creates, updates or pays anything.

Synthetic data only. USD only. Local DuckDB only. The design is in [docs/design.md](docs/design.md).

## Status
- PR 1 (this): project skeleton, synthetic data with planted answers (`data/plan.json`),
  DuckDB loading, normalization rules.
- Later PRs: embeddings + candidates + threshold tuning, model routing (Haiku -> Sonnet),
  proposal files, eval.

## Setup on Windows (PowerShell)

```powershell
py -3.14 -m venv .venv
Set-ExecutionPolicy -Scope Process Bypass
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

Optional, for the real embedding model (later PRs; not needed for tests):

```powershell
python -m pip install -e ".[embed]"
```

Copy `.env.example` to `.env` and fill in `ANTHROPIC_API_KEY` when you run real evals
(later PRs). Never commit `.env`.

## Build the data

```powershell
python -m suppmatch build
```

This writes:
- `data/csv/*.csv` — Fusion shapes (`fusion_suppliers`, `fusion_sites`, `fusion_bank`,
  `fusion_open_ap`) and SAP shapes (`sap_lfa1`, `sap_lfbk`, `sap_open_items`)
- `data/plan.json` — the planted answer for every SAP vendor (group, split, expected outcome,
  expected Fusion vendor, expected flags). This is the only source of eval labels.
- `data/suppmatch.duckdb` — the CSVs as typed tables (money is `DECIMAL(18,2)`).

Same seed, same files: `python -m suppmatch build --seed 42` always gives identical output.

## Run the tests

```powershell
python -m pytest -q
```

## Layout

```
src/suppmatch/
  config.py      thresholds (owned by Supplier Master), model names, paths
  generator.py   synthetic CSVs + plan.json (fixed seed)
  db.py          load CSVs into DuckDB with real types
  normalize.py   name / address / tax ID / bank normalization, embedding text
  __main__.py    command line (python -m suppmatch build)
tests/
docs/design.md
```
