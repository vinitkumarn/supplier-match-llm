# supplier-match-llm

A company is moving suppliers from SAP into Oracle Fusion. Many SAP vendors already exist in
Fusion with slightly different names, addresses and formats. For each SAP vendor this tool
**suggests** MATCH (to one Fusion supplier), NEW, or NEEDS_HUMAN, with reasons, flags and the
dollars exposed to double payment. The Supplier Master team decides. The tool never merges,
creates, updates or pays anything.

Synthetic data only. USD only. Local DuckDB only. The design is in [docs/design.md](docs/design.md).

## Status
- PR 1: project skeleton, synthetic data with planted answers (`data/plan.json`),
  DuckDB loading, normalization rules.
- PR 2: embeddings + embedding cache, same-country candidates, decision rules, baselines
  (exact / fuzzy / embed-only), threshold tuning on the dev split, eval reports.
- PR 3: Claude judge for the unclear middle band with routing (Haiku first, Sonnet only when
  Haiku says UNSURE), evidence guards, response cache, cost per SAP vendor, audit trail.
- Later PRs: proposal files.

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

## Tune the thresholds (dev split only)

```powershell
python -m suppmatch tune --embedder local
```

Writes `data/thresholds_local.json` (read automatically by later runs with `--embedder local`)
and `reports/thresholds_local.md`. **Run this on your laptop before any real eval.**
`--embedder fake` gives thresholds for the test embedder only; they mean nothing for the real
model. Until a file exists, the placeholders in `config.py` are used.

- T_HIGH: the most duplicates matched by rule while dev match precision stays >= 98%.
- T_LOW: as high as possible while no dev duplicate scores below it against its real
  Fusion supplier (no true duplicate is thrown out as NEW).

## Evaluate a method

```powershell
python -m suppmatch eval --method exact --split dev
python -m suppmatch eval --method fuzzy --split heldout
python -m suppmatch eval --method embed --split heldout --embedder local
```

## Evaluate with Claude (needs ANTHROPIC_API_KEY in .env; costs real money)

```powershell
python -m suppmatch eval --method routed --split dev --embedder local --limit 10
python -m suppmatch eval --method routed --split heldout --embedder local
python -m suppmatch eval --method sonnet-only --split heldout --embedder local
```

- `routed`: rules first; only the unclear middle band goes to Haiku; Sonnet only when Haiku
  says UNSURE; Sonnet UNSURE -> NEEDS_HUMAN.
- `sonnet-only`: the same middle band and the same prompt, straight to Sonnet (cost comparison).
- The report adds routing share (rules / haiku / sonnet / human), calls per tier, max_tokens
  stops, and cost (total and per SAP vendor; prices per MTok in `config.PRICES`).
- Answers are cached in `data/cache/responses/` (key = sha256 of model + prompt version +
  packet), so a second run makes 0 calls. Errors and invalid answers are never cached.
- `logs/audit_<method>_<split>.jsonl`: ids, hashes, tier used, tokens, cost, outcome (no names,
  no tax or bank numbers). `logs/suppmatch.log`: full error messages.
- The model never sees a tax ID or bank account number, only SAME / DIFFERENT / MISSING flags and
  bank last4.

Options: `--split dev|heldout|redteam`, `--limit N`, `--embedder local|fake`, `--no-cache`
(no embedding or response cache).
Writes `reports/eval_<method>_<split>.md`. Labels come only from `data/plan.json`.
Embeddings are cached in `data/cache/embeddings/` (key = sha256 of model name + text), so a
second run computes 0 embeddings.

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
  records.py     DuckDB tables -> normalized vendor records + open amounts (Decimal)
  embed.py       LocalEmbedder (sentence-transformers, lazy), FakeEmbedder (tests), cache
  candidates.py  same-country blocking, top-5 by cosine, tax / bank / postal flags
  decide.py      decision rules: FRAUD, SURE MATCH, AMBIGUOUS, CONFLICT, SURE NEW, needs judge;
                 apply_verdict turns the judge's verdict into an outcome
  packet.py      what the model sees (UNTRUSTED-wrapped fields, features, flags, last4)
  schema.py      the model's answer (pydantic, extra="forbid")
  guards.py      evidence quotes must be exact substrings of real fields
  judge.py       Haiku -> Sonnet routing, one retry, response cache, cost
  audit.py       JSON-lines audit trail
  baselines.py   exact, fuzzy (difflib >= 0.90), embed-only
  tune.py        T_HIGH / T_LOW grid on the dev split only
  evaluate.py    metrics vs plan.json, reports/eval_<method>_<split>.md
  __main__.py    command line (build, tune, eval)
tests/fakes.py   scripted fake Claude client (tests never call the real API)
tests/
docs/design.md
```
