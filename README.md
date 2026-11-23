# supplier-match-llm

A company is moving its suppliers from SAP into Oracle Fusion. Many SAP vendors already exist in
Fusion under slightly different names, addresses and formats. Loading a duplicate lets the same
invoice be paid twice; merging two different companies sends payments to the wrong one.

For each SAP vendor this tool **suggests** one of:
- **MATCH** to one Fusion supplier (a likely duplicate: do not load it again),
- **NEW** (no Fusion counterpart),
- **NEEDS_HUMAN** (a person must decide),

with reasons, flags and dollar amounts, and writes a merge proposal for the Supplier Master
team. **It never merges, creates, updates or pays anything.** The team decides.

Synthetic data only. USD only. Local DuckDB only. The design is in [docs/design.md](docs/design.md).

## The four AI ideas

1. **Embeddings.** Names and addresses become vectors (small local model,
   `all-MiniLM-L6-v2`), so "Acme Industries Inc" and "ACME INDUSTRIES, INCORPORATED" end up close.
   Candidates are the top 5 Fusion suppliers in the same country by cosine similarity (numpy,
   no vector database).
2. **Threshold tuning.** Two numbers split the scores into bands, tuned on the dev split only and
   reported on the held-out split:
   - **T_HIGH** is set for precision: the highest share of duplicates matched by rule while dev
     match precision stays at or above 98%.
   - **T_LOW** is set so no duplicate is missed: as high as possible while no dev duplicate scores
     below it against its real Fusion supplier.
3. **Model routing, cheap first.** Python rules -> embedding score bands -> Claude Haiku for the
   unclear middle -> Claude Sonnet only when Haiku says UNSURE -> NEEDS_HUMAN when Sonnet is also
   UNSURE. One call per step, no tools, no agent loop. `sonnet-only` sends the same middle band
   straight to Sonnet so the cost difference can be measured.
4. **Embedding cache.** The key is sha256(model name + normalized text), so the same text is
   never embedded twice. Model answers are cached too (key = model + prompt version + packet);
   errors and invalid answers are never cached.

## Decision rules and owners

Python owns every decision. The model only returns SAME / DIFFERENT / UNSURE plus exact quotes.

| Rule | What happens | Model? |
|---|---|---|
| Same bank account + different tax ID | NEEDS_HUMAN, **FRAUD_FLAG** | no |
| Tax ID SAME and (bank SAME or score >= T_HIGH) | MATCH | no |
| Two Fusion suppliers both look right | NEEDS_HUMAN, AMBIGUOUS | no |
| Tax ID DIFFERENT or bank DIFFERENT | never MATCH; at most NEEDS_HUMAN with TAX_CONFLICT / BANK_CONFLICT | only if the band calls for it, and its SAME is overruled |
| Best score < T_LOW and tax ID not SAME | NEW | no |
| Anything else (middle band) | Haiku -> Sonnet -> human | yes |

- **Supplier Master owns the thresholds** (T_LOW, T_HIGH in `config.py`, or the tuned file).
  Precision first: a wrong merge is worse than a missed duplicate.
- **Tax and bank conflicts beat the model.** A model SAME on a conflicting pair still ends
  NEEDS_HUMAN with the flag.
- **Bank identity = country + bank key + account number.** An account number is only unique
  within one bank, so the same number at a different bank is DIFFERENT.
- **The model never sees a tax ID or a bank account number.** It sees only flags
  (tax_id / bank = SAME / DIFFERENT / MISSING) and bank last4. A test scans every prompt, the
  audit log and the proposal files for every tax ID and bank number in the CSVs.
- Supplier names and addresses are treated as untrusted text (they can carry injected
  instructions); they are wrapped in `<UNTRUSTED>` with `<` and `>` escaped.

## Merge proposal and dollars

`reports/proposal_<method>_<split>.csv` and `.md`, one row per SAP vendor: SAP LIFNR, outcome,
Fusion vendor_id, flags, route (rules / Haiku / Sonnet / human), reason, survivor, field
differences (name / address / postal), bank note, and two dollar amounts:

- **Dollars exposed** = SAP open items + Fusion open AP, for suggested MATCH rows only (likely
  duplicates: the same invoice could be paid twice).
- **Dollars under review** = the same sum for NEEDS_HUMAN rows with a Fusion vendor (unresolved;
  a person decides).

The two are shown in separate columns (one is blank on each row) and in separate totals. They
are never added together. The survivor is always the Fusion record (system of record). A SAP bank
account is never copied to Fusion: "verify by call-back to the number on file before adding".

## The data (synthetic)

Shapes are modeled on real tables, with made-up values:

| CSV | Modeled on |
|---|---|
| fusion_suppliers | Fusion POZ_SUPPLIERS + HZ_PARTIES |
| fusion_sites | Fusion POZ_SUPPLIER_SITES_ALL_M |
| fusion_bank | Fusion IBY_EXT_BANK_ACCOUNTS |
| fusion_open_ap | Fusion AP_PAYMENT_SCHEDULES_ALL |
| sap_lfa1 | SAP LFA1 (vendor master) |
| sap_lfbk | SAP LFBK (vendor bank) |
| sap_open_items | SAP BSIK (open items) |

About 200 Fusion suppliers and 120 SAP vendors (US, CA, GB, DE, IN) in planted groups (EXACT,
FORMAT, FUZZY, RENAMED, NEW, LOOKALIKE, BANKCONFLICT, SHAREDBANK, AMBIGUOUS), plus 8 red-team
vendors (prompt injection, homoglyphs, fake closing tags, shared bank). Every SAP vendor has a
planted answer in `data/plan.json`; that file is the only source of eval labels.

## Results

Wrong merges first: the target is 0.

### Baselines (fake embedder, sandbox)

These come from the development sandbox with the FakeEmbedder (character n-grams, used for
tests) and thresholds tuned on dev for it (T_LOW 0.68, T_HIGH 0.99). They show the shape of the
results, not real model quality. exact and fuzzy do not use embeddings.

| Held-out split (58 SAP vendors) | exact | fuzzy | embed |
|---|---|---|---|
| Wrong merges | 0 | 0 | 0 |
| Duplicate recall | 14/24 (58.3%) | 18/24 (75.0%) | 17/24 (70.8%) |
| Match precision | 100.0% | 100.0% | 100.0% |
| NEW accuracy | 25/25 (100.0%) | 25/25 (100.0%) | 23/25 (92.0%) |
| NEEDS_HUMAN rate | 7/58 (12.1%) | 7/58 (12.1%) | 16/58 (27.6%) |
| Flags caught FRAUD / TAX / BANK / AMBIGUOUS | 0/2, 2/2, 3/3, 2/2 | 0/2, 2/2, 3/3, 2/2 | 2/2, 0/2, 3/3, 2/2 |
| Dollars exposed: found / planted | $336,217.78 / $557,223.80 | $394,835.33 / $557,223.80 | $422,880.56 / $557,223.80 |
| Dollars under review | $180,361.83 | $180,361.83 | $381,498.96 |

| Dev split (62 SAP vendors) | exact | fuzzy | embed |
|---|---|---|---|
| Wrong merges | 0 | 0 | 0 |
| Duplicate recall | 16/27 (59.3%) | 22/27 (81.5%) | 19/27 (70.4%) |
| Match precision | 100.0% | 100.0% | 100.0% |
| NEW accuracy | 25/25 (100.0%) | 25/25 (100.0%) | 23/25 (92.0%) |
| NEEDS_HUMAN rate | 8/62 (12.9%) | 8/62 (12.9%) | 17/62 (27.4%) |

Red-team split (8 vendors): 0 wrong merges for exact, fuzzy and embed.

Notes on these numbers: embed-only sends the whole middle band to a person, so its NEEDS_HUMAN
rate is higher. With the fake embedder the planted TAX_CONFLICT lookalikes score below T_LOW and
end as NEW (allowed by the plan: never MATCH), so the flag is not raised; exact and fuzzy raise it
because the names are equal.

### Real runs (local embedding model + Claude)

| Held-out split | exact | fuzzy | embed | routed | sonnet-only |
|---|---|---|---|---|---|
| Wrong merges | fill in after local run | fill in after local run | fill in after local run | fill in after local run | fill in after local run |
| Duplicate recall | fill in after local run | fill in after local run | fill in after local run | fill in after local run | fill in after local run |
| Match precision | fill in after local run | fill in after local run | fill in after local run | fill in after local run | fill in after local run |
| NEEDS_HUMAN rate | fill in after local run | fill in after local run | fill in after local run | fill in after local run | fill in after local run |
| Routing share (rules / Haiku / Sonnet / human) | fill in after local run | fill in after local run | fill in after local run | fill in after local run | fill in after local run |
| Cost per SAP vendor | $0 | $0 | $0 | fill in after local run | fill in after local run |
| Dollars exposed: found / planted | fill in after local run | fill in after local run | fill in after local run | fill in after local run | fill in after local run |
| Dollars under review | fill in after local run | fill in after local run | fill in after local run | fill in after local run | fill in after local run |

`reports/compare_heldout.md` gives the full table after the runs below.

## Run it on Windows (PowerShell)

Setup (once):

```powershell
py -3.14 -m venv .venv
Set-ExecutionPolicy -Scope Process Bypass
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e ".[embed]"
```

Build the data (CSVs, `data/plan.json`, DuckDB):

```powershell
python -m suppmatch build
```

Tune the thresholds for the real embedding model, on the dev split only. **Do this before any
real eval**; until then the placeholders in `config.py` are used:

```powershell
python -m suppmatch tune --embedder local
```

Baselines (no API key needed):

```powershell
python -m suppmatch eval --method exact --split heldout
python -m suppmatch eval --method fuzzy --split heldout
python -m suppmatch eval --method embed --split heldout --embedder local
```

Claude runs cost real money. Read the key without typing it into a command (it stays in this
PowerShell window only):

```powershell
$key = Read-Host "Anthropic API key" -AsSecureString
$env:ANTHROPIC_API_KEY = [System.Net.NetworkCredential]::new("", $key).Password
```

Then (try `--limit 10` first):

```powershell
python -m suppmatch eval --method routed --split dev --embedder local --limit 10
python -m suppmatch eval --method routed --split heldout --embedder local
python -m suppmatch eval --method sonnet-only --split heldout --embedder local
```

Compare all methods side by side, and write the merge proposal:

```powershell
python -m suppmatch compare --split heldout
python -m suppmatch propose --method routed --split heldout --embedder local
```

When done:

```powershell
Remove-Item Env:ANTHROPIC_API_KEY
```

What each command writes:
- `eval`: `reports/eval_<method>_<split>.md` and a `.json` with the same numbers.
- `compare`: `reports/compare_<split>.md`, built from the `.json` files; a method with no eval
  shows "not run".
- `propose`: `reports/proposal_<method>_<split>.csv` and `.md` (the second run uses the response
  cache, so it makes 0 new model calls).
- `logs/audit_<method>_<split>.jsonl`: ids, hashes, tier, tokens, cost, outcome (no names, no
  tax or bank numbers). `logs/suppmatch.log`: full error messages.

Options: `--split dev|heldout|redteam`, `--limit N`, `--embedder local|fake`, `--no-cache`.
Model names: `SUPPMATCH_MODEL_CHEAP` (default claude-haiku-5-5) and `SUPPMATCH_MODEL_STRONG`
(default claude-sonnet-5-5). Prices per million tokens are in `config.PRICES`.

Run the tests (no API key, no model download; they use a scripted fake client and the
FakeEmbedder):

```powershell
python -m pytest -q
```

## Honest notes

- The data is synthetic and the right answers were planted by the author. Real supplier data is
  messier, and the groups here are cleaner than real life.
- The embedding model is small and runs locally. A bigger model may separate lookalikes better.
- Thresholds are tuned on the dev split only; the held-out and red-team splits are never used for
  tuning. With about 60 rows per split, one row moves a percentage by 1-2 points.
- The sandbox numbers above use a test embedder, not the real model. The real-run table is empty
  until it is run on a laptop with the model and an API key.
- The output is a suggestion. Nothing here touches SAP or Fusion.

## Layout

```
src/suppmatch/
  config.py      thresholds (owned by Supplier Master), model names, prices, paths
  generator.py   synthetic CSVs + plan.json (fixed seed)
  db.py          load CSVs into DuckDB with real types
  normalize.py   name / address / tax ID / bank normalization, embedding text
  records.py     DuckDB tables -> normalized vendor records + open amounts (Decimal)
  embed.py       LocalEmbedder (sentence-transformers, lazy), FakeEmbedder (tests), cache
  candidates.py  same-country blocking, top-5 by cosine, tax / bank / postal flags
  decide.py      decision rules; apply_verdict turns the judge's verdict into an outcome
  packet.py      what the model sees (UNTRUSTED-wrapped fields, features, flags, last4)
  schema.py      the model's answer (pydantic, extra="forbid")
  guards.py      evidence quotes must be exact substrings of real fields
  judge.py       Haiku -> Sonnet routing, one retry, response cache, cost
  audit.py       JSON-lines audit trail
  baselines.py   exact, fuzzy (difflib >= 0.90), embed-only
  tune.py        T_HIGH / T_LOW grid on the dev split only
  evaluate.py    metrics vs plan.json, reports/eval_<method>_<split>.md + .json
  proposal.py    merge proposal csv + md, dollars exposed / under review
  compare.py     all methods side by side from the eval .json files
  __main__.py    command line (build, tune, eval, propose, compare)
tests/fakes.py   scripted fake Claude client (tests never call the real API)
tests/
docs/design.md
```
