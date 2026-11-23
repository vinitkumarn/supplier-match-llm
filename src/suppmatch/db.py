"""Load the CSVs into a local DuckDB file with real column types."""

from __future__ import annotations

from pathlib import Path

import duckdb

# IDs, tax IDs, bank keys and account numbers stay VARCHAR so leading zeros and
# the original formatting survive. Money is DECIMAL(18,2), never float.
SCHEMAS: dict[str, dict[str, str]] = {
    "fusion_suppliers": {
        "vendor_id": "BIGINT", "party_id": "BIGINT", "vendor_name": "VARCHAR",
        "segment1": "VARCHAR", "tax_id": "VARCHAR", "country": "VARCHAR",
    },
    "fusion_sites": {
        "vendor_id": "BIGINT", "address_line1": "VARCHAR", "city": "VARCHAR",
        "state": "VARCHAR", "postal_code": "VARCHAR", "country": "VARCHAR",
    },
    "fusion_bank": {
        "vendor_id": "BIGINT", "country": "VARCHAR", "bank_key": "VARCHAR",
        "bank_name": "VARCHAR", "account_number": "VARCHAR", "last4": "VARCHAR",
    },
    "fusion_open_ap": {
        "vendor_id": "BIGINT", "invoice_num": "VARCHAR", "amount_remaining": "DECIMAL(18,2)",
    },
    "sap_lfa1": {
        "LIFNR": "VARCHAR", "NAME1": "VARCHAR", "NAME2": "VARCHAR", "STRAS": "VARCHAR",
        "ORT01": "VARCHAR", "REGIO": "VARCHAR", "PSTLZ": "VARCHAR", "LAND1": "VARCHAR",
        "STCD1": "VARCHAR",
    },
    "sap_lfbk": {"LIFNR": "VARCHAR", "BANKS": "VARCHAR", "BANKL": "VARCHAR", "BANKN": "VARCHAR"},
    "sap_open_items": {"LIFNR": "VARCHAR", "BELNR": "VARCHAR", "DMBTR": "DECIMAL(18,2)"},
}


def _sql_str(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def load_all(csv_dir: Path, db_path: Path) -> dict[str, int]:
    """(Re)create every table from its CSV. Returns row counts per table."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    with duckdb.connect(str(db_path)) as con:
        for table, cols in SCHEMAS.items():
            path = (Path(csv_dir) / f"{table}.csv").resolve().as_posix()
            columns = "{" + ", ".join(f"{_sql_str(c)}: {_sql_str(t)}" for c, t in cols.items()) + "}"
            con.execute(
                f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM read_csv("
                f"{_sql_str(path)}, header = true, columns = {columns}, "
                f"quote = '\"', escape = '\"', encoding = 'utf-8')"
            )
            counts[table] = con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
    return counts


def connect(db_path: Path, read_only: bool = True) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(db_path), read_only=read_only)
