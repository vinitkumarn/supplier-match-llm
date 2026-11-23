from decimal import Decimal

from suppmatch.db import SCHEMAS, connect


def test_row_counts_match_csvs(built):
    with connect(built["db_path"]) as con:
        for table in SCHEMAS:
            n = con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
            assert n == len(built["tables"][table]), table


def test_real_types(built):
    with connect(built["db_path"]) as con:
        for table, cols in SCHEMAS.items():
            rows = con.execute(
                "SELECT column_name, data_type FROM information_schema.columns "
                "WHERE table_name = ? ORDER BY ordinal_position", [table]).fetchall()
            assert dict(rows) == cols, table


def test_money_is_decimal_and_ids_keep_leading_zeros(built):
    with connect(built["db_path"]) as con:
        amt = con.execute("SELECT DMBTR FROM sap_open_items LIMIT 1").fetchone()[0]
        assert isinstance(amt, Decimal)
        total = con.execute("SELECT sum(amount_remaining) FROM fusion_open_ap").fetchone()[0]
        assert isinstance(total, Decimal)
        lifnr = con.execute("SELECT min(LIFNR) FROM sap_lfa1").fetchone()[0]
        assert lifnr.startswith("0000") and len(lifnr) == 10
        padded = con.execute("SELECT count(*) FROM sap_lfbk WHERE BANKN LIKE '0%'").fetchone()[0]
        assert padded > 0
