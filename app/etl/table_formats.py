"""Loads the staged Parquet extracts into real Delta Lake and Apache
Iceberg tables -- an additional consumption layer on top of the raw
Parquet staging area, matching how a warehouse would actually adopt a
transactional table format for downstream analytics (schema evolution,
time travel, ACID writes) rather than reading raw Parquet files
directly.

Both are local, catalog-backed table formats (no Spark cluster
required): Delta Lake via the native Rust-backed `deltalake` library,
Iceberg via `pyiceberg`'s SQL catalog (SQLite-backed here; point it at
a real Postgres/Glue catalog in production).
"""

import os
from pathlib import Path

import pandas as pd
import pyarrow as pa
from deltalake import write_deltalake
from pyiceberg.catalog.sql import SqlCatalog

WORKSPACE_DIR = Path(__file__).resolve().parents[2]
STAGED_DIR = WORKSPACE_DIR / "data" / "staged"
DELTA_DIR = WORKSPACE_DIR / "data" / "delta"
ICEBERG_WAREHOUSE_DIR = WORKSPACE_DIR / "data" / "iceberg_warehouse"
ICEBERG_CATALOG_DB = WORKSPACE_DIR / "data" / "iceberg_catalog.db"

ICEBERG_NAMESPACE = "staging"


def get_iceberg_catalog() -> SqlCatalog:
    ICEBERG_WAREHOUSE_DIR.mkdir(parents=True, exist_ok=True)
    catalog = SqlCatalog(
        "eventintel",
        **{
            "uri": f"sqlite:///{ICEBERG_CATALOG_DB}",
            "warehouse": f"file://{ICEBERG_WAREHOUSE_DIR}",
        },
    )
    catalog.create_namespace_if_not_exists(ICEBERG_NAMESPACE)
    return catalog


def load_to_delta(table_name: str, df: pd.DataFrame) -> None:
    output_path = DELTA_DIR / table_name
    write_deltalake(str(output_path), df, mode="overwrite")
    print(f"[+] Wrote {len(df):,} rows to Delta table: {output_path}")


def _cast_null_columns_to_string(arrow_tbl: pa.Table) -> pa.Table:
    """An all-null source column (e.g. a legacy field never populated in
    this synthetic dataset) infers as pyarrow's null type, which Iceberg's
    table-format spec only supports from format-version 3 onward. Casting
    to string keeps these columns representable under the default v2
    format rather than forcing every table onto v3."""
    for i, field in enumerate(arrow_tbl.schema):
        if pa.types.is_null(field.type):
            arrow_tbl = arrow_tbl.set_column(i, field.name, arrow_tbl.column(i).cast(pa.string()))
    return arrow_tbl


def load_to_iceberg(catalog: SqlCatalog, table_name: str, df: pd.DataFrame) -> None:
    arrow_tbl = pa.Table.from_pandas(df, preserve_index=False)
    arrow_tbl = _cast_null_columns_to_string(arrow_tbl)
    identifier = f"{ICEBERG_NAMESPACE}.{table_name}"
    table = catalog.create_table_if_not_exists(identifier, schema=arrow_tbl.schema)
    table.overwrite(arrow_tbl)
    print(f"[+] Wrote {len(df):,} rows to Iceberg table: {identifier}")


def main() -> None:
    if not STAGED_DIR.exists():
        raise FileNotFoundError(f"{STAGED_DIR} not found -- run etl_extract.py first.")

    parquet_files = sorted(STAGED_DIR.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No staged Parquet files found in {STAGED_DIR}.")

    print("=" * 60)
    print("TABLE FORMAT LOADER -- Delta Lake & Apache Iceberg")
    print(f"Source: {STAGED_DIR} ({len(parquet_files)} tables)")
    print("=" * 60)

    catalog = get_iceberg_catalog()

    for parquet_file in parquet_files:
        table_name = parquet_file.stem
        df = pd.read_parquet(parquet_file)
        load_to_delta(table_name, df)
        load_to_iceberg(catalog, table_name, df)

    print("\n[+] Table format load complete.")


if __name__ == "__main__":
    main()
