import pandas as pd
from deltalake import DeltaTable

from app.etl.table_formats import get_iceberg_catalog, load_to_delta, load_to_iceberg, DELTA_DIR


def test_delta_lake_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr("app.etl.table_formats.DELTA_DIR", tmp_path)
    df = pd.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"]})
    load_to_delta("sample", df)
    result = DeltaTable(str(tmp_path / "sample")).to_pandas()
    assert len(result) == 3
    assert sorted(result["name"]) == ["a", "b", "c"]


def test_iceberg_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr("app.etl.table_formats.ICEBERG_WAREHOUSE_DIR", tmp_path / "warehouse")
    monkeypatch.setattr("app.etl.table_formats.ICEBERG_CATALOG_DB", tmp_path / "catalog.db")
    catalog = get_iceberg_catalog()
    df = pd.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"]})
    load_to_iceberg(catalog, "sample", df)
    result = catalog.load_table("staging.sample").scan().to_pandas()
    assert len(result) == 3
    assert sorted(result["name"]) == ["a", "b", "c"]


def test_null_columns_cast_to_string_for_iceberg_v2_compatibility():
    import pyarrow as pa
    from app.etl.table_formats import _cast_null_columns_to_string

    df = pd.DataFrame({"id": [1, 2], "always_empty": [None, None]})
    arrow_tbl = pa.Table.from_pandas(df, preserve_index=False)
    assert pa.types.is_null(arrow_tbl.schema.field("always_empty").type)

    fixed = _cast_null_columns_to_string(arrow_tbl)
    assert pa.types.is_string(fixed.schema.field("always_empty").type)
