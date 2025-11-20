# TalentVenue EventIntel

Enterprise data consolidation and predictive intelligence pipeline for a talent-booking/event-management business (contracts, artists, presenters, venues, general ledger). Built as an ElevateMe Bootcamp capstone project.

## What this is

A legacy SQL Server database (an internal event-booking/talent-management system) is extracted, cleaned, and staged, then used to train a model that predicts contract cancellation risk — surfaced through an internal Streamlit data portal, with the target production data warehouse (Azure ADLS Gen2 + Snowflake) provisioned as Infrastructure as Code.

## Pipeline

1. **Extraction** (`app/etl/etl_extract.py`) — pulls 10 core tables (Artist, Presenter, Venue, Contract, GeneralLedgerJournal, etc.) from SQL Server via `pyodbc`, normalizes column names to `UPPER_SNAKE_CASE`, SHA-256-hashes PII fields (TIN, bank account/routing numbers), and writes each table to Parquet under `data/staged/`.
2. **Feature engineering & training** (`app/ml/train_risk_model.py`) — engineers lead-time, seasonality, and Bayesian-smoothed presenter/artist cancellation-rate features, then trains a Random Forest classifier predicting contract cancellation.
   - **Accuracy: 86.20%, ROC-AUC: 0.8707** (reproducible — `python app/ml/train_risk_model.py` against the staged data in this repo).
3. **Deployment** (`app/ml/deploy_udf.py`) — generates a Snowflake Python UDF (Snowpark) SQL script that loads the trained model into Snowflake and serves real-time inference directly from SQL.
4. **Data portal** (`app/dashboard/app_dashboard.py`) — a multi-phase Streamlit application: operational baseline, legacy data diagnostics/audit, data governance & EDA, business-intelligence Q&A, and the predictive/prescriptive ML view surfacing the cancellation-risk model.
5. **Infrastructure** (`terraform/main.tf`) — provisions the target production data warehouse: an Azure ADLS Gen2 storage account/container for staging, and Snowflake database/schemas/warehouse plus the storage integration connecting the two.

## Running it

```bash
pip install -r requirements.txt

# 1. Extract from SQL Server (requires a local SQL Server instance with the source database restored)
python app/etl/etl_extract.py

# 2. Train the risk model against the staged Parquet files
python app/ml/train_risk_model.py

# 3. Generate the Snowflake UDF deployment script
python app/ml/deploy_udf.py

# 4. Launch the data portal
streamlit run app/dashboard/app_dashboard.py
```

`data/staged/*.parquet` and `data/models/` are committed so steps 2-4 run immediately without a live SQL Server connection.

## Testing

```bash
pytest tests/ -v
```

## Note on scope

The original source SQL Server database backup (`.bacpac`) is not included here (multi-GB, and contains data structurally similar to production data) — the staged, PII-hashed Parquet extracts in `data/staged/` are the shareable artifact.
