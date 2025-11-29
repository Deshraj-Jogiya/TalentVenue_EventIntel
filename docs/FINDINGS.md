# Findings

Measured results from this repository. Each number below comes from re-running the repository's own code on the staged data it
ships (`data/staged/*.parquet`), offline.

## How well does the model predict contract cancellations, and where does it fall short?

**How it was tested:** `python -m app.ml.train_risk_model` engineers features from the staged contract tables, splits them 80/20
(stratified, `random_state=42`) and trains a Random Forest (100 trees, depth 12). Of the 2,000 held-out contracts, 352 were
cancelled.

| Measure | Value |
|---|---|
| Accuracy | 86.20% |
| ROC-AUC | 0.8707 |
| Cancellation precision | 0.65 |
| Cancellation recall | 0.48 |
| Completed-contract precision / recall | 0.89 / 0.94 |

**What it shows:** the model ranks risk well (ROC-AUC 0.8707), but at the default threshold it finds about half of the cancellations
(recall 0.48). Accuracy alone overstates it, because most contracts complete.

**Decision:** the model runs as a Snowflake Python UDF (`app/ml/deploy_udf.py`) so scores are computed in SQL, and a TypeScript CLI
(zod, Node's test runner) validates the feature-mappings file before the dashboard uses it (`tools/validate-feature-mappings/`).

**Skills shown:** scikit-learn (RandomForestClassifier, stratified split, ROC-AUC, classification report), pandas on Parquet,
Snowflake Python UDF, TypeScript with zod, pytest.

**Source:** `app/ml/train_risk_model.py` (printed metrics and classification report, re-run on the committed staged Parquet files),
`data/models/feature_mappings.json`, `tests/test_ml.py`.
