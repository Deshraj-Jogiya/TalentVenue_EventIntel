import sys
from pathlib import Path

# Snowflake Python UDF Deployment Script
# Demonstrates how to register a scikit-learn model inside Snowflake using Snowpark.

SNOWFLAKE_UDF_SQL = """
-- ============================================================================
-- Snowflake Python UDF Registration
-- Run this in your Snowflake worksheet after uploading the trained model file.
-- ============================================================================

-- 1. Create a stage to store machine learning models
CREATE OR REPLACE STAGE ENTERPRISE_DW.ANALYTICS.STAGE_MODELS;

-- Note: The model file 'cancellation_risk_model.joblib' should be uploaded to @STAGE_MODELS
-- This can be done via SnowSQL command: 
-- PUT file://./data/models/cancellation_risk_model.joblib @STAGE_MODELS AUTO_COMPRESS=FALSE OVERWRITE=TRUE;

-- 2. Create the Python User-Defined Function (UDF) for inference
CREATE OR REPLACE FUNCTION ENTERPRISE_DW.ANALYTICS.PREDICT_CANCELLATION_RISK(
    LEAD_TIME INT,
    GROSS_AMOUNT FLOAT,
    EVENT_MONTH INT,
    PRESENTER_CANCELLATION_RATE FLOAT,
    ARTIST_CANCELLATION_RATE FLOAT
)
RETURNS FLOAT
LANGUAGE PYTHON
RUNTIME_VERSION = '3.8'
PACKAGES = ('scikit-learn', 'joblib', 'pandas', 'numpy')
IMPORTS = ('@ENTERPRISE_DW.ANALYTICS.STAGE_MODELS/cancellation_risk_model.joblib')
HANDLER = 'predict_cancellation'
AS
$$
import os
import sys
import joblib
import pandas as pd

# Locate the imported model file inside the Snowflake virtual environment
import_dir = sys._xoptions.get("snowflake_import_directory")
if import_dir:
    model_path = os.path.join(import_dir, "cancellation_risk_model.joblib")
else:
    model_path = "cancellation_risk_model.joblib"

# Load the Random Forest model once when the function initializes (cached in memory)
model = joblib.load(model_path)

def predict_cancellation(lead_time, gross_amount, event_month, presenter_cancellation_rate, artist_cancellation_rate):
    \"\"\"
    Performs real-time machine learning inference inside the Snowflake engine.
    \"\"\"
    # Create the feature DataFrame matching the model's training columns
    features = pd.DataFrame([{
        'LEAD_TIME': lead_time,
        'GROSS_AMOUNT': gross_amount,
        'EVENT_MONTH': event_month,
        'PRESENTER_CANCELLATION_RATE': presenter_cancellation_rate,
        'ARTIST_CANCELLATION_RATE': artist_cancellation_rate
    }])
    
    # Get the probability score for the '1' (cancellation) class
    risk_probability = model.predict_proba(features)[0, 1]
    return float(risk_probability)
$$;

-- 3. Query the UDF directly using SQL!
-- SELECT 
--     contract_id,
--     gross_amount,
--     PREDICT_CANCELLATION_RISK(
--         lead_time,
--         gross_amount,
--         event_month,
--         presenter_cancellation_rate,
--         artist_cancellation_rate
--     ) AS cancellation_risk_score
-- FROM ENTERPRISE_DW.ANALYTICS.FACT_CONTRACT_FINANCE
-- LIMIT 10;
"""

def main():
    print("=" * 60)
    print("SNOWFLAKE MACHINE LEARNING UDF GENERATOR")
    print("=" * 60)
    
    output_dir = Path(__file__).resolve().parents[2] / "app" / "sql"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "deploy_ml_udf.sql"
    
    with open(output_file, "w") as f:
        f.write(SNOWFLAKE_UDF_SQL)
        
    print(f"[+] Snowflake ML UDF script generated at {output_file}")
    print("\nInstruct your DBA or use SnowSQL to run this script to deploy model in Snowflake.")

if __name__ == "__main__":
    main()
