import pytest
import pandas as pd
import numpy as np
from app.ml.train_risk_model import engineer_features

def test_engineer_features():
    """Verifies that ML features are correctly engineered from raw inputs."""
    # Create mock contract data
    contracts_data = {
        "CONTRACT_ID": [1, 2, 3],
        "PRESENTER_ID": [101, 101, 102],
        "GROSS": ["10000.00", "20000.00", "50000.00"],
        "CREATED_DATE": ["2025-01-01", "2025-01-05", "2025-01-10"],
        "CONTRACT_DUE_DATE": ["2025-01-11", "2025-01-15", "2025-02-10"],
        "CANCELLATION_DATE": [None, "2025-01-10", None]
    }
    df_contract = pd.DataFrame(contracts_data)
    
    # Create mock contract artist mapping
    artist_data = {
        "CONTRACT_ID": [1, 2, 3],
        "ARTIST_ID": [501, 501, 502]
    }
    df_contract_artist = pd.DataFrame(artist_data)
    
    # Run feature engineering
    X, y, features = engineer_features(df_contract, df_contract_artist)
    
    # Asserts
    assert list(X.columns) == [
        "LEAD_TIME",
        "GROSS",
        "EVENT_MONTH",
        "PRESENTER_CANCELLATION_RATE",
        "ARTIST_CANCELLATION_RATE"
    ]
    
    # Lead time check: 10 days for contract 1, 10 days for contract 2, 31 days for contract 3
    assert list(X["LEAD_TIME"].values) == [10.0, 10.0, 31.0]
    
    # Gross amount numeric cast check
    assert list(X["GROSS"].values) == [10000.0, 20000.0, 50000.0]
    
    # Seasonality check: event months are January (1), January (1), and February (2)
    assert list(X["EVENT_MONTH"].values) == [1, 1, 2]
    
    # Target variable check: contract 2 is cancelled (1), contract 1 and 3 are completed (0)
    assert list(y.values) == [0, 1, 0]
    
    # Check that there are no NaNs in the output features
    assert X.isna().sum().sum() == 0
