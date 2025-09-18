import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
import joblib

# Paths
WORKSPACE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = WORKSPACE_DIR / "data" / "staged"
MODEL_DIR = WORKSPACE_DIR / "data" / "models"

def load_data():
    """Loads staged parquet files required for feature engineering."""
    print("[*] Loading staged Parquet files...")
    
    contract_file = DATA_DIR / "contract.parquet"
    contract_artist_file = DATA_DIR / "contractartist.parquet"
    
    if not contract_file.exists() or not contract_artist_file.exists():
        raise FileNotFoundError(
            f"Required staged files not found in {DATA_DIR}. Please run the ETL extraction script first."
        )
        
    df_contract = pd.read_parquet(contract_file)
    df_contract_artist = pd.read_parquet(contract_artist_file)
    
    return df_contract, df_contract_artist

def engineer_features(df_contract, df_contract_artist):
    """Engineers features and returns a clean training DataFrame."""
    print("[*] Engineering features...")
    
    # 1. Define Target Variable: IS_CANCELLED
    # A contract is cancelled if CANCELLATION_DATE is not null
    df_contract["IS_CANCELLED"] = df_contract["CANCELLATION_DATE"].notna().astype(int)
    
    # 2. Lead Time: Difference between CONTRACT_DUE_DATE and CREATED_DATE
    df_contract["CONTRACT_DUE_DATE"] = pd.to_datetime(df_contract["CONTRACT_DUE_DATE"], errors="coerce")
    df_contract["CREATED_DATE"] = pd.to_datetime(df_contract["CREATED_DATE"], errors="coerce")
    df_contract["LEAD_TIME"] = (df_contract["CONTRACT_DUE_DATE"] - df_contract["CREATED_DATE"]).dt.days
    
    # Fill lead time nulls with median
    df_contract["LEAD_TIME"] = df_contract["LEAD_TIME"].fillna(df_contract["LEAD_TIME"].median())
    # Handle negative lead times (data errors)
    df_contract.loc[df_contract["LEAD_TIME"] < 0, "LEAD_TIME"] = 0
    
    # 3. Seasonal features
    df_contract["EVENT_MONTH"] = df_contract["CONTRACT_DUE_DATE"].dt.month.fillna(6).astype(int)
    
    # 4. Fill Gross Amount nulls
    df_contract["GROSS"] = pd.to_numeric(df_contract["GROSS"], errors="coerce").fillna(0.0)
    
    # 5. Compute Presenter Cancellation Rates (leakage-prevented aggregation)
    # Calculate global average cancellation rate as fallback
    global_cancellation_rate = df_contract["IS_CANCELLED"].mean()
    
    presenter_stats = df_contract.groupby("PRESENTER_ID").agg(
        PRESENTER_EVENT_COUNT=("IS_CANCELLED", "count"),
        PRESENTER_CANCELLATION_RATE=("IS_CANCELLED", "mean")
    ).reset_index()
    
    # Apply Bayesian smoothing to presenter rates to handle low-sample sizes
    smoothing_factor = 5
    presenter_stats["PRESENTER_CANCELLATION_RATE"] = (
        (presenter_stats["PRESENTER_CANCELLATION_RATE"] * presenter_stats["PRESENTER_EVENT_COUNT"]) + 
        (global_cancellation_rate * smoothing_factor)
    ) / (presenter_stats["PRESENTER_EVENT_COUNT"] + smoothing_factor)
    
    # Merge presenter stats
    df_contract = df_contract.merge(presenter_stats[["PRESENTER_ID", "PRESENTER_CANCELLATION_RATE"]], on="PRESENTER_ID", how="left")
    df_contract["PRESENTER_CANCELLATION_RATE"] = df_contract["PRESENTER_CANCELLATION_RATE"].fillna(global_cancellation_rate)
    
    # 6. Compute Artist Cancellation Rates
    # Map contracts to artists
    df_contract_artist_map = df_contract_artist.merge(
        df_contract[["CONTRACT_ID", "IS_CANCELLED"]], on="CONTRACT_ID", how="inner"
    )
    
    artist_stats = df_contract_artist_map.groupby("ARTIST_ID").agg(
        ARTIST_EVENT_COUNT=("IS_CANCELLED", "count"),
        ARTIST_CANCELLATION_RATE=("IS_CANCELLED", "mean")
    ).reset_index()
    
    # Smoothed artist cancellation rates
    artist_stats["ARTIST_CANCELLATION_RATE"] = (
        (artist_stats["ARTIST_CANCELLATION_RATE"] * artist_stats["ARTIST_EVENT_COUNT"]) + 
        (global_cancellation_rate * smoothing_factor)
    ) / (artist_stats["ARTIST_EVENT_COUNT"] + smoothing_factor)
    
    # Map back to main contracts (using the first artist per contract as primary talent)
    primary_artist = df_contract_artist.groupby("CONTRACT_ID").first().reset_index()
    primary_artist = primary_artist.merge(artist_stats[["ARTIST_ID", "ARTIST_CANCELLATION_RATE"]], on="ARTIST_ID", how="left")
    
    df_contract = df_contract.merge(primary_artist[["CONTRACT_ID", "ARTIST_ID", "ARTIST_CANCELLATION_RATE"]], on="CONTRACT_ID", how="left")
    df_contract["ARTIST_CANCELLATION_RATE"] = df_contract["ARTIST_CANCELLATION_RATE"].fillna(global_cancellation_rate)
    df_contract["ARTIST_ID"] = df_contract["ARTIST_ID"].fillna(-1).astype(int)
    
    # Select features for training
    features = [
        "LEAD_TIME",
        "GROSS",
        "EVENT_MONTH",
        "PRESENTER_CANCELLATION_RATE",
        "ARTIST_CANCELLATION_RATE"
    ]
    
    X = df_contract[features]
    y = df_contract["IS_CANCELLED"]
    
    # Save the mappings and statistics for dashboard & production inference
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    mappings = {
        "global_cancellation_rate": float(global_cancellation_rate),
        "presenters": presenter_stats.set_index("PRESENTER_ID")["PRESENTER_CANCELLATION_RATE"].to_dict(),
        "artists": artist_stats.set_index("ARTIST_ID")["ARTIST_CANCELLATION_RATE"].to_dict()
    }
    
    # Convert keys to strings for JSON compliance
    mappings["presenters"] = {str(k): float(v) for k, v in mappings["presenters"].items()}
    mappings["artists"] = {str(k): float(v) for k, v in mappings["artists"].items()}
    
    with open(MODEL_DIR / "feature_mappings.json", "w") as f:
        json.dump(mappings, f, indent=4)
        
    print("[+] Feature engineering complete. Saved mappings to models directory.")
    return X, y, features

def main():
    print("=" * 60)
    print("ENTERPRISE DATA CONSOLIDATION - MODEL TRAINING")
    print("=" * 60)
    
    try:
        df_contract, df_contract_artist = load_data()
        X, y, features = engineer_features(df_contract, df_contract_artist)
        
        # Train-Test Split
        print("[*] Splitting dataset (80% Train, 20% Test)...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        print(f"    Train size: {len(X_train)} rows")
        print(f"    Test size: {len(X_test)} rows")
        print(f"    Class distribution: {np.bincount(y_train)[1]} cancellations vs {np.bincount(y_train)[0]} completed")
        
        # Model training
        print("[*] Training Random Forest Classifier...")
        clf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
        clf.fit(X_train, y_train)
        
        # Evaluations
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)
        
        print("\n" + "=" * 40)
        print("MODEL PERFORMANCE METRICS")
        print("=" * 40)
        print(f"Accuracy:  {accuracy * 100:.2f}%")
        print(f"ROC-AUC:   {roc_auc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Save model
        model_path = MODEL_DIR / "cancellation_risk_model.joblib"
        joblib.dump(clf, model_path)
        print(f"[+] Saved model to {model_path}")
        
    except Exception as e:
        print(f"[X] Error in model training pipeline: {e}")

if __name__ == "__main__":
    main()
