import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Add project root to python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from src.models.train import train_classifier, save_model

DATA_PROCESSED_PATH = os.path.join(project_root, 'data', 'processed', 'features.parquet')
MODELS_DIR = os.path.join(project_root, 'models', 'trained_models')

TARGET_COLS = [
    "Cooler_Condition",
    "Valve_Condition",
    "Internal_Pump_Leakage",
    "Hydraulic_Accumulator",
    "Stable_Flag"
]

def main():
    if not os.path.exists(DATA_PROCESSED_PATH):
        print(f"Processed data not found at {DATA_PROCESSED_PATH}. Please run preprocess_data.py first.")
        sys.exit(1)
        
    print(f"Loading processed data from {DATA_PROCESSED_PATH}...")
    df = pd.read_parquet(DATA_PROCESSED_PATH)
    
    # Split features and targets
    X = df.drop(columns=TARGET_COLS)
    y_full = df[TARGET_COLS]
    
    # Optional global scaling scaler: 
    # For a robust ML pipeline, we should scale per target if splitting differs,
    # but since data is identical, we can split and scale once.
    # We will pick a fixed seed to ensure X_train is identical for all targets.
    
    print("Training models for each target...")
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Train robust Random Forest for all targets as default
    for target in TARGET_COLS:
        print(f"--- Processing Target: {target} ---")
        y = y_full[target]
        
        # We split individually, though the indices will be the same due to identical random state.
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Train model
        model = train_classifier(X_train_scaled, y_train, model_type="rf", random_state=42)
        
        # Save model and scaler
        scaler_path = os.path.join(MODELS_DIR, f"{target}_scaler.pkl")
        model_path = os.path.join(MODELS_DIR, f"{target}_rf_model.pkl")
        
        save_model(scaler, scaler_path)
        save_model(model, model_path)
        print(f"Saved model and scaler for {target} to {MODELS_DIR}")
        
    print("Model training completed successfully.")

if __name__ == '__main__':
    main()
