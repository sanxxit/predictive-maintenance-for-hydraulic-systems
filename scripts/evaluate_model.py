import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

# Add project root to python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from src.models.train import load_model
from src.models.evaluate import compute_metrics, save_confusion_matrix, save_metrics

DATA_PROCESSED_PATH = os.path.join(project_root, 'data', 'processed', 'features.parquet')
MODELS_DIR = os.path.join(project_root, 'models', 'trained_models')
REPORTS_DIR = os.path.join(project_root, 'reports')

TARGET_COLS = [
    "Cooler_Condition",
    "Valve_Condition",
    "Internal_Pump_Leakage",
    "Hydraulic_Accumulator",
    "Stable_Flag"
]

def main():
    if not os.path.exists(DATA_PROCESSED_PATH):
        print(f"Processed data not found at {DATA_PROCESSED_PATH}. Run preprocess_data.py first.")
        sys.exit(1)
        
    print(f"Loading processed data from {DATA_PROCESSED_PATH}...")
    df = pd.read_parquet(DATA_PROCESSED_PATH)
    
    X = df.drop(columns=TARGET_COLS)
    y_full = df[TARGET_COLS]
    
    os.makedirs(REPORTS_DIR, exist_ok=True)
    all_metrics = {}
    
    print("Evaluating models for each target...")
    for target in TARGET_COLS:
        print(f"--- Evaluating Target: {target} ---")
        y = y_full[target]
        
        # Consistent split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        scaler_path = os.path.join(MODELS_DIR, f"{target}_scaler.pkl")
        model_path = os.path.join(MODELS_DIR, f"{target}_rf_model.pkl")
        
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            print(f"Skipping {target}: Model or scaler not found in {MODELS_DIR}.")
            continue
            
        scaler = load_model(scaler_path)
        model = load_model(model_path)
        
        X_test_scaled = scaler.transform(X_test)
        y_pred = model.predict(X_test_scaled)
        
        # Compute metrics
        metrics = compute_metrics(y_test, y_pred, average="weighted")
        all_metrics[target] = metrics
        
        # Save confusion matrix
        cm_path = os.path.join(REPORTS_DIR, f"{target}_confusion_matrix.png")
        save_confusion_matrix(y_test, y_pred, filepath=cm_path, title=f"Confusion Matrix: {target}")
        
    # Save absolute metrics JSON
    metrics_path = os.path.join(REPORTS_DIR, 'metrics.json')
    save_metrics(all_metrics, metrics_path)
    print(f"Evaluation completed. Reports saved to {REPORTS_DIR}")

if __name__ == '__main__':
    main()
