# 🏭 Predictive Maintenance for Hydraulic Systems

![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange.svg)
![pandas](https://img.shields.io/badge/pandas-1.3%2B-lightgrey.svg)
![Status](https://img.shields.io/badge/status-production--ready-success.svg)

## 📌 Project Overview
This repository implements an automated, end-to-end machine learning pipeline for **Hydraulic System Condition Monitoring**. By leveraging multivariate time-series data from distributed sensors, the system performs classification to predict the health state of critical hydraulic components. 

The primary business objective is to transition from reactive maintenance to **predictive maintenance**, minimizing equipment downtime, reducing maintenance costs, and preventing catastrophic system failures.

### 🎯 Target Variables (Multitask Classification)
The pipeline simultaneously predicts five distinct component conditions:
1. **Cooler Condition** (3 classes)
2. **Valve Condition** (4 classes)
3. **Internal Pump Leakage** (3 classes)
4. **Hydraulic Accumulator** (4 classes)
5. **Stable Flag** (Binary)

---

## 🏗️ System Architecture

The project has been refactored from a monolithic Jupyter Notebook into a modular, highly cohesive, and loosely coupled ML pipeline.

```text
predictive_maintenance/
├── data/
│   ├── raw/               <- Immutable raw sensor data (PS1.txt, etc.)
│   └── processed/         <- Cached parity output (features.parquet)
├── src/                   <- Core modular logic 
│   ├── data/              <- Loaders & Preprocessors
│   ├── features/          <- Time-series aggregations (mean, std)
│   └── models/            <- Model abstractions (RandomForest, LR)
├── scripts/               <- Pipeline orchestrators
│   ├── preprocess_data.py
│   ├── train_model.py
│   └── evaluate_model.py
├── models/                <- Serialized model artifacts (.pkl)
└── reports/               <- Generated metrics, JSONs, & confusion matrices
```

---

## 🚀 Quickstart & Pipeline Execution

### 1. Environment Setup
We enforce strict dependency management to ensure absolute reproducibility across environments.
```bash
# Clone the repository
git clone https://github.com/sanxxit/predictive-maintenance-for-hydraulic-systems.git
cd predictive-maintenance-for-hydraulic-systems

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the ML Pipeline
The pipeline is designed to be executed sequentially. All stochastic operations rely on a fixed `random_state=42` to guarantee deterministic builds.

**Step 1: Data Ingestion & Feature Engineering**
Extracts 17 raw `.txt` sensor arrays, engineers aggregate features across the time axis, and persists a highly optimized subset.
```bash
python scripts/preprocess_data.py
# Outputs: data/processed/features.parquet
```

**Step 2: Model Training**
Splits the engineered dataset (80/20 train-test split), applies standard scaling, and fits specific `RandomForestClassifier` models mapping to all 5 target variables.
```bash
python scripts/train_model.py
# Outputs: 5 Model artifacts and 5 standard scalers to models/trained_models/
```

**Step 3: Verification & Evaluation**
Validates model robustness using a held-out testing partition, generating exhaustive performance reporting.
```bash
python scripts/evaluate_model.py
# Outputs: reports/metrics.json and comprehensive confusion matrices.
```

---

## 📊 Model Performance

Our current champion model (`RandomForestClassifier` with 100 estimators) demonstrates near-perfect convergence across our hold-out evaluation metric validation standard (`test_size=0.2`). Check inside `reports/metrics.json` for live details.

| Target | Accuracy | Precision | Recall |
| :--- | :---: | :---: | :---: |
| **Cooler Condition** | 100.0% | 1.000 | 1.000 |
| **Valve Condition** | 96.6% | 0.966 | 0.966 |
| **Internal Pump Leakage** | 99.5% | 0.995 | 0.995 |
| **Hydraulic Accumulator** | 97.5% | 0.976 | 0.975 |
| **Stable_Flag** | 96.8% | 0.969 | 0.968 |

*Note: Models are sensitive to data drift. Ensure continuous monitoring in live environments.*

---

## 🔮 Future Work
- **Distributed Processing:** Migrate `feature_engineering.py` onto PySpark or Ray for horizontal scaling when processing TB-level highly volatile multi-sensor arrays.
- **MLOps Integration:** Containerize via Docker and deploy model training pipelines strictly tracked to an MLflow registry.
- **Real-time Inference:** Expose the `trained_models/` payloads within a FastAPI microservice.

---
*Maintained by the AI Platform Engineering Team.*
