# Hydraulic Predictive Maintenance

## Project Overview
This repository contains a dataset and modeling code for Hydraulic System Condition Monitoring, used in predictive maintenance research. The project analyzes multivariate time-series data from various sensors to predict the health state of hydraulic system components like cooler, valve, pump leakage, and accumulator pressure stability.

## Dataset Description
- 2205 machine cycles
- Each cycle = 60 seconds
- Multivariate time-series data
- Sensors included: Pressure sensors (PS1-PS6), Motor power (EPS1), Flow sensors (FS1, FS2), Temperature sensors (TS1-TS4), Vibration sensor (VS1), Virtual sensors (CE, CP, SE)

## Repository Structure
- `data/`: Contains raw, processed, and metadata.
- `notebooks/`: Jupyter notebooks for exploration and modeling.
- `src/`: Source code for data loading, preprocessing, feature engineering, and model training.
- `scripts/`: Executable scripts.
- `config/`: Configuration files.
- `models/`: Trained models.
- `reports/`: Generated figures and reports.
- `tests/`: Unit tests.

## Installation
```bash
pip install -r requirements.txt
```

## Usage
### Running the Notebook
The main exploration and modeling code can be found in `notebooks/predictive_maintenance.ipynb`. To run it:
```bash
jupyter notebook notebooks/predictive_maintenance.ipynb
```

## Future Work
- Implement automated pipeline using scripts.
- Hyperparameter tuning for deployed models.
- Model monitoring in a simulated production environment.

## References
- Source data documentation is available in `data/metadata/`.
