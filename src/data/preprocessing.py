import numpy as np
import pandas as pd

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fills or drops missing values in the given DataFrame.
    In the original notebook, there were no missing values in PS1,
    but this provides a standard mechanism if any exist.
    """
    # Simple strategy: forward fill then backward fill
    return df.ffill().bfill()

def normalize_sensor_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes or scales sensor data if needed.
    (Note: StandardScaler is usually applied after train/test split.
    This function is kept minimal as a placeholder for global preprocessing)
    """
    # Standard scaling should preferably be done inside the ML pipeline
    # per split, but this can be used for any pre-split transformation.
    return df
