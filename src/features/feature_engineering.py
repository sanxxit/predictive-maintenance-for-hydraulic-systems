import numpy as np
import pandas as pd

def process_sensor_arrays(file_names: list, data_arrays: list) -> pd.DataFrame:
    """
    Aggregates time-series sensor arrays using the mean over axis 1.
    
    Args:
        file_names: List of filenames (e.g. ['PS1.txt']).
        data_arrays: List of corresponding numpy arrays loaded from text.
        
    Returns:
        A DataFrame with the aggregated features.
    """
    feature_dict = {}
    for name, arr in zip(file_names, data_arrays):
        feature_name = name.split('.')[0] # e.g., 'PS1'
        feature_dict[feature_name] = arr.mean(axis=1)
        
    return pd.DataFrame(feature_dict)

def create_feature_dataset(sensor_df: pd.DataFrame, profile_df: pd.DataFrame) -> pd.DataFrame:
    """
    Combines the sensor features and target profiles into a single dataset.
    """
    return pd.concat([sensor_df, profile_df], axis=1)
