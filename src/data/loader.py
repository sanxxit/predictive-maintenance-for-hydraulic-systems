import os
import numpy as np
import pandas as pd

def load_sensor_data(data_dir: str, file_names: list) -> list:
    """
    Loads sensor data text files from the specified directory.
    
    Args:
        data_dir: Path to the raw data directory.
        file_names: List of filenames to load (e.g., ['PS1.txt', 'PS2.txt']).
        
    Returns:
        List of numpy arrays corresponding to the loaded files.
    """
    data_arrays = []
    for file_name in file_names:
        file_path = os.path.join(data_dir, file_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Sensor file {file_path} not found.")
        data_arrays.append(np.genfromtxt(file_path))
    return data_arrays

def load_profile_data(data_dir: str) -> pd.DataFrame:
    """
    Loads the target profile data.
    
    Args:
        data_dir: Path to the raw data directory.
        
    Returns:
        DataFrame containing the targets.
    """
    file_path = os.path.join(data_dir, "profile.txt")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Profile file {file_path} not found.")
    
    target = np.genfromtxt(file_path)
    columns = [
        "Cooler_Condition",
        "Valve_Condition",
        "Internal_Pump_Leakage",
        "Hydraulic_Accumulator",
        "Stable_Flag"
    ]
    df_profile = pd.DataFrame(target, columns=columns)
    return df_profile
