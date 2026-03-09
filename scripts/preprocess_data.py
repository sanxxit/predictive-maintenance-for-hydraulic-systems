import os
import sys

# Add the project root to the python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from src.data.loader import load_sensor_data, load_profile_data
from src.data.preprocessing import handle_missing_values
from src.features.feature_engineering import process_sensor_arrays, create_feature_dataset

DATA_RAW_DIR = os.path.join(project_root, 'data', 'raw')
DATA_PROCESSED_DIR = os.path.join(project_root, 'data', 'processed')

SENSOR_FILES = [
    'PS1.txt', 'PS2.txt', 'PS3.txt', 'PS4.txt', 'PS5.txt', 'PS6.txt',
    'EPS1.txt', 
    'FS1.txt', 'FS2.txt', 
    'TS1.txt', 'TS2.txt', 'TS3.txt', 'TS4.txt', 
    'VS1.txt', 
    'CE.txt', 'CP.txt', 
    'SE.txt'
]

def main():
    print(f"Loading sensor data from {DATA_RAW_DIR}...")
    sensor_arrays = load_sensor_data(DATA_RAW_DIR, SENSOR_FILES)
    
    print("Loading profile targets...")
    profile_df = load_profile_data(DATA_RAW_DIR)
    
    print("Aggregating sensor features...")
    sensor_df = process_sensor_arrays(SENSOR_FILES, sensor_arrays)
    
    print("Handling missing values...")
    sensor_df = handle_missing_values(sensor_df)
    
    print("Creating combined feature dataset...")
    final_df = create_feature_dataset(sensor_df, profile_df)
    
    # Ensure processed directory exists
    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
    
    output_path = os.path.join(DATA_PROCESSED_DIR, 'features.parquet')
    print(f"Saving processed features to {output_path}...")
    final_df.to_parquet(output_path, index=False)
    
    print("Data preprocessing completed successfully.")

if __name__ == '__main__':
    main()
