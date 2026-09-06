"""
Test loading the data
"""
import pandas as pd
import os

print("Loading data...")
try:
    df = pd.read_csv("data/processed_data/patient_data_processed.csv")
    print(f"[OK] Data loaded successfully. Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"First few rows:")
    print(df.head())
except Exception as e:
    print(f"[FAIL] Failed to load data: {e}")
    import traceback
    traceback.print_exc()

print("Test complete.")