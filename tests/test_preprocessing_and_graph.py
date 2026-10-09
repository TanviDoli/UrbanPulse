import os
import pytest
import pandas as pd
import numpy as np

def test_zone_adjacency_graph():
    adj_path = 'data/processed/zone_adjacency.csv'
    assert os.path.exists(adj_path), "zone_adjacency.csv does not exist"
    df = pd.read_csv(adj_path)
    assert len(df) == 105, f"Expected 105 zone pairs for 15 zones, got {len(df)}"
    adjacent_pairs = df[df['adjacent'] == 1]
    assert len(adjacent_pairs) == 29, f"Expected 29 adjacent zone pairs, got {len(adjacent_pairs)}"
    assert (adjacent_pairs['shared_boundary_length'] > 0).all()

def test_station_zone_mapping():
    mapping_path = 'data/processed/station_zone_mapping.csv'
    assert os.path.exists(mapping_path), "station_zone_mapping.csv does not exist"
    df = pd.read_csv(mapping_path)
    assert len(df) >= 15, "Expected mapping records for all 15 zones"
    assert set(df['zone_id']) == set(range(1, 16)), "All 15 zone IDs must be present"
    unmonitored = df[df['mapping_method'] == 'Unmonitored']
    assert len(unmonitored) == 7, f"Expected exactly 7 unmonitored zones, got {len(unmonitored)}"
    assert (unmonitored['mapping_confidence'] == 0.0).all()

def test_master_table_integrity():
    master_path = 'data/processed/master_table.csv'
    assert os.path.exists(master_path), "master_table.csv does not exist"
    df = pd.read_csv(master_path, low_memory=False)
    expected_rows = 17544 * 15 # 263160
    assert len(df) == expected_rows, f"Expected {expected_rows} rows, got {len(df)}"
    
    # Check required columns
    required_cols = [
        'timestamp', 'zone_id', 'zone_label', 'station_name',
        'AQI', 'aqi_observed', 'rainfall', 'rainfall_observed',
        'temperature', 'temperature_observed'
    ]
    for c in required_cols:
        assert c in df.columns, f"Missing required column {c}"
        
    # Check timestamp parsing
    ts = pd.to_datetime(df['timestamp'], errors='coerce')
    assert ts.isnull().sum() == 0, "All timestamps must parse without error"
    
    # Check temperature cutoff (no IMD temperature claimed after 2024-06-22)
    after_cutoff = df[ts > pd.Timestamp('2024-06-22 23:59:59', tz='UTC')]
    assert after_cutoff['temperature_observed'].sum() == 0, "No temperature observations allowed after 2024-06-22"
    assert after_cutoff['temperature'].dropna().empty, "Temperature values must be NaN after 2024-06-22"

def test_chronological_splits():
    train_path = 'data/processed/train.csv'
    val_path = 'data/processed/validate.csv'
    test_path = 'data/processed/test.csv'
    
    assert os.path.exists(train_path)
    assert os.path.exists(val_path)
    assert os.path.exists(test_path)
    
    df_train = pd.read_csv(train_path, low_memory=False)
    df_val = pd.read_csv(val_path, low_memory=False)
    df_test = pd.read_csv(test_path, low_memory=False)
    
    t_train_max = pd.to_datetime(df_train['timestamp']).max()
    t_val_min = pd.to_datetime(df_val['timestamp']).min()
    t_val_max = pd.to_datetime(df_val['timestamp']).max()
    t_test_min = pd.to_datetime(df_test['timestamp']).min()
    
    # Verify strict chronological separation (no leakage)
    assert t_train_max < t_val_min, f"Leakage detected: train max {t_train_max} >= val min {t_val_min}"
    assert t_val_max < t_test_min, f"Leakage detected: val max {t_val_max} >= test min {t_test_min}"
    
    total = len(df_train) + len(df_val) + len(df_test)
    assert total == 263160, f"Expected sum 263160 rows, got {total}"
