import pandas as pd
import numpy as np
import os
from pipeline import process_pipeline

def test_without_suggested_cols():
    print("\n--- Running Test: Missing Suggested TP/SL Columns (Backwards Compatibility) ---")
    data = {
        'Time': [1000, 1005, 1010, 1015, 1020, 1025, 1030, 1035],
        'Open': [100, 95, 97, 101, 108, 120, 130, 125],
        'High': [105, 98, 102, 110, 125, 135, 132, 136],
        'Low': [90, 94, 96, 99, 105, 118, 124, 120],
        'Close': [95, 97, 101, 108, 122, 130, 125, 132],
        'Volume': [100, 100, 100, 100, 100, 100, 100, 100],
        
        # indicator outputs (with mock TV names)
        'sweep_direction (MNQ Export)': [np.nan, np.nan, 1.0, np.nan, np.nan, np.nan, -1.0, np.nan],
        'liquidity_type (MNQ Export)': [np.nan, np.nan, 3.0, np.nan, np.nan, np.nan, 2.0, np.nan],
        'liquidity_strength (MNQ Export)': [np.nan, np.nan, 0.8, np.nan, np.nan, np.nan, 0.6, np.nan],
        'bos_up_strength (MNQ Export)': [np.nan, np.nan, 1.0, np.nan, np.nan, np.nan, np.nan, np.nan],
        'bos_down_strength (MNQ Export)': [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0.0, np.nan],
        'bullish_fvg_rejected (MNQ Export)': [np.nan, np.nan, 0.0, np.nan, np.nan, np.nan, np.nan, np.nan],
        'bearish_fvg_rejected (MNQ Export)': [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 1.0, np.nan],
        'retracement_depth (MNQ Export)': [np.nan, np.nan, 0.5, np.nan, np.nan, np.nan, 0.4, np.nan],
        'distance_to_equilibrium (MNQ Export)': [np.nan, np.nan, 0.01, np.nan, np.nan, np.nan, -0.02, np.nan],
        'time_since_sweep (MNQ Export)': [np.nan, np.nan, 2.0, np.nan, np.nan, np.nan, 1.0, np.nan],
        'ny_session (MNQ Export)': [np.nan, np.nan, 1.0, np.nan, np.nan, np.nan, 1.0, np.nan],
        'london_session (MNQ Export)': [np.nan, np.nan, 0.0, np.nan, np.nan, np.nan, 0.0, np.nan],
        'asian_session (MNQ Export)': [np.nan, np.nan, 0.0, np.nan, np.nan, np.nan, 0.0, np.nan]
    }
    
    df = pd.DataFrame(data)
    mock_input = "mock_tv_export_no_suggested.csv"
    mock_output = "mock_ml_dataset_no_suggested.csv"
    
    df.to_csv(mock_input, index=False)
    
    try:
        res_df = process_pipeline(mock_input, mock_output)
        
        assert len(res_df) == 2, f"Expected 2 trade sequences, got {len(res_df)}"
        
        # Sequence 1: Bullish Trade on Bar Index 2
        # Entry close = 101, sweep_extreme = 90. R = 11.
        # Fallback Target = 101 + 22 = 123. Fallback SL = 90.
        s1 = res_df.iloc[0]
        assert s1['suggested_tp'] == 123.0, f"Expected fallback TP 123.0, got {s1['suggested_tp']}"
        assert s1['suggested_sl'] == 90.0, f"Expected fallback SL 90.0, got {s1['suggested_sl']}"
        assert s1['label'] == 1, "Expected successful long trade (label=1)"
        assert s1['time_to_target'] == 2
        
        # Sequence 2: Bearish Trade on Bar Index 6
        # Entry close = 125, sweep_extreme = 135. R = 10.
        # Fallback Target = 125 - 20 = 105. Fallback SL = 135.
        s2 = res_df.iloc[1]
        assert s2['suggested_tp'] == 105.0, f"Expected fallback TP 105.0, got {s2['suggested_tp']}"
        assert s2['suggested_sl'] == 135.0, f"Expected fallback SL 135.0, got {s2['suggested_sl']}"
        assert s2['label'] == 0, "Expected stopped out short trade (label=0)"
        assert s2['time_to_target'] == 1
        
        print("Missing columns (backwards compatibility) test PASSED!")
        
    finally:
        if os.path.exists(mock_input):
            os.remove(mock_input)
        if os.path.exists(mock_output):
            os.remove(mock_output)

def test_with_suggested_cols():
    print("\n--- Running Test: Suggested TP/SL Columns Present (Custom & Fallback NaN) ---")
    data = {
        'Time': [1000, 1005, 1010, 1015, 1020, 1025, 1030, 1035],
        'Open': [100, 95, 97, 101, 108, 120, 130, 125],
        'High': [105, 98, 102, 110, 125, 135, 132, 136],
        'Low': [90, 94, 96, 99, 105, 118, 124, 120],
        'Close': [95, 97, 101, 108, 122, 130, 125, 132],
        'Volume': [100, 100, 100, 100, 100, 100, 100, 100],
        
        # indicator outputs
        'sweep_direction (MNQ Export)': [np.nan, np.nan, 1.0, np.nan, np.nan, np.nan, -1.0, np.nan],
        'liquidity_type (MNQ Export)': [np.nan, np.nan, 3.0, np.nan, np.nan, np.nan, 2.0, np.nan],
        'liquidity_strength (MNQ Export)': [np.nan, np.nan, 0.8, np.nan, np.nan, np.nan, 0.6, np.nan],
        'bos_up_strength (MNQ Export)': [np.nan, np.nan, 1.0, np.nan, np.nan, np.nan, np.nan, np.nan],
        'bos_down_strength (MNQ Export)': [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0.0, np.nan],
        'bullish_fvg_rejected (MNQ Export)': [np.nan, np.nan, 0.0, np.nan, np.nan, np.nan, np.nan, np.nan],
        'bearish_fvg_rejected (MNQ Export)': [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 1.0, np.nan],
        'retracement_depth (MNQ Export)': [np.nan, np.nan, 0.5, np.nan, np.nan, np.nan, 0.4, np.nan],
        'distance_to_equilibrium (MNQ Export)': [np.nan, np.nan, 0.01, np.nan, np.nan, np.nan, -0.02, np.nan],
        'time_since_sweep (MNQ Export)': [np.nan, np.nan, 2.0, np.nan, np.nan, np.nan, 1.0, np.nan],
        'ny_session (MNQ Export)': [np.nan, np.nan, 1.0, np.nan, np.nan, np.nan, 1.0, np.nan],
        'london_session (MNQ Export)': [np.nan, np.nan, 0.0, np.nan, np.nan, np.nan, 0.0, np.nan],
        'asian_session (MNQ Export)': [np.nan, np.nan, 0.0, np.nan, np.nan, np.nan, 0.0, np.nan],
        
        # New suggested TP/SL columns with suffixes
        'suggested_tp (MNQ Export)': [np.nan, np.nan, 115.0, np.nan, np.nan, np.nan, np.nan, np.nan],
        'suggested_sl (MNQ Export)': [np.nan, np.nan, 95.0, np.nan, np.nan, np.nan, np.nan, np.nan]
    }
    
    df = pd.DataFrame(data)
    mock_input = "mock_tv_export_suggested.csv"
    mock_output = "mock_ml_dataset_suggested.csv"
    
    df.to_csv(mock_input, index=False)
    
    try:
        res_df = process_pipeline(mock_input, mock_output)
        
        assert len(res_df) == 2, f"Expected 2 trade sequences, got {len(res_df)}"
        
        # Sequence 1: Bullish Trade on Bar Index 2
        # Custom values: Target = 115.0, SL = 95.0
        # Check outcomes at:
        # - Bar 3: high = 110, low = 99 (no hit)
        # - Bar 4: high = 125, low = 105 (hits custom target 115.0!)
        # Label = 1, time_to_target = 2.
        s1 = res_df.iloc[0]
        assert s1['suggested_tp'] == 115.0, f"Expected custom TP 115.0, got {s1['suggested_tp']}"
        assert s1['suggested_sl'] == 95.0, f"Expected custom SL 95.0, got {s1['suggested_sl']}"
        assert s1['label'] == 1, "Expected successful long trade (label=1)"
        assert s1['time_to_target'] == 2, f"Expected time_to_target = 2, got {s1['time_to_target']}"
        
        # Sequence 2: Bearish Trade on Bar Index 6
        # Both suggested_tp and suggested_sl are NaN.
        # Should fall back to: Target = 105.0, SL = 135.0
        s2 = res_df.iloc[1]
        assert s2['suggested_tp'] == 105.0, f"Expected fallback TP 105.0, got {s2['suggested_tp']}"
        assert s2['suggested_sl'] == 135.0, f"Expected fallback SL 135.0, got {s2['suggested_sl']}"
        assert s2['label'] == 0, "Expected stopped out short trade (label=0)"
        assert s2['time_to_target'] == 1
        
        print("Suggested columns (custom & NaN fallback) test PASSED!")
        
    finally:
        if os.path.exists(mock_input):
            os.remove(mock_input)
        if os.path.exists(mock_output):
            os.remove(mock_output)

def run_all_tests():
    test_without_suggested_cols()
    test_with_suggested_cols()
    print("\nALL PIPELINE TESTS PASSED SUCCESSFULLY!")

if __name__ == '__main__':
    run_all_tests()
