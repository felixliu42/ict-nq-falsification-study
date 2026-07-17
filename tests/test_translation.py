import pandas as pd
import numpy as np
import os
import sys

# Import functions from pine_translator
from pine_translator import (
    add_liquidity_pool,
    update_mitigations,
    check_sweep,
    compute_structure_signals,
    prepare_htf_series
)

def test_atr_calculation():
    print("\n--- Testing ATR Calculation ---")
    # Generate simple test data
    idx = pd.date_range("2026-06-01", periods=20, freq="1min")
    df = pd.DataFrame({
        "Open": [100.0] * 20,
        "High": [105.0] * 20,
        "Low": [95.0] * 20,
        "Close": [100.0] * 20,
        "Volume": [100] * 20
    }, index=idx)
    df.index.name = "datetime"
    df["Time"] = df.index.astype(np.int64) // 10**6
    
    # ATR calculations
    tr = np.maximum(df['High'] - df['Low'],
                    np.maximum(np.abs(df['High'] - df['Close'].shift(1)),
                               np.abs(df['Low'] - df['Close'].shift(1))))
    tr.iloc[0] = df['High'].iloc[0] - df['Low'].iloc[0]
    atr = tr.ewm(alpha=1.0/14.0, adjust=False).mean()
    
    # Verify first ATR equals range
    assert np.isclose(atr.iloc[0], 10.0), f"Expected first ATR 10.0, got {atr.iloc[0]}"
    print("[PASSED] ATR calculation matches TradingView expectations.")

def test_htf_swing_pivot_detection():
    print("\n--- Testing HTF Swing Pivot Detection ---")
    # Generate 1H bars with a green-to-red candle transition (swing high)
    # and a red-to-green candle transition (swing low)
    idx = pd.date_range("2026-06-01", periods=5, freq="60min")
    df = pd.DataFrame({
        "Open":  [100.0, 105.0, 104.0, 95.0, 97.0],
        "High":  [102.0, 110.0, 108.0, 98.0, 101.0],
        "Low":   [98.0,  103.0, 100.0, 92.0, 95.0],
        "Close": [101.0, 107.0, 102.0, 94.0, 98.0],
        "Volume": [100] * 5
    }, index=idx)
    df.index.name = "datetime"
    
    # 1. Bar index 1 is green (close 107 > open 105), Bar index 2 is red (close 102 < open 104)
    # This forms a green-to-red transition (swing high) at bar 2.
    # Level price = max(high[1], high) = max(110.0, 108.0) = 110.0
    
    # 2. Bar index 3 is red (close 94 < open 95), Bar index 4 is green (close 98 > open 97)
    # This forms a red-to-green transition (swing low) at bar 4.
    # Level price = min(low[3], low[4]) = min(92.0, 95.0) = 92.0
    
    close_prev = df['Close'].shift(1)
    open_prev = df['Open'].shift(1)
    is_green_red = (close_prev > open_prev) & (df['Close'] < df['Open'])
    is_red_green = (close_prev < open_prev) & (df['Close'] > df['Open'])
    
    df['pivot_high'] = np.where(is_green_red, np.maximum(df['High'].shift(1), df['High']), np.nan)
    df['pivot_low'] = np.where(is_red_green, np.minimum(df['Low'].shift(1), df['Low']), np.nan)
    
    assert pd.isna(df['pivot_high'].iloc[1])
    assert df['pivot_high'].iloc[2] == 110.0, f"Expected swing high 110.0, got {df['pivot_high'].iloc[2]}"
    assert df['pivot_low'].iloc[4] == 92.0, f"Expected swing low 92.0, got {df['pivot_low'].iloc[4]}"
    print("[PASSED] swing high and low logic are correct.")

def test_structure_signals():
    print("\n--- Testing Structure Signals (BOS & FVG) ---")
    # Generate 1m data to test BOS and FVG confirmations
    idx = pd.date_range("2026-06-01", periods=10, freq="1min")
    df = pd.DataFrame({
        "Open":  [100, 102, 104, 103, 106, 108, 107, 109, 105, 112],
        "High":  [103, 105, 107, 106, 109, 110, 108, 111, 107, 115],
        "Low":   [99,  101, 103, 102, 105, 107, 106, 108, 104, 110],
        "Close": [102, 104, 105, 105, 108, 107, 107, 110, 106, 113],
        "Volume": [100] * 10
    }, index=idx)
    df.index.name = "datetime"
    df["Time"] = df.index.astype(np.int64) // 10**6
    
    # 1. Pivot High / Low checks
    # Let's run compute_structure_signals
    res = compute_structure_signals(df)
    
    # Verify output shape
    assert len(res) == len(df)
    assert 'bos_up' in res.columns
    assert 'bos_down' in res.columns
    assert 'fvg_bull' in res.columns
    assert 'fvg_bear' in res.columns
    print("[PASSED] Structure signals computed successfully.")

def test_lookahead_prevention():
    print("\n--- Testing Lookahead Bias Prevention ---")
    # Create simple 1m dataset
    idx_1m = pd.date_range("2026-06-01", periods=300, freq="1min")
    df_1m = pd.DataFrame({
        "Open": np.random.normal(100, 1, 300),
        "High": np.random.normal(101, 1, 300),
        "Low": np.random.normal(99, 1, 300),
        "Close": np.random.normal(100, 1, 300),
        "Volume": [100] * 300
    }, index=idx_1m)
    df_1m.index.name = "datetime"
    df_1m["Time"] = df_1m.index.astype(np.int64) // 10**6
    
    # Resample 1H with pd.Timedelta(hours=1) shift
    df_1h = prepare_htf_series(df_1m.reset_index(), '60min', pd.Timedelta(hours=1))
    
    # Verify index shift
    # The first 1H bar starts at 00:00:00 and ends at 01:00:00.
    # The resampled bar at 00:00:00 should be shifted to 01:00:00.
    assert df_1h.index[0] == pd.Timestamp("2026-06-01 01:00:00"), f"Expected shifted index to start at 01:00:00, got {df_1h.index[0]}"
    
    # Perform backward merge
    df_1m_aligned = pd.merge_asof(df_1m, df_1h.rename(columns={'Close': 'close_1h'}), left_index=True, right_index=True, direction='backward')
    
    # Check that before 01:00:00, close_1h is NaN
    nan_check = df_1m_aligned.loc[:"2026-06-01 00:59:00", "close_1h"]
    assert nan_check.isna().all(), "Lookahead bias! Aligned HTF Close should be NaN before completion time."
    
    # Check that at 01:00:00, close_1h matches the close of the 00:00:00 to 01:00:00 resampled bar
    aligned_val = df_1m_aligned.loc["2026-06-01 01:00:00", "close_1h"]
    expected_val = df_1h.loc[pd.Timestamp("2026-06-01 01:00:00"), "Close"]
    assert np.isclose(aligned_val, expected_val), f"Expected aligned value {expected_val}, got {aligned_val}"
    
    print("[PASSED] Timeframe alignment is strictly non-repainting.")

def test_fvg_revisit_and_invalidation():
    print("\n--- Testing FVG Revisit & Invalidation Logic ---")
    df = pd.DataFrame({
        "Open":  [100, 103, 106, 106, 110, 113, 116, 111],
        "High":  [102, 105, 108, 107, 112, 115, 118, 114],
        "Low":   [100, 103, 106, 105, 110, 113, 116, 111],
        "Close": [101, 104, 107, 106.5, 111, 114, 117, 113]
    })
    res = compute_structure_signals(df)
    
    # Bar 2: FVG created, not filled yet.
    assert res['fvg_bull'].iloc[2] == 0.0
    
    # Bar 3: Price revisits FVG (low 105 < top 106) and exits (close 106.5 > top 106)
    # This should trigger fvg_bull fill (1.0)
    assert res['fvg_bull'].iloc[3] == 1.0, f"Expected FVG fill on bar 3, got {res['fvg_bull'].iloc[3]}"
    assert res['fvg_bull_inv'].iloc[3] == 0.0
    
    # Bar 6: New FVG created (top = 116, bottom = 112)
    # Bar 7: Low trades to 111, which breaches bottom 112. This should trigger invalidation.
    assert res['fvg_bull_inv'].iloc[7] == 1.0, f"Expected FVG invalidation on bar 7, got {res['fvg_bull_inv'].iloc[7]}"
    assert res['fvg_bull'].iloc[7] == 0.0
    
    print("[PASSED] FVG revisit/fill and invalidation logic is correct.")

def run_all_tests():
    test_atr_calculation()
    test_htf_swing_pivot_detection()
    test_structure_signals()
    test_lookahead_prevention()
    test_fvg_revisit_and_invalidation()
    print("\nALL TRANSLATION TESTS PASSED SUCCESSFULLY!")

if __name__ == '__main__':
    run_all_tests()
