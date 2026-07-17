import pandas as pd
import numpy as np
import os
import sys

from backtest import clean_columns, simulate_trade_execution, calculate_metrics

def test_clean_columns():
    print("\n--- Testing: clean_columns ---")
    data = {
        'Time (Export)': [1, 2],
        'Suggested_TP (Export)': [10, 20],
        'Suggested_SL (Export)': [5, 15]
    }
    df = pd.DataFrame(data)
    df_cleaned = clean_columns(df)
    assert 'time' in df_cleaned.columns
    assert 'suggested_tp' in df_cleaned.columns
    assert 'suggested_sl' in df_cleaned.columns
    print("clean_columns test passed!")

def test_dynamic_tpsl_and_duration():
    print("\n--- Testing: simulate_trade_execution (Dynamic TP/SL & Duration) ---")
    # Raw data: 10 bars
    # Time goes from 1000 to 1090
    df_raw = pd.DataFrame({
        'time': [1000, 1010, 1020, 1030, 1040, 1050, 1060, 1070, 1080, 1090],
        'open':  [100,  102,  101,  104,  103,  106,  108,  107,  111,  110],
        'high':  [105,  104,  103,  106,  105,  107,  110,  109,  115,  112],
        'low':   [95,   100,  99,   101,  100,  102,  105,  104,  108,  106],
        'close': [102,  101,  102,  103,  102,  105,  107,  106,  110,  108]
    })
    
    # Test case 1: Fallback (no suggested TP/SL)
    # Entry at bar 2 (time=1020, close=102), sweep at bar 0 (time=1000, low=95)
    # R = 102 - 95 = 7.0
    # Bullish setup (sweep_direction = 1)
    # tp1 fallback = 102 + 2 * 7 = 116. stop_loss fallback = 95
    # High reaches 115 on bar 8, Low never hits 95. End of data reached.
    row_fallback = pd.Series({
        'time': 1020,
        'sweep_direction': 1,
        'time_since_sweep': 2,
    })
    
    ret, outcome, duration = simulate_trade_execution(row_fallback, df_raw, tp_mode='tp1')
    assert outcome == 'end_of_data'
    # entry idx = 2, final bar idx = 9. duration should be 9 - 2 = 7
    assert duration == 7
    # final return = (108 - 102) / 7.0 = 6/7 = 0.857...
    assert np.isclose(ret, 6.0 / 7.0)
    
    # Test case 2: Custom suggested TP/SL
    # Entry at bar 2 (time=1020, close=102)
    # Custom TP = 110 (hits on bar 6), SL = 100 (never hit)
    # R = 7.0 (based on sweep extreme)
    # actual_rr = abs(110 - 102) / 7.0 = 8.0 / 7.0 = 1.1428...
    # Exits on bar 6 (duration = 6 - 2 = 4)
    row_custom = pd.Series({
        'time': 1020,
        'sweep_direction': 1,
        'time_since_sweep': 2,
        'suggested_tp': 110.0,
        'suggested_sl': 98.0
    })
    
    ret, outcome, duration = simulate_trade_execution(row_custom, df_raw, tp_mode='tp1')
    assert outcome == 'tp1'
    assert duration == 4
    assert np.isclose(ret, 8.0 / 7.0)
    
    print("simulate_trade_execution tests passed!")

def test_calculate_metrics():
    print("\n--- Testing: calculate_metrics ---")
    returns = [2.0, -1.0, 3.0, -0.5, 0.0]
    durations = [5, 2, 10, 3, 4]
    
    metrics = calculate_metrics(returns, durations)
    assert metrics['total_trades'] == 5
    # wins: 2.0, 3.0. win_rate = 2 / 5 = 0.4
    assert np.isclose(metrics['win_rate'], 0.4)
    # gains: 5.0, losses: 1.5. profit_factor = 5.0 / 1.5 = 3.333
    assert np.isclose(metrics['profit_factor'], 5.0 / 1.5)
    # expectancy = sum(returns) / 5 = 3.5 / 5 = 0.7
    assert np.isclose(metrics['expectancy'], 0.7)
    
    # avg_win = mean([2.0, 3.0]) = 2.5
    assert np.isclose(metrics['avg_win'], 2.5)
    # avg_loss = mean([1.0, 0.5]) = 0.75
    assert np.isclose(metrics['avg_loss'], 0.75)
    # win_loss_ratio = 2.5 / 0.75 = 3.333...
    assert np.isclose(metrics['win_loss_ratio'], 2.5 / 0.75)
    # avg_duration = mean([5, 2, 10, 3, 4]) = 4.8
    assert np.isclose(metrics['avg_duration'], 4.8)
    
    # Test division by zero cases
    empty_metrics = calculate_metrics([], [])
    assert empty_metrics['total_trades'] == 0
    assert empty_metrics['win_loss_ratio'] == 0.0
    
    all_wins = calculate_metrics([2.0], [5])
    assert all_wins['win_loss_ratio'] == np.inf
    
    print("calculate_metrics tests passed!")

def run_all_tests():
    test_clean_columns()
    test_dynamic_tpsl_and_duration()
    test_calculate_metrics()
    print("\nALL BACKTEST REFRACTOR TESTS PASSED SUCCESSFULLY!")

if __name__ == '__main__':
    run_all_tests()
