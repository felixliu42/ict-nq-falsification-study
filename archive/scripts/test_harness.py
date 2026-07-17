import pandas as pd
import numpy as np
import os
import unittest
from datetime import datetime
from research_harness import calculate_atr, get_trade_R, run_experiment

class TestResearchHarness(unittest.TestCase):
    def setUp(self):
        # Create a tiny raw price series (15 bars)
        # We'll set up a mock day of price activity starting at 09:15 EST on 2026-06-25
        # 1782393300000 ms is 2026-06-25 13:15:00 UTC (09:15:00 EST)
        start_time = 1782393300000 
        self.times = [start_time + i * 300000 for i in range(15)] # 5-min intervals
        
        self.df_raw = pd.DataFrame({
            'time': self.times,
            'open':  [100, 102, 101, 105, 106, 104, 102, 103, 106, 108, 105, 103, 104, 102, 101],
            'high':  [105, 104, 103, 110, 108, 107, 105, 115, 112, 110, 107, 105, 106, 104, 103], # Bar 7 high changed to 115
            'low':   [95,  100, 99,  101, 104, 102, 100, 101, 103, 106, 95,  101, 102, 100, 98], # Bar 10 low changed to 95
            'close': [102, 101, 102, 106, 105, 103, 102, 104, 108, 107, 104, 102, 103, 101, 100],
            'volume': [100] * 15
        })
        
        # Set up a mock ML dataset with a few setups:
        # Setup 1: Bullish sweep on Bar 3 (09:30 EST)
        # Setup 2: Bearish sweep on Bar 8 (09:55 EST)
        # Setup 3: Bullish sweep on Bar 12 (10:15 EST)
        self.df_ml = pd.DataFrame({
            'time': [self.times[3], self.times[8], self.times[12]],
            'sweep_direction': [1.0, -1.0, 1.0],
            'time_since_sweep': [2.0, 1.0, 2.0],
            'liquidity_type': [1.0, 2.0, 3.0], # Daily, 1H, 4H
            'liquidity_strength': [1.0, 1.0, 2.0], # Standard, Standard, Stacked
            'ny_session': [1.0, 1.0, 1.0],
            'london_session': [0.0, 0.0, 0.0],
            'asian_session': [0.0, 0.0, 0.0],
            'suggested_tp': [na_value() for _ in range(3)],
            'suggested_sl': [na_value() for _ in range(3)]
        })
        
        self.atr_series = calculate_atr(self.df_raw, 14)
        
    def test_calculate_atr(self):
        self.assertEqual(len(self.atr_series), 15)
        # Wilder's ATR is cumulative ewm, so first value equals simple true range
        self.assertGreater(self.atr_series.iloc[1], 0)
        
    def test_get_trade_R(self):
        row = self.df_ml.iloc[0]
        # Setup 1 (Bar 3): entry Close=106, sweep_idx = 3-2 = 1.
        # Bullish sweep, so sweep_extreme = low of Bar 1 = 100.
        # Expected R = |106 - 100| = 6.0
        R = get_trade_R(row, self.df_raw)
        self.assertEqual(R, 6.0)
        
    def test_session_filter(self):
        # 1. Morning Only configuration
        # Setup 1: Bar 3 is at 09:30 EST -> Accepted
        # Setup 2: Bar 8 is at 09:55 EST -> Accepted
        # Setup 3: Bar 12 is at 10:15 EST -> Accepted
        # Let's shift Setup 3 to 13:00 EST to test blocking
        # Bar 12 shifted to 13:00 EST (start_time + 45 min is 10:00, 13:00 is start_time + 225 min)
        df_ml_shifted = self.df_ml.copy()
        # Bar 12 in self.times is self.times[12], which is 09:15 + 60 mins = 10:15 EST.
        # Let's shift the time of setup 3 to a late hour (e.g. 15:00 EST = start_time + 345 min = 1782390900000 + 345*60*1000)
        df_ml_shifted.loc[2, 'time'] = self.times[12] + 5 * 60 * 60 * 1000 # 5 hours later (15:15 EST)
        
        config = {
            'session_filter': 'morning_only',
            'enforce_daily_bias': False,
            'stop_multiplier': 1.0,
            'allowed_liquidity_types': 'all',
            'max_setup_number': 3,
            'allowed_penetration': ['small', 'medium', 'large'],
            'tp_mode': 'split'
        }
        
        res = run_experiment(df_ml_shifted, self.df_raw, config, self.atr_series)
        # Should accept setups 1 & 2, reject setup 3
        self.assertEqual(res['total_trades'], 2)
        
    def test_daily_bias_filter(self):
        # Configuration with Single Daily Directional Bias
        # Setup 1: Bullish (sweep_direction = 1) -> Set bias to LONG
        # Setup 2: Bearish (sweep_direction = -1) -> Blocked because bias is LONG
        # Setup 3: Bullish (sweep_direction = 1) -> Accepted
        config = {
            'session_filter': 'none',
            'enforce_daily_bias': True,
            'bias_reset_mode': 'entire_day',
            'stop_multiplier': 1.0,
            'allowed_liquidity_types': 'all',
            'max_setup_number': 3,
            'allowed_penetration': ['small', 'medium', 'large'],
            'tp_mode': 'split'
        }
        
        res = run_experiment(self.df_ml, self.df_raw, config, self.atr_series)
        # Expect Setup 1 & 3 accepted, Setup 2 blocked
        self.assertEqual(res['total_trades'], 2)
        
    def test_liquidity_filter(self):
        # Configuration allowing daily_stacked_4h
        # Setup 1: Daily (liquidity_type = 1) -> Accepted
        # Setup 2: 1H (liquidity_type = 2) -> Rejected
        # Setup 3: 4H Stacked (liquidity_type = 3, strength = 2.0) -> Accepted
        config = {
            'session_filter': 'none',
            'enforce_daily_bias': False,
            'stop_multiplier': 1.0,
            'allowed_liquidity_types': 'daily_stacked_4h',
            'max_setup_number': 3,
            'allowed_penetration': ['small', 'medium', 'large'],
            'tp_mode': 'split'
        }
        
        res = run_experiment(self.df_ml, self.df_raw, config, self.atr_series)
        # Expect setups 1 & 3 accepted, setup 2 rejected
        self.assertEqual(res['total_trades'], 2)
        
    def test_setup_number_filter(self):
        # Create a mock with multiple setups on the same sweep event
        # Group setups by sweep_id: if sweep_id is the same, cumcount Cumulates
        df_ml_multi = pd.DataFrame({
            'time': [self.times[3], self.times[4], self.times[5]],
            'sweep_direction': [1.0, 1.0, 1.0],
            'time_since_sweep': [2.0, 3.0, 4.0], # All point to sweep bar index 3-2 = 1, 4-3 = 1, 5-4 = 1
            'liquidity_type': [1.0, 1.0, 1.0],
            'liquidity_strength': [1.0, 1.0, 1.0],
            'ny_session': [1.0, 1.0, 1.0],
            'london_session': [0.0, 0.0, 0.0],
            'asian_session': [0.0, 0.0, 0.0],
            'suggested_tp': [na_value() for _ in range(3)],
            'suggested_sl': [na_value() for _ in range(3)]
        })
        
        # Max setups allowed = 2
        config = {
            'session_filter': 'none',
            'enforce_daily_bias': False,
            'stop_multiplier': 1.0,
            'allowed_liquidity_types': 'all',
            'max_setup_number': 2,
            'allowed_penetration': ['small', 'medium', 'large'],
            'tp_mode': 'split'
        }
        
        res = run_experiment(df_ml_multi, self.df_raw, config, self.atr_series)
        # Expect first 2 setups accepted, 3rd setup rejected
        self.assertEqual(res['total_trades'], 2)

def na_value():
    return np.nan

if __name__ == '__main__':
    unittest.main()
