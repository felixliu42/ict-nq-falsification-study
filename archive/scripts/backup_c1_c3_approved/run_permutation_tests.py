import os
import sys
import subprocess
import pandas as pd
import numpy as np
import traceback
from backtest import simulate_trade_execution, get_trade_R, clean_columns
from evaluate_compounding_regimes import calculate_monthly_sharpe
from evaluate_regime_balancing import run_walk_forward

TEST_YEARS = [2020, 2021, 2022, 2023, 2024, 2025]

def create_temp_translator(c1_on, c3_on):
    """
    Read pine_translator.py and apply the string replacements for the active configuration.
    """
    with open("pine_translator.py", "r") as f:
        code = f.read()
        
    # Normalize line endings to LF for Windows compatibility
    code = code.replace("\r\n", "\n")
        
    # --- 1. ALWAYS ADD SPLIT FVG COLUMNS TO ARRAYS & EXPORTS ---
    # Declare output arrays in main()
    code = code.replace(
        "    out_bearish_fvg_rejected = np.full(n, np.nan, dtype=np.float32)",
        "    out_bearish_fvg_rejected = np.full(n, np.nan, dtype=np.float32)\n    out_opp_fvg_rejected = np.full(n, np.nan, dtype=np.float32)\n    out_same_fvg_filled = np.full(n, np.nan, dtype=np.float32)"
    )
    
    # Initialize variables at the start of main()'s chronological loop (line 617) to prevent UnboundLocalError
    code = code.replace(
        "    for i in range(n):\n        bar_time = times[i]",
        "    for i in range(n):\n        opp_fvg_occurred = False\n        same_fvg_occurred = False\n        bar_time = times[i]"
    )
    
    # Bullish setup FVG logs (reconstruction loop)
    code = code.replace(
        "                            if bos_ev:\n                                bos_occurred = True\n                            if inv_ev:\n                                fvg_occurred = True\n                                \n                            if fill_ev or eq_touch:\n                                continuation_confirmed = True\n                            if fill_ev:\n                                fvg_occurred = True",
        "                            if bos_ev:\n                                bos_occurred = True\n                            if inv_ev:\n                                fvg_occurred = True\n                                opp_fvg_occurred = True\n                                \n                            if fill_ev or eq_touch:\n                                continuation_confirmed = True\n                            if fill_ev:\n                                fvg_occurred = True\n                                same_fvg_occurred = True"
    )
    
    # Bearish setup FVG logs (reconstruction loop)
    code = code.replace(
        "                            if bos_ev:\n                                bos_occurred = True\n                            if inv_ev:\n                                fvg_occurred = True\n                                \n                            if fill_ev or eq_touch:\n                                continuation_confirmed = True\n                            if fill_ev:\n                                fvg_occurred = True",
        "                            if bos_ev:\n                                bos_occurred = True\n                            if inv_ev:\n                                fvg_occurred = True\n                                opp_fvg_occurred = True\n                                \n                            if fill_ev or eq_touch:\n                                continuation_confirmed = True\n                            if fill_ev:\n                                fvg_occurred = True\n                                same_fvg_occurred = True"
    )
    
    # Assign arrays inside setup block
    code = code.replace(
        "            if sweep_direction_val == 1:\n                out_bos_up_strength[i] = 1.0 if bos_confirmed else 0.0\n                out_bullish_fvg_rejected[i] = 1.0 if fvg_confirmed else 0.0\n            elif sweep_direction_val == -1:\n                out_bos_down_strength[i] = 1.0 if bos_confirmed else 0.0\n                out_bearish_fvg_rejected[i] = 1.0 if fvg_confirmed else 0.0",
        "            if sweep_direction_val == 1:\n                out_bos_up_strength[i] = 1.0 if bos_confirmed else 0.0\n                out_bullish_fvg_rejected[i] = 1.0 if fvg_confirmed else 0.0\n            elif sweep_direction_val == -1:\n                out_bos_down_strength[i] = 1.0 if bos_confirmed else 0.0\n                out_bearish_fvg_rejected[i] = 1.0 if fvg_confirmed else 0.0\n            \n            out_opp_fvg_rejected[i] = 1.0 if opp_fvg_occurred else 0.0\n            out_same_fvg_filled[i] = 1.0 if same_fvg_occurred else 0.0"
    )
    
    # Save to df_raw
    code = code.replace(
        "    df_raw['bearish_fvg_rejected' + suffix] = out_bearish_fvg_rejected",
        "    df_raw['bearish_fvg_rejected' + suffix] = out_bearish_fvg_rejected\n    df_raw['opp_fvg_rejected' + suffix] = out_opp_fvg_rejected\n    df_raw['same_fvg_filled' + suffix] = out_same_fvg_filled"
    )
    
    # Save to CSV columns
    code = code.replace(
        "        'bearish_fvg_rejected' + suffix,",
        "        'bearish_fvg_rejected' + suffix,\n        'opp_fvg_rejected' + suffix,\n        'same_fvg_filled' + suffix,"
    )
    
    # --- 2. CONDITIONALS ---
    if c1_on:
        # Reversal only confirmed if BOS has occurred
        code = code.replace("if bos_ev or inv_ev:", "if bos_ev:")
        
    if c3_on:
        # Running swing extremes for equilibrium (tracked inside StructureEngine)
        code = code.replace(
            "        self.last_ph = None\n        self.last_pl = None",
            "        self.last_ph = None\n        self.last_pl = None\n        self.latest_ph = None\n        self.latest_pl = None"
        )
        code = code.replace(
            "        if ph is not None: self.last_ph = ph\n        if pl is not None: self.last_pl = pl",
            "        if ph is not None:\n            self.last_ph = ph\n            self.latest_ph = ph\n        if pl is not None:\n            self.last_pl = pl\n            self.latest_pl = pl"
        )
        
        # Replace equilibrium calculations using struct_1m's latest_ph/latest_pl
        old_eq = """            impulse_size = abs(sweep_extreme - current_extreme)
            ret_depth = 0.0
            if impulse_size > 0.0:
                if sweep_dir == -1:
                    ret_depth = max(0.0, min(1.0, (c - current_extreme) / impulse_size))
                else:
                    ret_depth = max(0.0, min(1.0, (current_extreme - c) / impulse_size))
            
            equilibrium_val = (sweep_extreme + current_extreme) / 2.0
            dist_eq = 0.0
            if c != 0.0:
                dist_eq = sweep_dir * (c - equilibrium_val) / c"""
                
        new_eq = """            high_extreme = struct_1m.latest_ph if struct_1m.latest_ph is not None else h
            low_extreme = struct_1m.latest_pl if struct_1m.latest_pl is not None else l
            impulse_size = abs(high_extreme - low_extreme)
            ret_depth = 0.0
            if impulse_size > 0.0:
                if sweep_dir == -1:
                    ret_depth = max(0.0, min(1.0, (c - low_extreme) / impulse_size))
                else:
                    ret_depth = max(0.0, min(1.0, (high_extreme - c) / impulse_size))
            
            equilibrium_val = (high_extreme + low_extreme) / 2.0
            dist_eq = 0.0
            if c != 0.0:
                dist_eq = sweep_dir * (c - equilibrium_val) / c"""
                
        code = code.replace(old_eq, new_eq)
        
    with open("temp_translator.py", "w") as f:
        f.write(code)

def run_test_configuration(c1, c2, c3):
    """
    Generate indicators with temp_translator.py, label with pipeline.py, 
    and run 6-year walk-forward backtest.
    """
    create_temp_translator(c1, c3)
    
    raw_dfs = []
    ml_dfs = []
    
    for year in TEST_YEARS:
        raw_path = f"data/MNQ_{year}/raw_data.csv"
        temp_tv_export = f"data/temp_tv_export_{year}.csv"
        temp_labeled_path = f"data/temp_labeled_{year}.csv"
        
        # 1. Run temp_translator.py to compute the custom indicators
        # We run it as a script, passing raw_path and output path
        cmd = [sys.executable, "temp_translator.py", "--input", raw_path, "--output", temp_tv_export]
        subprocess.run(cmd, check=True)
        
        # 2. Run pipeline.py to label setups
        from pipeline import process_pipeline
        process_pipeline(input_csv=temp_tv_export, output_csv=temp_labeled_path)
        
        # Load labeled data
        df_ml_yr = pd.read_csv(temp_labeled_path)
        ml_dfs.append(df_ml_yr)
        raw_dfs.append(pd.read_csv(raw_path))
        
        # Clean up temp files
        if os.path.exists(temp_tv_export): os.remove(temp_tv_export)
        if os.path.exists(temp_labeled_path): os.remove(temp_labeled_path)
        
    # Combine datasets
    df_raw_combined = pd.concat(raw_dfs, ignore_index=True)
    df_raw_combined = clean_columns(df_raw_combined)
    df_raw_combined['time'] = pd.to_numeric(df_raw_combined['time'], errors='coerce')
    df_raw_combined = df_raw_combined.sort_values('time').drop_duplicates(subset=['time']).reset_index(drop=True)
    
    df_ml_combined = pd.concat(ml_dfs, ignore_index=True)
    
    # Rename custom columns to remove the TradingView suffix
    rename_dict = {}
    for col in df_ml_combined.columns:
        if 'opp_fvg_rejected' in col:
            rename_dict[col] = 'opp_fvg_rejected'
        elif 'same_fvg_filled' in col:
            rename_dict[col] = 'same_fvg_filled'
    df_ml_combined = df_ml_combined.rename(columns=rename_dict)
    
    df_ml_combined['time'] = pd.to_numeric(df_ml_combined['time'], errors='coerce')
    df_ml_combined = df_ml_combined.sort_values('time').drop_duplicates(subset=['time']).reset_index(drop=True)
    
    # Features selection based on Change 2
    if c2:
        # Split FVG features
        features = [
            'liquidity_type', 'liquidity_strength', 'sweep_direction', 'sweep_size',
            'bos_strength', 'opp_fvg_rejected', 'same_fvg_filled', 'retracement_depth', 'time_since_sweep',
            'ny_session', 'london_session', 'asian_session'
        ]
    else:
        # Standard combined FVG features
        features = [
            'liquidity_type', 'liquidity_strength', 'sweep_direction', 'sweep_size',
            'bos_strength', 'fvg_rejected', 'retracement_depth', 'time_since_sweep',
            'ny_session', 'london_session', 'asian_session'
        ]
        
    # Run 6-Month rolling walk-forward validation (W=1800)
    df_oos = run_walk_forward(df_ml_combined, window_size=1800, features=features)
    df_oos['dt'] = pd.to_datetime(df_oos['time'], unit='ms').dt.tz_localize('UTC').dt.tz_convert('America/New_York')
    df_oos['year_num'] = df_oos['dt'].dt.year
    
    trades_df = df_oos[(df_oos['pred_prob'] > 0.25) & (df_oos['ny_session'] == 1.0)].copy()
    
    trade_outcomes = []
    for _, row in trades_df.iterrows():
        R = get_trade_R(row, df_raw_combined)
        if R is not None:
            if R > 60.0:
                continue
            ret, outcome, duration = simulate_trade_execution(row, df_raw_combined, tp_mode='split')
            if ret is not None:
                dt = pd.to_datetime(row['time'], unit='ms').tz_localize('UTC').tz_convert('America/New_York').date()
                trade_outcomes.append({
                    'ret': ret,
                    'R': R,
                    'date': dt,
                    'year': row['year_num']
                })
                
    active_years = sorted(list(set([t['year'] for t in trade_outcomes])))
    results = []
    flat_balances = [100000.0]
    
    for yr in active_years:
        yr_trades = [t for t in trade_outcomes if t['year'] == yr]
        n_trades = len(yr_trades)
        if n_trades == 0:
            continue
            
        flat_yr_pnls = []
        flat_bal_start = flat_balances[-1]
        flat_yr_balances = [flat_bal_start]
        flat_yr_dates = []
        
        for t in yr_trades:
            p_flat = 16.0 * 2.0 * t['R'] * t['ret']
            flat_yr_pnls.append(p_flat)
            flat_balances.append(flat_balances[-1] + p_flat)
            flat_yr_balances.append(flat_yr_balances[-1] + p_flat)
            flat_yr_dates.append(t['date'])
            
        flat_ret = (flat_yr_balances[-1] - flat_bal_start) / flat_bal_start * 100
        
        flat_running_max = np.maximum.accumulate(flat_yr_balances)
        flat_dds = (flat_running_max - flat_yr_balances) / flat_running_max * 100
        flat_max_dd = np.max(flat_dds)
        
        start_date = f"{yr}-01-01"
        end_date = f"{yr}-12-31"
        flat_sharpe = calculate_monthly_sharpe(flat_yr_pnls, flat_yr_dates, start_date, end_date, is_compounded=False, initial_balance=flat_bal_start)
        
        results.append({
            'year': yr,
            'trades': n_trades,
            'flat_ret': flat_ret,
            'flat_dd': flat_max_dd,
            'flat_sharpe': flat_sharpe
        })
        
    avg_trades = np.mean([r['trades'] for r in results]) if results else 0
    avg_flat_ret = np.mean([r['flat_ret'] for r in results]) if results else 0
    avg_flat_dd = np.mean([r['flat_dd'] for r in results]) if results else 0
    avg_flat_sharpe = np.mean([r['flat_sharpe'] for r in results]) if results else 0
    
    return avg_trades, avg_flat_ret, avg_flat_dd, avg_flat_sharpe

def main():
    permutations = [
        (False, False, False, "Baseline (All OFF)"),
        (True, False, False, "Change 1 ON (BOS Trigger Only)"),
        (False, True, False, "Change 2 ON (Split FVG Features)"),
        (False, False, True, "Change 3 ON (Dealing Range Equilibrium)"),
        (True, True, False, "Change 1 & 2 ON"),
        (True, False, True, "Change 1 & 3 ON"),
        (False, True, True, "Change 2 & 3 ON"),
        (True, True, True, "All Changes ON (1, 2, & 3)")
    ]
    
    leaderboard = []
    
    for c1, c2, c3, name in permutations:
        print(f"\n=======================================================")
        print(f"RUNNING PERMUTATION: {name}")
        print(f"=======================================================")
        try:
            trades, ret, dd, sharpe = run_test_configuration(c1, c2, c3)
            leaderboard.append({
                "Configuration": name,
                "Avg Trades": f"{trades:.1f}",
                "Avg Return": f"{ret:.1f}%",
                "Avg Max DD": f"{dd:.1f}%",
                "Monthly Sharpe": f"{sharpe:.2f}"
            })
            print(f"Completed: Return={ret:.1f}%, DD={dd:.1f}%, Sharpe={sharpe:.2f}")
        except Exception as e:
            traceback.print_exc()
            print(f"Failed permutation {name}: {e}")
            
    # Clean up temp_translator.py at end
    if os.path.exists("temp_translator.py"):
        os.remove("temp_translator.py")
        
    df_leaderboard = pd.DataFrame(leaderboard)
    print("\n" + "="*80)
    print("ICT FEATURE MODIFICATION PERMUTATION LEADERBOARD (2020-2025)")
    print("="*80)
    print(df_leaderboard.to_string(index=False))
    print("="*80)
    
    df_leaderboard.to_csv("results_permutation_tests.csv", index=False)
    print("Results saved to results_permutation_tests.csv")

if __name__ == '__main__':
    main()
