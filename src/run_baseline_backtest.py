import os
import sys
import subprocess
import traceback
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pipeline import clean_columns
from walk_forward import run_walk_forward
from backtest import get_trade_R, simulate_trade_execution
from metrics import calculate_monthly_sharpe

SRC_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("ICT_DATA_DIR", SRC_DIR.parent / "data"))

TEST_YEARS = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]

def main():
    raw_dfs = []
    ml_dfs = []

    print("================================================================================")
    # 1. Generate indicators and label data for each year
    for year in TEST_YEARS:
        raw_path = str(DATA_DIR / f"MNQ_{year}" / "raw_data.csv")
        temp_tv_export = str(DATA_DIR / f"temp_tv_export_{year}.csv")
        temp_labeled_path = str(DATA_DIR / f"temp_labeled_{year}.csv")

        print(f"Running pine_translator.py for year {year}...")
        cmd = [sys.executable, str(SRC_DIR / "pine_translator.py"), "--input", raw_path, "--output", temp_tv_export]
        subprocess.run(cmd, check=True)
        
        print(f"Running pipeline.py for year {year}...")
        from pipeline import process_pipeline
        process_pipeline(input_csv=temp_tv_export, output_csv=temp_labeled_path)
        
        df_ml_yr = pd.read_csv(temp_labeled_path)
        ml_dfs.append(df_ml_yr)
        raw_dfs.append(pd.read_csv(raw_path))
        
        if os.path.exists(temp_tv_export): os.remove(temp_tv_export)
        if os.path.exists(temp_labeled_path): os.remove(temp_labeled_path)
        
    print("================================================================================")
    print("Combining datasets...")
    df_raw_combined = pd.concat(raw_dfs, ignore_index=True)
    df_raw_combined = clean_columns(df_raw_combined)
    df_raw_combined['time'] = pd.to_numeric(df_raw_combined['time'], errors='coerce')
    df_raw_combined = df_raw_combined.sort_values('time').drop_duplicates(subset=['time']).reset_index(drop=True)
    
    df_ml_combined = pd.concat(ml_dfs, ignore_index=True)
    df_ml_combined['time'] = pd.to_numeric(df_ml_combined['time'], errors='coerce')
    df_ml_combined = df_ml_combined.sort_values('time').drop_duplicates(subset=['time']).reset_index(drop=True)
    
    # Standard combined FVG features list
    features = [
        'liquidity_type', 'liquidity_strength', 'sweep_direction', 'sweep_size',
        'bos_strength', 'fvg_rejected', 'retracement_depth', 'time_since_sweep',
        'ny_session', 'london_session', 'asian_session'
    ]
    
    print("Running walk-forward validation...")
    df_oos = run_walk_forward(df_ml_combined, window_size=180 * 24 * 3600 * 1000, features=features)
    df_oos['dt'] = pd.to_datetime(df_oos['time'], unit='ms').dt.tz_localize('UTC').dt.tz_convert('America/New_York')
    df_oos['year_num'] = df_oos['dt'].dt.year
    
    trades_df = df_oos[(df_oos['pred_prob'] > 0.25) & (df_oos['ny_session'] == 1.0)].copy()
    print(f"Simulating trade execution for {len(trades_df)} prospective setups...")
    
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
    
    print("================================================================================")
    print("FINAL PERFORMANCE STATS BY YEAR (Change 1 & 3 ON)")
    print("================================================================================")
    
    flat_balances = [100000.0]
    results = []
    
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
        print(f"Year {yr}: Trades = {n_trades}, Return = {flat_ret:.2f}%, Max Drawdown = {flat_max_dd:.2f}%, Sharpe = {flat_sharpe:.2f}")
        
    print("================================================================================")
    print("AVERAGE METRICS OVER BACKTEST CYCLE")
    print("================================================================================")
    avg_trades = np.mean([r['trades'] for r in results]) if results else 0
    avg_flat_ret = np.mean([r['flat_ret'] for r in results]) if results else 0
    avg_flat_dd = np.mean([r['flat_dd'] for r in results]) if results else 0
    avg_flat_sharpe = np.mean([r['flat_sharpe'] for r in results]) if results else 0
    
    print(f"Average Annual Trades: {avg_trades:.1f}")
    print(f"Average Annual Return: {avg_flat_ret:.1f}%")
    print(f"Average Max Drawdown: {avg_flat_dd:.1f}%")
    print(f"Monthly Sharpe Ratio: {avg_flat_sharpe:.2f}")
    
    # Plot equity curve
    plt.figure(figsize=(10, 5))
    plt.plot(flat_balances, label='Strategy Equity Curve (Change 1 & 3 ON)', color='#4c6ef5', linewidth=2)
    plt.title('Walk-Forward Strategy Equity Curve (2020-2025)', fontsize=14, fontweight='bold')
    plt.xlabel('Trades Sequence', fontsize=12)
    plt.ylabel('Balance ($)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig('balance_curve_final.png', dpi=150)
    print("Final equity curve saved to balance_curve_final.png")
    
if __name__ == "__main__":
    main()
