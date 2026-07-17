import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from backtest import simulate_trade_execution, get_trade_R, clean_columns
from evaluate_compounding_regimes import calculate_daily_sharpe
from evaluate_regime_balancing import run_walk_forward, evaluate_backtest, calculate_regime_features, YEARS

def main():
    raw_dfs = []
    ml_dfs = []
    
    for year in YEARS:
        raw_path = f"data/MNQ_{year}/translated_tv_export.csv"
        ml_path = f"data/MNQ_{year}/demo_ml_dataset.csv"
        raw_dfs.append(pd.read_csv(raw_path))
        ml_dfs.append(pd.read_csv(ml_path))
        
    df_raw_combined = pd.concat(raw_dfs, ignore_index=True)
    df_ml_combined = pd.concat(ml_dfs, ignore_index=True)
    
    df_raw_combined = clean_columns(df_raw_combined)
    df_raw_combined['time'] = pd.to_numeric(df_raw_combined['time'], errors='coerce')
    df_raw_combined = df_raw_combined.sort_values('time').drop_duplicates(subset=['time']).reset_index(drop=True)
    
    df_ml_combined['time'] = pd.to_numeric(df_ml_combined['time'], errors='coerce')
    df_ml_combined = df_ml_combined.sort_values('time').drop_duplicates(subset=['time']).reset_index(drop=True)
    
    # Compute regime features
    df_raw_combined = calculate_regime_features(df_raw_combined)
    df_ml_combined = pd.merge_asof(
        df_ml_combined,
        df_raw_combined[['time', 'vol_ratio', 'ma_pos']],
        on='time',
        direction='backward'
    )
    
    # 6-Month rolling window (W=1800 setups) WITH regime features
    features = [
        'liquidity_type', 'liquidity_strength', 'sweep_direction', 'sweep_size',
        'bos_strength', 'fvg_rejected', 'retracement_depth', 'time_since_sweep',
        'ny_session', 'london_session', 'asian_session',
        'vol_ratio', 'ma_pos'
    ]
    
    print("\nTraining 6-Month Rolling Window + Explicit Regime Features...")
    df_oos = run_walk_forward(df_ml_combined, window_size=1800, features=features)
    
    df_oos['dt'] = pd.to_datetime(df_oos['time'], unit='ms').dt.tz_localize('UTC').dt.tz_convert('America/New_York')
    df_oos['year_num'] = df_oos['dt'].dt.year
    
    thresholds = [0.25, 0.30, 0.35, 0.40]
    
    print("\n" + "="*125)
    print("6-MONTH ROLLING WINDOW WITH REGIME FEATURES LEADERBOARD (Overall continuous metrics 2021-2026)")
    print("="*125)
    print("| Sizing Model      | Threshold | Trades | Total Return (%) | Max Drawdown (%) | Daily Sharpe |")
    print("|-------------------|-----------|--------|------------------|------------------|--------------|")
    for th in thresholds:
        f_ret, c_ret, f_dd, c_dd, f_sh, c_sh, trs = evaluate_backtest(df_oos, df_raw_combined, th)
        if trs > 0:
            print(f"| Flat Sizing       | {th:.2f}      | {trs:<6} | {f_ret:>16.1f}% | {f_dd:>16.1f}% | {f_sh:>12.2f} |")
            print(f"| Compounding (2.0%)| {th:.2f}      | {trs:<6} | {c_ret:>16.1f}% | {c_dd:>16.1f}% | {c_sh:>12.2f} |")
            print("|-------------------|-----------|--------|------------------|------------------|--------------|")
    print("="*125)

if __name__ == '__main__':
    main()
