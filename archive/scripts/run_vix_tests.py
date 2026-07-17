import os
import sys
import pandas as pd
import numpy as np
import lightgbm as lgb
from backtest import simulate_trade_execution, get_trade_R, clean_columns
from evaluate_compounding_regimes import calculate_monthly_sharpe
from evaluate_regime_balancing import run_walk_forward

# Backtest years: 2020 to 2025
TEST_YEARS = [2020, 2021, 2022, 2023, 2024, 2025]

def prepare_vix_data():
    """
    Load all VIX 1m datasets and calculate rolling moving averages and ratios.
    """
    print("Loading and concatenating 1m VIX futures data...")
    vix_dfs = []
    # Load all years from 2018 to 2025 to build a continuous dataset
    for year in [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]:
        vix_dfs.append(pd.read_csv(f"data/VIX_1M_{year}/raw_data.csv"))
        
    vix_df = pd.concat(vix_dfs, ignore_index=True)
    vix_df = clean_columns(vix_df)
    vix_df['time'] = pd.to_numeric(vix_df['time'], errors='coerce')
    vix_df = vix_df.sort_values('time').drop_duplicates(subset=['time']).reset_index(drop=True)
    
    print("Calculating rolling VIX moving averages (this may take a few seconds)...")
    vix_df['ma60'] = vix_df['close'].rolling(60).mean()
    vix_df['ma1440'] = vix_df['close'].rolling(1440).mean()
    vix_df['ma28800'] = vix_df['close'].rolling(28800).mean()
    vix_df['ma362880'] = vix_df['close'].rolling(362880).mean()
    
    print("Calculating ratios...")
    vix_df['vix_raw'] = vix_df['close']
    vix_df['vix_vs_ma60'] = vix_df['close'] / vix_df['ma60']
    vix_df['vix_vs_ma1440'] = vix_df['close'] / vix_df['ma1440']
    vix_df['vix_vs_ma28800'] = vix_df['close'] / vix_df['ma28800']
    vix_df['vix_vs_ma362880'] = vix_df['close'] / vix_df['ma362880']
    
    # Check if there is data
    print(f"VIX dataset prepared: {len(vix_df)} total bars.")
    return vix_df[['time', 'vix_raw', 'vix_vs_ma60', 'vix_vs_ma1440', 'vix_vs_ma28800', 'vix_vs_ma362880']]

def run_test_configuration(test_name, vix_feature, vix_df):
    """
    Run 6-year backtest for a single VIX feature configuration.
    """
    print(f"\n--- RUNNING TEST: {test_name} ({vix_feature}) ---")
    
    raw_dfs = []
    ml_dfs = []
    vix_cols = ['vix_raw', 'vix_vs_ma60', 'vix_vs_ma1440', 'vix_vs_ma28800', 'vix_vs_ma362880']
    
    for year in TEST_YEARS:
        raw_path = f"data/MNQ_{year}/raw_data.csv"
        translated_path = f"data/MNQ_{year}/translated_tv_export.csv"
        
        # Load raw data for backtest execution
        raw_dfs.append(pd.read_csv(raw_path))
        
        # Load NQ translated indicator data
        df_ml_yr = pd.read_csv(translated_path)
        df_ml_yr = clean_columns(df_ml_yr)
        df_ml_yr['time'] = pd.to_numeric(df_ml_yr['time'], errors='coerce')
        
        # Align VIX 1m data on time (exact same bar close)
        df_ml_yr = df_ml_yr.merge(vix_df[['time'] + vix_cols], on='time', how='left')
        df_ml_yr[vix_cols] = df_ml_yr[vix_cols].ffill().bfill()
        
        # Save merged dataframe to temporary file to pass to pipeline
        temp_input = f"data/temp_input_{year}.csv"
        temp_output = f"data/temp_output_{year}.csv"
        df_ml_yr.to_csv(temp_input, index=False)
        
        from pipeline import process_pipeline
        process_pipeline(input_csv=temp_input, output_csv=temp_output)
        
        # Read the labeled ML dataset
        df_labeled_yr = pd.read_csv(temp_output)
        ml_dfs.append(df_labeled_yr)
        
        # Clean up temp files
        if os.path.exists(temp_input): os.remove(temp_input)
        if os.path.exists(temp_output): os.remove(temp_output)
        
    df_raw_combined = pd.concat(raw_dfs, ignore_index=True)
    df_raw_combined = clean_columns(df_raw_combined)
    df_raw_combined['time'] = pd.to_numeric(df_raw_combined['time'], errors='coerce')
    df_raw_combined = df_raw_combined.sort_values('time').drop_duplicates(subset=['time']).reset_index(drop=True)
    
    df_ml_combined = pd.concat(ml_dfs, ignore_index=True)
    df_ml_combined['time'] = pd.to_numeric(df_ml_combined['time'], errors='coerce')
    df_ml_combined = df_ml_combined.sort_values('time').drop_duplicates(subset=['time']).reset_index(drop=True)
    
    # Define features
    features = [
        'liquidity_type', 'liquidity_strength', 'sweep_direction', 'sweep_size',
        'bos_strength', 'fvg_rejected', 'retracement_depth', 'time_since_sweep',
        'ny_session', 'london_session', 'asian_session'
    ]
    if vix_feature:
        features.append(vix_feature)
        
    print(f"Features used: {features}")
    
    # Run 6-Month rolling walk-forward validation (W=1800)
    df_oos = run_walk_forward(df_ml_combined, window_size=1800, features=features)
    df_oos['dt'] = pd.to_datetime(df_oos['time'], unit='ms').dt.tz_localize('UTC').dt.tz_convert('America/New_York')
    df_oos['year_num'] = df_oos['dt'].dt.year
    
    # Filter setups
    trades_df = df_oos[(df_oos['pred_prob'] > 0.25) & (df_oos['ny_session'] == 1.0)].copy()
    
    trade_outcomes = []
    skipped = 0
    for _, row in trades_df.iterrows():
        R = get_trade_R(row, df_raw_combined)
        if R is not None:
            if R > 60.0:
                skipped += 1
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
        
    avg_trades = np.mean([r['trades'] for r in results])
    avg_flat_ret = np.mean([r['flat_ret'] for r in results])
    avg_flat_dd = np.mean([r['flat_dd'] for r in results])
    avg_flat_sharpe = np.mean([r['flat_sharpe'] for r in results])
    
    print(f"Results for {test_name}: Return={avg_flat_ret:.1f}%, Drawdown={avg_flat_dd:.1f}%, Sharpe={avg_flat_sharpe:.2f}")
    return avg_flat_ret, avg_flat_dd, avg_flat_sharpe

def main():
    vix_df = prepare_vix_data()
    
    # Configurations to test
    tests = [
        ("Baseline (No VIX)", None),
        ("Raw 1m VIX close", "vix_raw"),
        ("VIX vs. ma60", "vix_vs_ma60"),
        ("VIX vs. ma1440", "vix_vs_ma1440"),
        ("VIX vs. ma28800", "vix_vs_ma28800"),
        ("VIX vs. ma362880", "vix_vs_ma362880")
    ]
    
    leaderboard = []
    
    import traceback
    for name, feature in tests:
        try:
            ret, dd, sharpe = run_test_configuration(name, feature, vix_df)
            leaderboard.append({
                "Test Name": name,
                "Feature": feature or "None",
                "Average Return": f"{ret:.1f}%",
                "Average Max DD": f"{dd:.1f}%",
                "Monthly Sharpe": f"{sharpe:.2f}"
            })
        except Exception as e:
            traceback.print_exc()
            print(f"Error executing test {name}: {e}")
            
    # Compile leaderboard markdown
    df_leaderboard = pd.DataFrame(leaderboard)
    print("\n" + "="*80)
    print("VIX FEATURE INTEGRATION LEADERBOARD (2020-2025)")
    print("="*80)
    print(df_leaderboard.to_string(index=False))
    print("="*80)
    
    # Save to file
    df_leaderboard.to_csv("results_vix_tests.csv", index=False)
    print("Leaderboard saved to results_vix_tests.csv")

if __name__ == '__main__':
    main()
