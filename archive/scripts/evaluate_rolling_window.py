import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from backtest import simulate_trade_execution, get_trade_R, clean_columns
from evaluate_compounding_regimes import calculate_daily_sharpe

YEARS = [2020, 2021, 2022, 2023, 2024, 2025]

def run_rolling_window_validation(df_ml, window_size=3500, step_size=5):
    """
    Run a rolling window walk-forward validation.
    The model trains on the most recent 'window_size' setups and predicts the next 'step_size' setups.
    """
    print(f"Running Rolling Window Validation (Window={window_size}, Step={step_size})...")
    features = [
        'liquidity_type', 'liquidity_strength', 'sweep_direction', 'sweep_size',
        'bos_strength', 'fvg_rejected', 'retracement_depth', 'time_since_sweep',
        'ny_session', 'london_session', 'asian_session'
    ]
    
    df_ml = df_ml.sort_values('time').reset_index(drop=True)
    df_ml['pred_prob'] = np.nan
    
    n_rows = len(df_ml)
    if n_rows <= window_size:
        raise ValueError(f"Dataset has {n_rows} rows, which is <= window_size ({window_size}).")
        
    for t in range(window_size, n_rows, step_size):
        # Training set is a rolling window of size 'window_size' right before index t
        train_df = df_ml.iloc[t - window_size : t]
        test_end = min(t + step_size, n_rows)
        test_df = df_ml.iloc[t:test_end]
        
        model = lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=4,
            num_leaves=15,
            random_state=42,
            min_child_samples=3,
            verbosity=-1
        )
        
        X_train = train_df[features]
        y_train = train_df['label']
        X_test = test_df[features]
        
        model.fit(X_train, y_train)
        
        probs = model.predict_proba(X_test)[:, 1]
        df_ml.loc[t:test_end-1, 'pred_prob'] = probs
        
    # Drop rows that fell inside the initial training window (they have no OOS prediction)
    df_oos = df_ml.iloc[window_size:].copy().reset_index(drop=True)
    print(f"Generated {len(df_oos)} out-of-sample predictions across years.")
    return df_oos

def main():
    print("Combining datasets for all 6 years...")
    raw_dfs = []
    ml_dfs = []
    
    for year in YEARS:
        raw_path = f"data/MNQ_{year}/translated_tv_export.csv"
        ml_path = f"data/MNQ_{year}/demo_ml_dataset.csv"
        
        if not os.path.exists(raw_path) or not os.path.exists(ml_path):
            print(f"Error: Missing data files for year {year}. Run evaluate_regimes.py first.")
            sys.exit(1)
            
        raw_dfs.append(pd.read_csv(raw_path))
        ml_dfs.append(pd.read_csv(ml_path))
        
    # Combine and sort chronologically
    df_raw_combined = pd.concat(raw_dfs, ignore_index=True)
    df_ml_combined = pd.concat(ml_dfs, ignore_index=True)
    
    # Sort and drop duplicates (e.g. at year boundaries)
    df_raw_combined = clean_columns(df_raw_combined)
    df_raw_combined['time'] = pd.to_numeric(df_raw_combined['time'], errors='coerce')
    df_raw_combined = df_raw_combined.sort_values('time').drop_duplicates(subset=['time']).reset_index(drop=True)
    
    df_ml_combined['time'] = pd.to_numeric(df_ml_combined['time'], errors='coerce')
    df_ml_combined = df_ml_combined.sort_values('time').drop_duplicates(subset=['time']).reset_index(drop=True)
    
    # Run rolling window validation
    # Window size: 300 setups (~1 month of data)
    df_oos = run_rolling_window_validation(df_ml_combined, window_size=300, step_size=5)
    
    # Convert timestamps to datetime to classify into calendar years
    df_oos['dt'] = pd.to_datetime(df_oos['time'], unit='ms').dt.tz_localize('UTC').dt.tz_convert('America/New_York')
    df_oos['year_num'] = df_oos['dt'].dt.year
    
    # We sweep multiple calibrated thresholds since the larger training set size regularizes predictions
    thresholds = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55]
    
    for th in thresholds:
        print("\n" + "="*115)
        print(f"CONTINUOUS ROLLING WINDOW PERFORMANCE FOR THRESHOLD: {th:.2f} (NY Session, Split TP1/TP2)")
        print("="*115)
        print("| Year   | Trades | Flat Return | Comp Return | Flat Max DD | Comp Max DD | Flat Sharpe | Comp Sharpe |")
        print("|--------|--------|-------------|-------------|-------------|-------------|-------------|-------------|")
        
        # Filter for setups > threshold and NY session
        trades_df = df_oos[(df_oos['pred_prob'] > th) & (df_oos['ny_session'] == 1.0)].copy()
        
        trade_outcomes = []
        for _, row in trades_df.iterrows():
            ret, outcome, duration = simulate_trade_execution(row, df_raw_combined, tp_mode='split')
            R = get_trade_R(row, df_raw_combined)
            if ret is not None and R is not None:
                dt = pd.to_datetime(row['time'], unit='ms').tz_localize('UTC').tz_convert('America/New_York').date()
                trade_outcomes.append({
                    'ret': ret,
                    'R': R,
                    'date': dt,
                    'year': row['year_num']
                })
                
        if len(trade_outcomes) == 0:
            print(f"No trades for threshold {th:.2f}")
            continue
            
        flat_balances = [100000.0]
        comp_balances = [100000.0]
        active_years = sorted(list(set([t['year'] for t in trade_outcomes])))
        
        results = []
        flat_all_pnls = []
        flat_all_dates = []
        comp_all_pnls = []
        comp_all_dates = []
        
        for yr in active_years:
            yr_trades = [t for t in trade_outcomes if t['year'] == yr]
            n_trades = len(yr_trades)
            
            if n_trades == 0:
                continue
                
            flat_yr_pnls = []
            comp_yr_pnls = []
            flat_yr_dates = []
            comp_yr_dates = []
            
            flat_bal_start = flat_balances[-1]
            comp_bal_start = comp_balances[-1]
            
            flat_yr_balances = [flat_bal_start]
            comp_yr_balances = [comp_bal_start]
            
            for t in yr_trades:
                # 1. Flat
                p_flat = 32.0 * t['R'] * t['ret']
                flat_yr_pnls.append(p_flat)
                flat_all_pnls.append(p_flat)
                flat_balances.append(flat_balances[-1] + p_flat)
                flat_yr_balances.append(flat_yr_balances[-1] + p_flat)
                flat_yr_dates.append(t['date'])
                flat_all_dates.append(t['date'])
                
                # 2. Comp
                current_bal = comp_balances[-1]
                risk_amt = current_bal * 0.02
                n_contracts = risk_amt / (t['R'] * 2.0)
                n_contracts = max(1, int(round(n_contracts)))
                
                p_comp = n_contracts * (t['R'] * 2.0) * t['ret']
                comp_yr_pnls.append(p_comp)
                comp_all_pnls.append(p_comp)
                comp_balances.append(current_bal + p_comp)
                comp_yr_balances.append(comp_yr_balances[-1] + p_comp)
                comp_yr_dates.append(t['date'])
                comp_all_dates.append(t['date'])
                
            # Returns
            flat_ret = (flat_yr_balances[-1] - flat_bal_start) / flat_bal_start * 100
            comp_ret = (comp_yr_balances[-1] - comp_bal_start) / comp_bal_start * 100
            
            # Max Drawdown
            flat_running_max = np.maximum.accumulate(flat_yr_balances)
            flat_dds = (flat_running_max - flat_yr_balances) / flat_running_max * 100
            flat_max_dd = np.max(flat_dds)
            
            comp_running_max = np.maximum.accumulate(comp_yr_balances)
            comp_dds = (comp_running_max - comp_yr_balances) / comp_running_max * 100
            comp_max_dd = np.max(comp_dds)
            
            # Sharpe
            start_date = f"{yr}-01-01"
            end_date = f"{yr}-12-31"
            flat_sharpe = calculate_daily_sharpe(flat_yr_pnls, flat_yr_dates, start_date, end_date, is_compounded=False, initial_balance=flat_bal_start)
            comp_sharpe = calculate_daily_sharpe(comp_yr_pnls, comp_yr_dates, start_date, end_date, is_compounded=True, initial_balance=comp_bal_start)
            
            print(f"| {yr:<6} | {n_trades:<6} | {flat_ret:>10.1f}% | {comp_ret:>10.1f}% | {flat_max_dd:>10.1f}% | {comp_max_dd:>10.1f}% | {flat_sharpe:>11.2f} | {comp_sharpe:>11.2f} |")
            
            results.append({
                'trades': n_trades, 'flat_ret': flat_ret, 'comp_ret': comp_ret,
                'flat_dd': flat_max_dd, 'comp_dd': comp_max_dd,
                'flat_sharpe': flat_sharpe, 'comp_sharpe': comp_sharpe
            })
            
        avg_trades = np.mean([r['trades'] for r in results])
        avg_flat_ret = np.mean([r['flat_ret'] for r in results])
        avg_comp_ret = np.mean([r['comp_ret'] for r in results])
        avg_flat_dd = np.mean([r['flat_dd'] for r in results])
        avg_comp_dd = np.mean([r['comp_dd'] for r in results])
        avg_flat_sharpe = np.mean([r['flat_sharpe'] for r in results])
        avg_comp_sharpe = np.mean([r['comp_sharpe'] for r in results])
        
        print("|--------|--------|-------------|-------------|-------------|-------------|-------------|-------------|")
        print(f"| AVERAGE| {avg_trades:<6.1f} | {avg_flat_ret:>10.1f}% | {avg_comp_ret:>10.1f}% | {avg_flat_dd:>10.1f}% | {avg_comp_dd:>10.1f}% | {avg_flat_sharpe:>11.2f} | {avg_comp_sharpe:>11.2f} |")
        print("="*115)
        
        # Calculate overall continuous metrics
        flat_total_ret = (flat_balances[-1] - 100000.0) / 100000.0 * 100
        comp_total_ret = (comp_balances[-1] - 100000.0) / 100000.0 * 100
        
        flat_cum_max = np.maximum.accumulate(flat_balances)
        flat_total_dd = np.max((flat_cum_max - flat_balances) / flat_cum_max * 100)
        
        comp_cum_max = np.maximum.accumulate(comp_balances)
        comp_total_dd = np.max((comp_cum_max - comp_balances) / comp_cum_max * 100)
        
        # Continuous Sharpe ratio across the entire 5 years
        flat_total_sharpe = calculate_daily_sharpe(flat_all_pnls, flat_all_dates, "2021-01-01", "2025-12-31", is_compounded=False, initial_balance=100000.0)
        comp_total_sharpe = calculate_daily_sharpe(comp_all_pnls, comp_all_dates, "2021-01-01", "2025-12-31", is_compounded=True, initial_balance=100000.0)
        
        print(f"OVERALL CONTINUOUS RESULTS (2021-2026):")
        print(f"  Flat Sizing: Total Return = {flat_total_ret:>5.1f}% | Max Drawdown = {flat_total_dd:>4.1f}% | Sharpe = {flat_total_sharpe:>5.2f}")
        print(f"  Comp Sizing: Total Return = {comp_total_ret:>5.1f}% | Max Drawdown = {comp_total_dd:>4.1f}% | Sharpe = {comp_total_sharpe:>5.2f}")
        print("="*115)

if __name__ == '__main__':
    main()
