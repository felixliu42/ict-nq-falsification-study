import os
import sys
import pandas as pd
import numpy as np
import lightgbm as lgb
from multiprocessing import Pool
from backtest import simulate_trade_execution, get_trade_R, clean_columns
from evaluate_compounding_regimes import calculate_daily_sharpe
from evaluate_regime_balancing import run_walk_forward, evaluate_backtest, YEARS

# Windows to sweep: representing 2, 3, 4, 6, 9, 12 months
WINDOW_SIZES = {
    "2 Months (600)": 600,
    "3 Months (1000)": 1000,
    "4 Months (1200)": 1200,
    "6 Months (1800)": 1800,
    "9 Months (2700)": 2700,
    "12 Months (3500)": 3500
}

def worker_task(args):
    """
    Worker function to run walk forward and evaluate thresholds for a single window size.
    """
    name, size, df_ml_combined, df_raw_combined = args
    base_features = [
        'liquidity_type', 'liquidity_strength', 'sweep_direction', 'sweep_size',
        'bos_strength', 'fvg_rejected', 'retracement_depth', 'time_since_sweep',
        'ny_session', 'london_session', 'asian_session'
    ]
    
    print(f"Starting Walk-Forward for {name}...")
    df_oos = run_walk_forward(df_ml_combined, window_size=size, features=base_features)
    
    # Classify into calendar years
    df_oos['dt'] = pd.to_datetime(df_oos['time'], unit='ms').dt.tz_localize('UTC').dt.tz_convert('America/New_York')
    df_oos['year_num'] = df_oos['dt'].dt.year
    
    thresholds = [0.25, 0.30, 0.35]
    results = []
    
    for th in thresholds:
        flat_ret, comp_ret, flat_dd, comp_dd, flat_sh, comp_sh, n_trades = evaluate_backtest(df_oos, df_raw_combined, th)
        if n_trades > 0:
            results.append({
                'window_name': name,
                'window_size': size,
                'threshold': th,
                'trades': n_trades,
                'flat_ret': flat_ret,
                'comp_ret': comp_ret,
                'flat_dd': flat_dd,
                'comp_dd': comp_dd,
                'flat_sharpe': flat_sh,
                'comp_sharpe': comp_sh,
                'df_oos': df_oos # Store predictions for the best-performing selection
            })
            
    print(f"Finished evaluation for {name}.")
    return results

def main():
    print("Loading and combining datasets for all 6 years...")
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
    
    # Package arguments for parallel execution
    tasks = []
    for name, size in WINDOW_SIZES.items():
        tasks.append((name, size, df_ml_combined, df_raw_combined))
        
    # Run tasks in parallel using multiprocessing Pool
    # We use min(len(tasks), CPU count) to avoid over-subscribing CPU cores
    num_workers = min(len(tasks), os.cpu_count() or 4)
    print(f"Spawning pool with {num_workers} parallel workers...")
    
    with Pool(num_workers) as pool:
        pool_results = pool.map(worker_task, tasks)
        
    # Flatten results
    all_results = []
    for r_list in pool_results:
        all_results.extend(r_list)
        
    # Sort leaderboard by compounded return descending
    leaderboard = sorted(all_results, key=lambda x: x['comp_ret'], reverse=True)
    
    print("\n" + "="*125)
    print("ALL ROLLING WINDOWS LEADERBOARD (Overall continuous metrics 2021-2026)")
    print("="*125)
    print("| Window Size      | Threshold | Trades | Flat Return (%) | Comp Return (%) | Flat Max DD (%) | Comp Max DD (%) | Comp Sharpe |")
    print("|------------------|-----------|--------|-----------------|-----------------|-----------------|-----------------|-------------|")
    for r in leaderboard:
        print(f"| {r['window_name']:<16} | {r['threshold']:.2f}      | {r['trades']:<6} | {r['flat_ret']:>15.1f}% | {r['comp_ret']:>15.1f}% | {r['flat_dd']:>15.1f}% | {r['comp_dd']:>15.1f}% | {r['comp_sharpe']:>11.2f} |")
    print("="*125)
    
    # Select the best performing window
    best = leaderboard[0]
    print(f"\n[Winner] Best Configuration: {best['window_name']} Rolling Window at {best['threshold']:.2f} Threshold.")
    print("Generating full yearly breakdown for the winner...")
    
    df_oos_best = best['df_oos']
    trades_df = df_oos_best[(df_oos_best['pred_prob'] > best['threshold']) & (df_oos_best['ny_session'] == 1.0)].copy()
    
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
            
    print("\n" + "="*115)
    print(f"BEST CONFIGURATION YEARLY BREAKDOWN: {best['window_name']} (Threshold: {best['threshold']:.2f})")
    print("="*115)
    print("| Year   | Trades | Flat Return | Comp Return | Flat Max DD | Comp Max DD | Flat Sharpe | Comp Sharpe |")
    print("|--------|--------|-------------|-------------|-------------|-------------|-------------|-------------|")
    
    flat_balances = [100000.0]
    comp_balances = [100000.0]
    active_years = sorted(list(set([t['year'] for t in trade_outcomes])))
    
    yearly_breakdown_results = []
    
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
            # Flat
            p_flat = 32.0 * t['R'] * t['ret']
            flat_yr_pnls.append(p_flat)
            flat_balances.append(flat_balances[-1] + p_flat)
            flat_yr_balances.append(flat_yr_balances[-1] + p_flat)
            flat_yr_dates.append(t['date'])
            
            # Comp
            current_bal = comp_balances[-1]
            risk_amt = current_bal * 0.02
            n_contracts = risk_amt / (t['R'] * 2.0)
            n_contracts = max(1, int(round(n_contracts)))
            
            p_comp = n_contracts * (t['R'] * 2.0) * t['ret']
            comp_yr_pnls.append(p_comp)
            comp_balances.append(current_bal + p_comp)
            comp_yr_balances.append(comp_yr_balances[-1] + p_comp)
            comp_yr_dates.append(t['date'])
            
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
        
        yearly_breakdown_results.append({
            'trades': n_trades, 'flat_ret': flat_ret, 'comp_ret': comp_ret,
            'flat_dd': flat_max_dd, 'comp_dd': comp_max_dd,
            'flat_sharpe': flat_sharpe, 'comp_sharpe': comp_sharpe
        })
        
    avg_trades = np.mean([r['trades'] for r in yearly_breakdown_results])
    avg_flat_ret = np.mean([r['flat_ret'] for r in yearly_breakdown_results])
    avg_comp_ret = np.mean([r['comp_ret'] for r in yearly_breakdown_results])
    avg_flat_dd = np.mean([r['flat_dd'] for r in yearly_breakdown_results])
    avg_comp_dd = np.mean([r['comp_dd'] for r in yearly_breakdown_results])
    avg_flat_sharpe = np.mean([r['flat_sharpe'] for r in yearly_breakdown_results])
    avg_comp_sharpe = np.mean([r['comp_sharpe'] for r in yearly_breakdown_results])
    
    print("|--------|--------|-------------|-------------|-------------|-------------|-------------|-------------|")
    print(f"| AVERAGE| {avg_trades:<6.1f} | {avg_flat_ret:>10.1f}% | {avg_comp_ret:>10.1f}% | {avg_flat_dd:>10.1f}% | {avg_comp_dd:>10.1f}% | {avg_flat_sharpe:>11.2f} | {avg_comp_sharpe:>11.2f} |")
    print("="*115)

if __name__ == '__main__':
    main()
