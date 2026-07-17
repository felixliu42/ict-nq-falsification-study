import os
import pandas as pd
import numpy as np
from backtest import simulate_trade_execution, get_trade_R, clean_columns
from evaluate_compounding_regimes import calculate_monthly_sharpe
from evaluate_regime_balancing import run_walk_forward

YEARS = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]

def main():
    raw_dfs = []
    ml_dfs = []
    
    for year in YEARS:
        raw_path = f"data/MNQ_{year}/raw_data.csv"
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
    
    base_features = [
        'liquidity_type', 'liquidity_strength', 'sweep_direction', 'sweep_size',
        'bos_strength', 'fvg_rejected', 'retracement_depth', 'time_since_sweep',
        'ny_session', 'london_session', 'asian_session'
    ]
    
    print("Running 6-Month rolling walk-forward validation (W=1800)...")
    df_oos = run_walk_forward(df_ml_combined, window_size=1800, features=base_features)
    
    df_oos['dt'] = pd.to_datetime(df_oos['time'], unit='ms').dt.tz_localize('UTC').dt.tz_convert('America/New_York')
    df_oos['year_num'] = df_oos['dt'].dt.year
    
    # Filter for setups > 0.25 and NY session
    trades_df = df_oos[(df_oos['pred_prob'] > 0.25) & (df_oos['ny_session'] == 1.0)].copy()
    
    # Run backtest with NO CAP
    trade_outcomes = []
    for _, row in trades_df.iterrows():
        R = get_trade_R(row, df_raw_combined)
        if R is not None:
            ret, outcome, duration = simulate_trade_execution(row, df_raw_combined, tp_mode='split')
            if ret is not None:
                dt = pd.to_datetime(row['time'], unit='ms').tz_localize('UTC').tz_convert('America/New_York').date()
                trade_outcomes.append({
                    'ret': ret,
                    'R': R,
                    'date': dt,
                    'year': row['year_num']
                })
                
    print(f"\nSimulated {len(trade_outcomes)} trades (No Cap).")
    
    print("\n" + "="*115)
    print("MNQ 11-YEAR BACKTEST: 6-Month Window, NO RISK CAP (Threshold: 0.25, Split TP)")
    print("="*115)
    print("| Year   | Trades | Flat Return | Comp Return | Flat Max DD | Comp Max DD | Flat Sharpe (M) | Comp Sharpe (M) |")
    print("|--------|--------|-------------|-------------|-------------|-------------|-----------------|-----------------|")
    
    flat_balances = [100000.0]
    comp_balances = [100000.0]
    active_years = sorted(list(set([t['year'] for t in trade_outcomes])))
    
    results = []
    
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
            p_flat = 16.0 * 2.0 * t['R'] * t['ret']
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
        flat_sharpe = calculate_monthly_sharpe(flat_yr_pnls, flat_yr_dates, start_date, end_date, is_compounded=False, initial_balance=flat_bal_start)
        comp_sharpe = calculate_monthly_sharpe(comp_yr_pnls, comp_yr_dates, start_date, end_date, is_compounded=True, initial_balance=comp_bal_start)
        
        print(f"| {yr:<6} | {n_trades:<6} | {flat_ret:>10.1f}% | {comp_ret:>10.1f}% | {flat_max_dd:>10.1f}% | {comp_max_dd:>10.1f}% | {flat_sharpe:>15.2f} | {comp_sharpe:>15.2f} |")
        
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
    
    print("|--------|--------|-------------|-------------|-------------|-------------|-----------------|-----------------|")
    print(f"| AVERAGE| {avg_trades:<6.1f} | {avg_flat_ret:>10.1f}% | {avg_comp_ret:>10.1f}% | {avg_flat_dd:>10.1f}% | {avg_comp_dd:>10.1f}% | {avg_flat_sharpe:>15.2f} | {avg_comp_sharpe:>15.2f} |")
    print("="*115)

if __name__ == '__main__':
    main()
