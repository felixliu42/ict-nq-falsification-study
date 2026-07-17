import os
import pandas as pd
import numpy as np
from backtest import simulate_trade_execution, get_trade_R, clean_columns
from evaluate_compounding_regimes import calculate_daily_sharpe
from evaluate_regime_balancing import run_walk_forward, YEARS

def evaluate_backtest_with_cap(df_oos, df_raw_combined, threshold, max_risk_points):
    """
    Simulate trade outcomes and filter out trades where R > max_risk_points.
    """
    trades_df = df_oos[(df_oos['pred_prob'] > threshold) & (df_oos['ny_session'] == 1.0)].copy()
    
    trade_outcomes = []
    skipped_large_risk = 0
    
    for _, row in trades_df.iterrows():
        R = get_trade_R(row, df_raw_combined)
        if R is not None:
            # Filter by risk size
            if R > max_risk_points:
                skipped_large_risk += 1
                continue
                
            ret, outcome, duration = simulate_trade_execution(row, df_raw_combined, tp_mode='split')
            if ret is not None:
                dt = pd.to_datetime(row['time'], unit='ms').tz_localize('UTC').tz_convert('America/New_York').date()
                trade_outcomes.append({
                    'ret': ret,
                    'R': R,
                    'date': dt
                })
                
    if len(trade_outcomes) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, skipped_large_risk
        
    flat_balances = [100000.0]
    comp_balances = [100000.0]
    
    flat_pnls = []
    comp_pnls = []
    flat_dates = []
    comp_dates = []
    
    for t in trade_outcomes:
        # Flat
        p_flat = 32.0 * t['R'] * t['ret']
        flat_pnls.append(p_flat)
        flat_balances.append(flat_balances[-1] + p_flat)
        flat_dates.append(t['date'])
        
        # Comp
        current_bal = comp_balances[-1]
        risk_amt = current_bal * 0.02
        n_contracts = risk_amt / (t['R'] * 2.0)
        n_contracts = max(1, int(round(n_contracts)))
        
        p_comp = n_contracts * (t['R'] * 2.0) * t['ret']
        comp_pnls.append(p_comp)
        comp_balances.append(current_bal + p_comp)
        comp_dates.append(t['date'])
        
    flat_ret = (flat_balances[-1] - 100000.0) / 100000.0 * 100
    comp_ret = (comp_balances[-1] - 100000.0) / 100000.0 * 100
    
    flat_cum_max = np.maximum.accumulate(flat_balances)
    flat_dd = np.max((flat_cum_max - flat_balances) / flat_cum_max * 100)
    
    comp_cum_max = np.maximum.accumulate(comp_balances)
    comp_dd = np.max((comp_cum_max - comp_balances) / comp_cum_max * 100)
    
    flat_sharpe = calculate_daily_sharpe(flat_pnls, flat_dates, "2021-01-01", "2025-12-31", is_compounded=False, initial_balance=100000.0)
    comp_sharpe = calculate_daily_sharpe(comp_pnls, comp_dates, "2021-01-01", "2025-12-31", is_compounded=True, initial_balance=100000.0)
    
    return flat_ret, comp_ret, flat_dd, comp_dd, flat_sharpe, comp_sharpe, len(trade_outcomes), skipped_large_risk

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
    
    base_features = [
        'liquidity_type', 'liquidity_strength', 'sweep_direction', 'sweep_size',
        'bos_strength', 'fvg_rejected', 'retracement_depth', 'time_since_sweep',
        'ny_session', 'london_session', 'asian_session'
    ]
    
    print("Running 6-Month rolling walk-forward validation (W=1800)...")
    df_oos = run_walk_forward(df_ml_combined, window_size=1800, features=base_features)
    
    # Sweep different risk caps in MNQ points (e.g. 40, 50, 60, 80 points, and no cap)
    risk_caps = [40.0, 50.0, 60.0, 80.0, 9999.0]
    
    print("\n" + "="*125)
    print("RISK CAPPING RESULTS (6-Month Rolling Window, 0.25 Threshold, Split TP1/TP2)")
    print("="*125)
    print("| Max Risk (pts) | Trades | Skipped | Flat Return (%) | Comp Return (%) | Flat Max DD (%) | Comp Max DD (%) | Comp Sharpe |")
    print("|----------------|--------|---------|-----------------|-----------------|-----------------|-----------------|-------------|")
    
    for cap in risk_caps:
        cap_str = f"{cap} pts" if cap < 9999.0 else "No Cap"
        f_ret, c_ret, f_dd, c_dd, f_sh, c_sh, trs, skipped = evaluate_backtest_with_cap(df_oos, df_raw_combined, 0.25, cap)
        print(f"| {cap_str:<14} | {trs:<6} | {skipped:<7} | {f_ret:>15.1f}% | {c_ret:>15.1f}% | {f_dd:>15.1f}% | {c_dd:>15.1f}% | {c_sh:>11.2f} |")
        
    print("="*125)

if __name__ == '__main__':
    main()
