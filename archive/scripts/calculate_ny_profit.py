import pandas as pd
import numpy as np
import os
from datetime import time
import lightgbm as lgb
from research_harness import calculate_atr, precompute_metrics
from baseline_strategy import clean_columns, simulate_trade_execution, calculate_metrics
from backtest_variants import run_walk_forward_validation, filter_setups

def simulate_dollar_backtest_ny(df_oos, df_raw, tp_mode, threshold, multiplier):
    """
    Simulate NY-only trades and return dollar-based performance metrics.
    """
    # Exclusively filter for New York session trades
    trades_df = df_oos[(df_oos['pred_prob'] > threshold) & (df_oos['ny_session'] == 1.0)].copy()
    
    dollar_pnls = []
    r_returns = []
    
    for _, row in trades_df.iterrows():
        ret, outcome, duration = simulate_trade_execution(row, df_raw, tp_mode=tp_mode)
        if ret is not None:
            r_returns.append(ret)
            R = row['trade_R']
            if R is not None and not pd.isna(R):
                dollar_pnl = multiplier * R * ret
                dollar_pnls.append(dollar_pnl)
                
    if not dollar_pnls:
        return {
            'trades': 0,
            'win_rate': 0.0,
            'expectancy_r': 0.0,
            'final_balance': 100000.0,
            'net_profit': 0.0,
            'max_drawdown_dollars': 0.0,
            'return_to_drawdown': 0.0
        }
        
    win_rate = len([r for r in r_returns if r > 0]) / len(r_returns)
    expectancy_r = np.mean(r_returns)
    
    net_profit = sum(dollar_pnls)
    final_balance = 100000.0 + net_profit
    
    # Calculate Max Drawdown in dollars
    cum_bal = [100000.0] + list(100000.0 + np.cumsum(dollar_pnls))
    running_max = np.maximum.accumulate(cum_bal)
    drawdowns = running_max - cum_bal
    max_dd_dollars = np.max(drawdowns)
    
    return {
        'trades': len(dollar_pnls),
        'win_rate': win_rate,
        'expectancy_r': expectancy_r,
        'final_balance': final_balance,
        'net_profit': net_profit,
        'max_drawdown_dollars': max_dd_dollars,
        'return_to_drawdown': net_profit / max_dd_dollars if max_dd_dollars > 0 else np.inf
    }

def main():
    df_ml = pd.read_csv('demo_ml_dataset.csv')
    df_raw = clean_columns(pd.read_csv('translated_tv_export.csv'))
    
    df_ml['time'] = pd.to_numeric(df_ml['time'], errors='coerce')
    df_raw['time'] = pd.to_numeric(df_raw['time'], errors='coerce')
    
    print("Precomputing raw index mapping and baseline ATR...")
    atr_series = calculate_atr(df_raw, 14)
    raw_times = df_raw['time'].values
    raw_time_to_idx = {t: idx for idx, t in enumerate(raw_times)}
    
    # Precompute metrics for the entire ML dataset once
    precompute_metrics(df_ml, df_raw, raw_time_to_idx, atr_series)
    
    # Add time helpers for filtering
    df_ml['dt'] = pd.to_datetime(df_ml['time'], unit='ms').dt.tz_localize('UTC').dt.tz_convert('America/New_York')
    df_ml['date'] = df_ml['dt'].dt.date
    df_ml['time_only'] = df_ml['dt'].dt.time
    
    # Identify unique sweeps and setup sequence number
    df_ml['sweep_id'] = df_ml['time'] - df_ml['time_since_sweep'] * 300000
    df_ml = df_ml.sort_values('time').reset_index(drop=True)
    df_ml['setup_num'] = df_ml.groupby('sweep_id').cumcount() + 1
    
    # Run walk-forward validation for all 4 strategies
    print("\nRunning Walk-Forward Validation for Baseline...")
    baseline_oos = run_walk_forward_validation(df_ml)
    
    print("Running Walk-Forward Validation for Variant 1...")
    v1_df = filter_setups(df_ml, sess_cfg='morning_only', enforce_daily_bias=True, bias_reset_mode='htf_sweep', max_setup=3)
    v1_oos = run_walk_forward_validation(v1_df)
    
    print("Running Walk-Forward Validation for Variant 2...")
    v2_df = filter_setups(df_ml, sess_cfg='morning_only', enforce_daily_bias=False, max_setup=3)
    for idx, row in v2_df.iterrows():
        R = row['trade_R']
        if R is not None and not pd.isna(R):
            v2_df.loc[idx, 'suggested_sl'] = row['entry_price'] - row['sweep_direction'] * R * 0.90
    v2_oos = run_walk_forward_validation(v2_df)
    
    print("Running Walk-Forward Validation for Variant 3...")
    v3_df = filter_setups(df_ml, sess_cfg='none', enforce_daily_bias=False, max_setup=1, allowed_liquidity_types='stacked_only')
    v3_oos = run_walk_forward_validation(v3_df)
    
    strategies = [
        ('Baseline (NY Session Only)', baseline_oos),
        ('Variant 1 (Morning NY Only + HTF Sweep Bias)', v1_oos),
        ('Variant 2 (Morning NY Only + Stop Mult 0.90)', v2_oos),
        ('Variant 3 (Stacked Liq NY Only + Max 1 Setup)', v3_oos)
    ]
    
    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    modes = ['tp1', 'split']
    multipliers = [16.0, 32.0]
    
    all_results = []
    
    for mult in multipliers:
        for name, df_oos in strategies:
            if df_oos.empty:
                continue
            for mode in modes:
                for th in thresholds:
                    res = simulate_dollar_backtest_ny(df_oos, df_raw, mode, th, mult)
                    if res['trades'] > 0:
                        all_results.append({
                            'multiplier': mult,
                            'strategy': name,
                            'mode': 'TP1 Only' if mode == 'tp1' else 'Split TP1/TP2',
                            'threshold': th,
                            'trades': res['trades'],
                            'win_rate': res['win_rate'],
                            'expectancy_r': res['expectancy_r'],
                            'net_profit': res['net_profit'],
                            'final_balance': res['final_balance'],
                            'max_dd': res['max_drawdown_dollars'],
                            'profit_to_dd': res['return_to_drawdown']
                        })
                        
    df_results = pd.DataFrame(all_results)
    
    # Sort and display top configurations for Multiplier = 32 (16 contracts)
    print("\n" + "="*145)
    print("TOP NEW YORK ONLY CONFIGURATIONS FOR 16 CONTRACTS ($32/point multiplier)")
    print("="*145)
    df_32 = df_results[df_results['multiplier'] == 32.0].sort_values('final_balance', ascending=False).reset_index(drop=True)
    print(f"| Rank | {'Strategy Name':<45} | {'TP Mode':<13} | {'Thresh':<6} | {'Trades':<6} | {'Win Rate':<8} | {'Exp (R)':<8} | {'Net Profit':<12} | {'Final Balance':<14} | {'Max DD ($)':<12} | {'Profit/DD':<9} |")
    print(f"|------|----------------------------------------------|---------------|--------|--------|----------|---------|--------------|---------------|--------------|-----------|")
    for i, r in df_32.head(20).iterrows():
        print(f"| {i+1:<4} | {r['strategy']:<45} | {r['mode']:<13} | {r['threshold']:<6.2f} | {r['trades']:<6} | {r['win_rate']:<8.1%} | {r['expectancy_r']:<8.2f} | ${r['net_profit']:<11,.2f} | ${r['final_balance']:<12,.2f} | ${r['max_dd']:<11,.2f} | {r['profit_to_dd']:<9.2f} |")
        
    # Sort and display top configurations for Multiplier = 16 ($16/point multiplier)
    print("\n" + "="*145)
    print("TOP NEW YORK ONLY CONFIGURATIONS FOR $16/POINT MULTIPLIER (8 contracts)")
    print("="*145)
    df_16 = df_results[df_results['multiplier'] == 16.0].sort_values('final_balance', ascending=False).reset_index(drop=True)
    print(f"| Rank | {'Strategy Name':<45} | {'TP Mode':<13} | {'Thresh':<6} | {'Trades':<6} | {'Win Rate':<8} | {'Exp (R)':<8} | {'Net Profit':<12} | {'Final Balance':<14} | {'Max DD ($)':<12} | {'Profit/DD':<9} |")
    print(f"|------|----------------------------------------------|---------------|--------|--------|----------|---------|--------------|---------------|--------------|-----------|")
    for i, r in df_16.head(20).iterrows():
        print(f"| {i+1:<4} | {r['strategy']:<45} | {r['mode']:<13} | {r['threshold']:<6.2f} | {r['trades']:<6} | {r['win_rate']:<8.1%} | {r['expectancy_r']:<8.2f} | ${r['net_profit']:<11,.2f} | ${r['final_balance']:<12,.2f} | ${r['max_dd']:<11,.2f} | {r['profit_to_dd']:<9.2f} |")

if __name__ == '__main__':
    main()
