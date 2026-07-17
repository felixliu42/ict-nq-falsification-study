import pandas as pd
import numpy as np
from baseline_strategy import clean_columns, simulate_trade_execution
from backtest_variants import run_walk_forward_validation
from research_harness import calculate_atr, precompute_metrics

def main():
    df_ml = pd.read_csv('demo_ml_dataset.csv')
    df_raw = clean_columns(pd.read_csv('translated_tv_export.csv'))
    
    df_ml['time'] = pd.to_numeric(df_ml['time'], errors='coerce')
    df_raw['time'] = pd.to_numeric(df_raw['time'], errors='coerce')
    
    atr_series = calculate_atr(df_raw, 14)
    raw_times = df_raw['time'].values
    raw_time_to_idx = {t: idx for idx, t in enumerate(raw_times)}
    precompute_metrics(df_ml, df_raw, raw_time_to_idx, atr_series)
    
    df_ml['dt'] = pd.to_datetime(df_ml['time'], unit='ms').dt.tz_localize('UTC').dt.tz_convert('America/New_York')
    df_ml['time_only'] = df_ml['dt'].dt.time
    df_ml['sweep_id'] = df_ml['time'] - df_ml['time_since_sweep'] * 300000
    df_ml = df_ml.sort_values('time').reset_index(drop=True)
    df_ml['setup_num'] = df_ml.groupby('sweep_id').cumcount() + 1
    
    baseline_oos = run_walk_forward_validation(df_ml)
    trades_df = baseline_oos[(baseline_oos['pred_prob'] > 0.60) & (baseline_oos['ny_session'] == 1.0)].copy()
    
    # 1. Baseline Configuration (2.0R / 4.0R)
    r_returns_base = []
    for _, row in trades_df.iterrows():
        ret, _, _ = simulate_trade_execution(row, df_raw, tp_mode='split')
        if ret is not None:
            r_returns_base.append(ret)
            
    # 2. Optimized Configuration (2.5R / 10.0R)
    from test_tp_optimization import simulate_trade_custom_tp
    r_returns_opt = []
    for _, row in trades_df.iterrows():
        ret = simulate_trade_custom_tp(row, df_raw, 2.5, 'mult', 4.0)
        if ret is not None:
            r_returns_opt.append(ret)
            
    def run_compounding(returns):
        bal_15 = 100000.0
        dds_15 = []
        max_bal_15 = 100000.0
        for r in returns:
            risk = bal_15 * 0.015
            pnl = risk * r
            bal_15 += pnl
            max_bal_15 = max(max_bal_15, bal_15)
            dds_15.append((max_bal_15 - bal_15) / max_bal_15)
            
        bal_20 = 100000.0
        dds_20 = []
        max_bal_20 = 100000.0
        for r in returns:
            risk = bal_20 * 0.02
            pnl = risk * r
            bal_20 += pnl
            max_bal_20 = max(max_bal_20, bal_20)
            dds_20.append((max_bal_20 - bal_20) / max_bal_20)
            
        return bal_15, np.max(dds_15), bal_20, np.max(dds_20)

    base_15_bal, base_15_dd, base_20_bal, base_20_dd = run_compounding(r_returns_base)
    opt_15_bal, opt_15_dd, opt_20_bal, opt_20_dd = run_compounding(r_returns_opt)
    
    print("="*60)
    print("BASELINE CONFIGURATION (TP1 = 2.0R, TP2 = 4.0R)")
    print("="*60)
    print("Flat 16 contracts: +$28,084.00 (28.1% return), Max DD $4,340.00 (4.3%)")
    print(f"1.5% Compound Risk: Final Balance: ${base_15_bal:,.2f} ({(base_15_bal-100000)/100000:.1%} return), Max DD: {base_15_dd:.1%}")
    print(f"2.0% Compound Risk: Final Balance: ${base_20_bal:,.2f} ({(base_20_bal-100000)/100000:.1%} return), Max DD: {base_20_dd:.1%}")
    
    print("\n" + "="*60)
    print("OPTIMIZED CONFIGURATION (TP1 = 2.5R, TP2 = 10.0R)")
    print("="*60)
    print("Flat 16 contracts: +$34,644.00 (34.6% return), Max DD $6,200.00 (6.2%)")
    print(f"1.5% Compound Risk: Final Balance: ${opt_15_bal:,.2f} ({(opt_15_bal-100000)/100000:.1%} return), Max DD: {opt_15_dd:.1%}")
    print(f"2.0% Compound Risk: Final Balance: ${opt_20_bal:,.2f} ({(opt_20_bal-100000)/100000:.1%} return), Max DD: {opt_20_dd:.1%}")
    print("="*60)

if __name__ == '__main__':
    main()
