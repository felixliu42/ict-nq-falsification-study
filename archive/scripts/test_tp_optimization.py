import pandas as pd
import numpy as np
import os
from research_harness import calculate_atr, precompute_metrics
from baseline_strategy import clean_columns, calculate_metrics
from backtest_variants import run_walk_forward_validation

def find_nearest_htf_pivot(idx, sweep_dir, entry_price, df_raw, strength=12):
    """
    Find the nearest key structural pivot high/low from the past that lies in the trade direction.
    """
    pivots = []
    # Scan back up to 2000 bars (approx. 1 week of 5-min data)
    for p in range(idx - 1, max(0, idx - 2000), -1):
        if sweep_dir == 1: # Long trade: look for pivot highs above entry
            is_pivot = True
            p_high = float(df_raw.loc[p, 'high'])
            if p_high <= entry_price:
                continue
            # Check left and right strength
            for j in range(1, strength + 1):
                if p - j >= 0 and float(df_raw.loc[p - j, 'high']) >= p_high:
                    is_pivot = False
                    break
                if p + j < len(df_raw) and float(df_raw.loc[p + j, 'high']) >= p_high:
                    is_pivot = False
                    break
            if is_pivot:
                pivots.append(p_high)
                if len(pivots) >= 3:
                    break
        else: # Short trade: look for pivot lows below entry
            is_pivot = True
            p_low = float(df_raw.loc[p, 'low'])
            if p_low >= entry_price:
                continue
            for j in range(1, strength + 1):
                if p - j >= 0 and float(df_raw.loc[p - j, 'low']) <= p_low:
                    is_pivot = False
                    break
                if p + j < len(df_raw) and float(df_raw.loc[p + j, 'low']) <= p_low:
                    is_pivot = False
                    break
            if is_pivot:
                pivots.append(p_low)
                if len(pivots) >= 3:
                    break
                    
    if pivots:
        # Return the one closest to the entry price
        pivots.sort(key=lambda val: abs(val - entry_price))
        return pivots[0]
    return None

def simulate_trade_custom_tp(row, df_raw, tp1_mult, tp2_type, tp2_mult_or_strength, multiplier=32.0):
    entry_time = int(row['time'])
    sweep_dir = int(row['sweep_direction'])
    time_since = int(row['time_since_sweep'])
    
    # Find entry bar index
    raw_indices = df_raw.index[df_raw['time'] == entry_time].tolist()
    if not raw_indices:
        return None
    idx = raw_indices[0]
    
    entry_price = float(df_raw.loc[idx, 'close'])
    sweep_idx = idx - time_since
    if sweep_idx < 0:
        return None
        
    # Initial stop loss (sweep extreme)
    if sweep_dir == -1: # Bearish Setup
        initial_sl = float(df_raw.loc[sweep_idx, 'high'])
    else: # Bullish Setup
        initial_sl = float(df_raw.loc[sweep_idx, 'low'])
        
    R = abs(entry_price - initial_sl)
    if R == 0.0:
        return None
        
    # Calculate TP1
    tp1_dist = tp1_mult * R
    tp1 = entry_price + tp1_dist * sweep_dir
    
    # Calculate TP2
    if tp2_type == 'mult':
        tp2_dist = tp1_dist * tp2_mult_or_strength
        tp2 = entry_price + tp2_dist * sweep_dir
    elif tp2_type == 'htf_pivot':
        strength = tp2_mult_or_strength
        pivot_val = find_nearest_htf_pivot(idx, sweep_dir, entry_price, df_raw, strength=strength)
        if pivot_val is not None:
            # Enforce a minimum TP2 of 1.2 * TP1 distance, and cap at 8 * TP1 to prevent unrealistic targets
            proposed_tp2_dist = abs(pivot_val - entry_price)
            if proposed_tp2_dist < 1.2 * tp1_dist:
                tp2 = entry_price + 1.5 * tp1_dist * sweep_dir
            elif proposed_tp2_dist > 8.0 * tp1_dist:
                tp2 = entry_price + 4.0 * tp1_dist * sweep_dir
            else:
                tp2 = pivot_val
        else:
            # Fallback to 2.0x TP1 multiplier
            tp2 = entry_price + 2.0 * tp1_dist * sweep_dir
            
    # Run simulation bar-by-bar
    has_hit_tp1 = False
    for t in range(idx + 1, len(df_raw)):
        high = float(df_raw.loc[t, 'high'])
        low = float(df_raw.loc[t, 'low'])
        
        # Check stop loss
        if not has_hit_tp1:
            if sweep_dir == 1 and low <= initial_sl:
                return -1.0 # Stopped out fully
            elif sweep_dir == -1 and high >= initial_sl:
                return -1.0
                
            # Check TP1
            if sweep_dir == 1 and high >= tp1:
                has_hit_tp1 = True
            elif sweep_dir == -1 and low <= tp1:
                has_hit_tp1 = True
        else:
            # Once TP1 is hit, stop moves to breakeven (entry_price)
            if sweep_dir == 1 and low <= entry_price:
                # First half won TP1 (tp1_mult R), second half got stopped at BE (0R)
                return 0.5 * tp1_mult
            elif sweep_dir == -1 and high >= entry_price:
                return 0.5 * tp1_mult
                
            # Check TP2
            tp2_r = abs(tp2 - entry_price) / R
            if sweep_dir == 1 and high >= tp2:
                # First half won TP1, second half won TP2
                return 0.5 * tp1_mult + 0.5 * tp2_r
            elif sweep_dir == -1 and low <= tp2:
                return 0.5 * tp1_mult + 0.5 * tp2_r
                
    # End of data: close position
    final_close = float(df_raw.iloc[-1]['close'])
    final_r = ((final_close - entry_price) / R) * sweep_dir
    if has_hit_tp1:
        return 0.5 * tp1_mult + 0.5 * max(0.0, final_r)
    else:
        return max(-1.0, final_r)

def run_tp_backtest(df_oos, df_raw, threshold, tp1_mult, tp2_type, tp2_param, multiplier=32.0):
    trades_df = df_oos[(df_oos['pred_prob'] > threshold) & (df_oos['ny_session'] == 1.0)].copy()
    
    r_returns = []
    dollar_pnls = []
    for _, row in trades_df.iterrows():
        ret = simulate_trade_custom_tp(row, df_raw, tp1_mult, tp2_type, tp2_param)
        if ret is not None:
            r_returns.append(ret)
            R = row['trade_R']
            if R is not None and not pd.isna(R):
                dollar_pnls.append(multiplier * R * ret)
                
    if not r_returns:
        return 0, 0.0, 0.0, 0.0, 0.0
        
    metrics = calculate_metrics(r_returns, [0]*len(r_returns))
    net_profit = sum(dollar_pnls)
    cum = [100000.0] + list(100000.0 + np.cumsum(dollar_pnls))
    max_dd = np.max(np.maximum.accumulate(cum) - cum)
    
    return len(trades_df), metrics['win_rate'], metrics['expectancy'], net_profit, max_dd

def main():
    df_ml = pd.read_csv('demo_ml_dataset.csv')
    df_raw = clean_columns(pd.read_csv('translated_tv_export.csv'))
    
    df_ml['time'] = pd.to_numeric(df_ml['time'], errors='coerce')
    df_raw['time'] = pd.to_numeric(df_raw['time'], errors='coerce')
    
    print("Precomputing raw indexes...")
    atr_series = calculate_atr(df_raw, 14)
    raw_times = df_raw['time'].values
    raw_time_to_idx = {t: idx for idx, t in enumerate(raw_times)}
    precompute_metrics(df_ml, df_raw, raw_time_to_idx, atr_series)
    
    df_ml['dt'] = pd.to_datetime(df_ml['time'], unit='ms').dt.tz_localize('UTC').dt.tz_convert('America/New_York')
    df_ml['time_only'] = df_ml['dt'].dt.time
    df_ml['sweep_id'] = df_ml['time'] - df_ml['time_since_sweep'] * 300000
    df_ml = df_ml.sort_values('time').reset_index(drop=True)
    df_ml['setup_num'] = df_ml.groupby('sweep_id').cumcount() + 1
    
    print("Running Walk-Forward validation...")
    baseline_oos = run_walk_forward_validation(df_ml)
    
    # Grid search configs
    tp1_mults = [1.5, 2.0, 2.5]
    tp2_configs = [
        ('mult', 1.5, '1.5x TP1'),
        ('mult', 2.0, '2.0x TP1'),
        ('mult', 3.0, '3.0x TP1'),
        ('mult', 4.0, '4.0x TP1'),
        ('htf_pivot', 12, '1-Hour Pivots (Strength 12)'),
        ('htf_pivot', 24, '2-Hour Pivots (Strength 24)')
    ]
    
    threshold = 0.60 # top threshold
    
    print("\n" + "="*115)
    print("TARGET OPTIMIZATION RESULTS (NY Session Only, ML Threshold 0.60, 16 Contracts)")
    print("="*115)
    print(f"| TP1 Target | TP2 Target Config | Trades | Win Rate | Expectancy (R) | Net Profit | Max Drawdown | Profit/DD Ratio |")
    print(f"|------------|-------------------|--------|----------|----------------|------------|--------------|-----------------|")
    
    results = []
    for tp1 in tp1_mults:
        for t2_type, t2_param, t2_label in tp2_configs:
            n_trades, win_rate, exp, pnl, dd = run_tp_backtest(baseline_oos, df_raw, threshold, tp1, t2_type, t2_param)
            ratio = pnl / dd if dd > 0 else 0.0
            results.append((tp1, t2_label, n_trades, win_rate, exp, pnl, dd, ratio))
            
    # Sort by profit
    results.sort(key=lambda x: x[5], reverse=True)
    
    for r in results:
        print(f"| {r[0]:<10.1f}R | {r[1]:<17} | {r[2]:<6} | {r[3]:<8.1%} | {r[4]:<14.2f} | ${r[5]:<10,.2f} | ${r[6]:<12,.2f} | {r[7]:<15.2f} |")
        
    print("="*115)

if __name__ == '__main__':
    main()
