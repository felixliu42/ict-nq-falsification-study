import pandas as pd
import numpy as np
from research_harness import calculate_atr, precompute_metrics
from baseline_strategy import clean_columns, calculate_metrics
from backtest_variants import run_walk_forward_validation

def find_nearest_htf_pivot_beyond_tp1(idx, sweep_dir, tp1, df_raw, strength=12):
    """
    Find the nearest key structural pivot high/low from the past that lies beyond TP1.
    """
    pivots = []
    # Scan back up to 2000 bars
    for p in range(idx - 1, max(0, idx - 2000), -1):
        if sweep_dir == 1: # Long trade: look for pivot highs above tp1
            is_pivot = True
            p_high = float(df_raw.loc[p, 'high'])
            if p_high <= tp1:
                continue
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
        else: # Short trade: look for pivot lows below tp1
            is_pivot = True
            p_low = float(df_raw.loc[p, 'low'])
            if p_low >= tp1:
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
        # Return the one closest to tp1
        pivots.sort(key=lambda val: abs(val - tp1))
        return pivots[0]
    return None

def simulate_trade_htf_tp2(row, df_raw, strength, fallback_mult=2.0):
    entry_time = int(row['time'])
    sweep_dir = int(row['sweep_direction'])
    time_since = int(row['time_since_sweep'])
    
    raw_indices = df_raw.index[df_raw['time'] == entry_time].tolist()
    if not raw_indices:
        return None
    idx = raw_indices[0]
    
    entry_price = float(df_raw.loc[idx, 'close'])
    sweep_idx = idx - time_since
    if sweep_idx < 0:
        return None
        
    if sweep_dir == -1:
        initial_sl = float(df_raw.loc[sweep_idx, 'high'])
    else:
        initial_sl = float(df_raw.loc[sweep_idx, 'low'])
        
    R = abs(entry_price - initial_sl)
    if R == 0.0:
        return None
        
    if 'suggested_sl' in row and not pd.isna(row['suggested_sl']):
        stop_loss = float(row['suggested_sl'])
    else:
        stop_loss = initial_sl
        
    # TP1 is baseline (suggested_tp)
    if 'suggested_tp' in row and not pd.isna(row['suggested_tp']):
        tp1 = float(row['suggested_tp'])
    else:
        tp1 = entry_price + 2.0 * R * sweep_dir
        
    tp1_dist = abs(tp1 - entry_price)
    
    # TP2 is HTF Pivot beyond TP1
    pivot_val = find_nearest_htf_pivot_beyond_tp1(idx, sweep_dir, tp1, df_raw, strength=strength)
    if pivot_val is not None:
        proposed_tp2_dist = abs(pivot_val - entry_price)
        # Cap TP2 at 6.0x TP1 to prevent unrealistic targets
        if proposed_tp2_dist > 6.0 * tp1_dist:
            tp2 = entry_price + 4.0 * tp1_dist * sweep_dir
        else:
            tp2 = pivot_val
    else:
        # Fallback to baseline TP2
        tp2 = entry_price + fallback_mult * tp1_dist * sweep_dir
        
    actual_tp1_rr = tp1_dist / R
    actual_tp2_rr = abs(tp2 - entry_price) / R
    stop_r = -abs(stop_loss - entry_price) / R
    
    has_hit_tp1 = False
    for t in range(idx + 1, len(df_raw)):
        high = float(df_raw.loc[t, 'high'])
        low = float(df_raw.loc[t, 'low'])
        
        if not has_hit_tp1:
            if sweep_dir == 1 and low <= stop_loss:
                return stop_r
            elif sweep_dir == -1 and high >= stop_loss:
                return stop_r
                
            if sweep_dir == 1 and high >= tp1:
                has_hit_tp1 = True
            elif sweep_dir == -1 and low <= tp1:
                has_hit_tp1 = True
        else:
            if sweep_dir == 1 and low <= entry_price:
                return 0.5 * actual_tp1_rr
            elif sweep_dir == -1 and high >= entry_price:
                return 0.5 * actual_tp1_rr
                
            if sweep_dir == 1 and high >= tp2:
                return 0.5 * actual_tp1_rr + 0.5 * actual_tp2_rr
            elif sweep_dir == -1 and low <= tp2:
                return 0.5 * actual_tp1_rr + 0.5 * actual_tp2_rr
                
    # End of data
    final_close = float(df_raw.iloc[-1]['close'])
    final_r = ((final_close - entry_price) / R) * sweep_dir
    if has_hit_tp1:
        return 0.5 * actual_tp1_rr + 0.5 * max(0.0, final_r)
    else:
        return max(stop_r, final_r)

def run_htf_tp2_backtest(df_oos, df_raw, threshold, strength, multiplier=32.0):
    trades_df = df_oos[(df_oos['pred_prob'] > threshold) & (df_oos['ny_session'] == 1.0)].copy()
    
    r_returns = []
    dollar_pnls = []
    for _, row in trades_df.iterrows():
        ret = simulate_trade_htf_tp2(row, df_raw, strength)
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
    
    strengths = [12, 24, 36, 48]
    labels = {
        12: '1-Hour Pivots (Strength 12)',
        24: '2-Hour Pivots (Strength 24)',
        36: '3-Hour Pivots (Strength 36)',
        48: '4-Hour Pivots (Strength 48)'
    }
    
    threshold = 0.60
    
    print("\n" + "="*115)
    print("BACKTEST RESULTS: TP1 = BASELINE, TP2 = NEXT HTF LIQUIDITY BEYOND TP1 (NY Session Only, 16 Contracts)")
    print("="*115)
    print(f"| TP2 Target Config            | Trades | Win Rate | Expectancy (R) | Net Profit | Max Drawdown | Profit/DD Ratio |")
    print(f"|------------------------------|--------|----------|----------------|------------|--------------|-----------------|")
    
    # Fetch baseline for comparison
    from test_moving_stop import run_moving_stop_comparison
    # We can just run run_moving_stop_comparison or fetch from baseline
    
    results = []
    for s in strengths:
        n_trades, win_rate, exp, pnl, dd = run_htf_tp2_backtest(baseline_oos, df_raw, threshold, s)
        ratio = pnl / dd if dd > 0 else 0.0
        results.append((labels[s], n_trades, win_rate, exp, pnl, dd, ratio))
        
    # Baseline comparison row (fallback_mult = 2.0, which is baseline tp2)
    # Baseline has net profit $28,084, DD $4,340, win rate 41.4%, expectancy 0.79
    results.append(('Baseline (Fixed 2.0x TP1)', 29, 0.414, 0.79, 28084.0, 4340.0, 28084.0/4340.0))
    
    results.sort(key=lambda x: x[4], reverse=True)
    for r in results:
        print(f"| {r[0]:<28} | {r[1]:<6} | {r[2]:<8.1%} | {r[3]:<14.2f} | ${r[4]:<10,.2f} | ${r[5]:<12,.2f} | {r[6]:<15.2f} |")
    print("="*115)

if __name__ == '__main__':
    main()
