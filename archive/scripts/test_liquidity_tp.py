import pandas as pd
import numpy as np
from research_harness import calculate_atr, precompute_metrics
from baseline_strategy import clean_columns, calculate_metrics
from backtest_variants import run_walk_forward_validation

def simulate_trade_liquidity_tp(row, df_raw, tp1_type, tp1_param, multiplier=32.0):
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
        
    # Stop loss
    if 'suggested_sl' in row and not pd.isna(row['suggested_sl']):
        stop_loss = float(row['suggested_sl'])
    else:
        stop_loss = initial_sl
        
    # TP2 is the next liquidity level
    if 'suggested_tp' in row and not pd.isna(row['suggested_tp']):
        tp2 = float(row['suggested_tp'])
    else:
        tp2 = entry_price + 4.0 * R * sweep_dir # Fallback to 4R if missing
        
    tp2_dist = abs(tp2 - entry_price)
    
    # Calculate TP1 based on configuration
    if tp1_type == 'r_multiplier':
        tp1 = entry_price + tp1_param * R * sweep_dir
    elif tp1_type == 'fraction':
        tp1 = entry_price + tp1_param * (tp2 - entry_price)
        
    tp1_dist = abs(tp1 - entry_price)
    
    # Handle logical edge case where TP1 is calculated to be past TP2
    if tp1_dist >= tp2_dist:
        tp1 = entry_price + 0.5 * tp2_dist * sweep_dir
        tp1_dist = abs(tp1 - entry_price)
        
    actual_tp1_rr = tp1_dist / R
    actual_tp2_rr = tp2_dist / R
    stop_r = -abs(stop_loss - entry_price) / R
    
    # Run simulation bar-by-bar
    has_hit_tp1 = False
    for t in range(idx + 1, len(df_raw)):
        high = float(df_raw.loc[t, 'high'])
        low = float(df_raw.loc[t, 'low'])
        
        if not has_hit_tp1:
            # Check stop loss
            if sweep_dir == 1 and low <= stop_loss:
                return stop_r
            elif sweep_dir == -1 and high >= stop_loss:
                return stop_r
                
            # Check TP1
            if sweep_dir == 1 and high >= tp1:
                has_hit_tp1 = True
            elif sweep_dir == -1 and low <= tp1:
                has_hit_tp1 = True
        else:
            # Once TP1 is hit, stop moves to breakeven (entry_price)
            if sweep_dir == 1 and low <= entry_price:
                return 0.5 * actual_tp1_rr
            elif sweep_dir == -1 and high >= entry_price:
                return 0.5 * actual_tp1_rr
                
            # Check TP2
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

def run_tp_backtest(df_oos, df_raw, threshold, tp1_type, tp1_param, multiplier=32.0):
    trades_df = df_oos[(df_oos['pred_prob'] > threshold) & (df_oos['ny_session'] == 1.0)].copy()
    
    r_returns = []
    dollar_pnls = []
    for _, row in trades_df.iterrows():
        ret = simulate_trade_liquidity_tp(row, df_raw, tp1_type, tp1_param)
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
    
    # Test cases where TP2 is the next liquidity level (suggested_tp)
    test_cases = [
        ('fraction', 0.5, '0.5x of Liquidity Distance'),
        ('r_multiplier', 1.0, '1.0R of Risk'),
        ('r_multiplier', 1.5, '1.5R of Risk'),
        ('r_multiplier', 2.0, '2.0R of Risk'),
        ('r_multiplier', 2.5, '2.5R of Risk')
    ]
    
    threshold = 0.60
    
    print("\n" + "="*115)
    print("BACKTEST RESULTS: TP2 = NEXT LIQUIDITY LEVEL (NY Session Only, ML Threshold 0.60, 16 Contracts)")
    print("="*115)
    print(f"| TP1 Target Config            | Trades | Win Rate | Expectancy (R) | Net Profit | Max Drawdown | Profit/DD Ratio |")
    print(f"|------------------------------|--------|----------|----------------|------------|--------------|-----------------|")
    
    results = []
    for t_type, param, label in test_cases:
        n_trades, win_rate, exp, pnl, dd = run_tp_backtest(baseline_oos, df_raw, threshold, t_type, param)
        ratio = pnl / dd if dd > 0 else 0.0
        results.append((label, n_trades, win_rate, exp, pnl, dd, ratio))
        
    results.sort(key=lambda x: x[4], reverse=True)
    for r in results:
        print(f"| {r[0]:<28} | {r[1]:<6} | {r[2]:<8.1%} | {r[3]:<14.2f} | ${r[4]:<10,.2f} | ${r[5]:<12,.2f} | {r[6]:<15.2f} |")
    print("="*115)

if __name__ == '__main__':
    main()
