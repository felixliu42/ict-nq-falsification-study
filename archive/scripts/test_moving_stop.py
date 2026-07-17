import pandas as pd
import numpy as np
import os
import lightgbm as lgb
from datetime import time
from research_harness import calculate_atr, precompute_metrics
from baseline_strategy import clean_columns, calculate_metrics
from backtest_variants import run_walk_forward_validation

def simulate_trade_moving_stop(row, df_raw, tp_mode='split', left_strength=2, right_strength=2, buffer_atr_mult=0.5, atr_series=None):
    """
    Simulate trade execution using a structure-based (pivot-based) moving stop-loss.
    """
    entry_time = int(row['time'])
    sweep_dir = int(row['sweep_direction'])
    time_since = int(row['time_since_sweep'])
    
    # Find entry bar index
    raw_indices = df_raw.index[df_raw['time'] == entry_time].tolist()
    if not raw_indices:
        return None, 'missing_data', None
    idx = raw_indices[0]
    
    entry_price = float(df_raw.loc[idx, 'close'])
    sweep_idx = idx - time_since
    if sweep_idx < 0:
        return None, 'before_history', None
        
    # Initial stop loss (sweep extreme)
    if sweep_dir == -1: # Bearish Setup
        initial_sl = float(df_raw.loc[sweep_idx, 'high'])
    else: # Bullish Setup
        initial_sl = float(df_raw.loc[sweep_idx, 'low'])
        
    R = abs(entry_price - initial_sl)
    if R == 0.0:
        return None, 'zero_risk', None
        
    # Take profit levels
    if 'suggested_tp' in row and not pd.isna(row['suggested_tp']):
        tp1 = float(row['suggested_tp'])
    else:
        tp1 = entry_price + 2.0 * R * sweep_dir
        
    reward = abs(tp1 - entry_price)
    tp2 = entry_price + 2.0 * reward * sweep_dir
    
    actual_rr = reward / R
    
    # Track current stop-loss price
    current_sl = initial_sl
    has_hit_tp1 = False
    
    # Loop bar-by-bar starting after the entry bar
    for t in range(idx + 1, len(df_raw)):
        high = float(df_raw.loc[t, 'high'])
        low = float(df_raw.loc[t, 'low'])
        close = float(df_raw.loc[t, 'close'])
        
        duration = t - idx
        
        # 1. Update structure-based trailing stop
        # A pivot is confirmed at index p = t - right_strength
        p = t - right_strength
        if p >= left_strength:
            if sweep_dir == 1: # Long trade: look for pivot lows to move stop UP
                is_pivot = True
                p_low = float(df_raw.loc[p, 'low'])
                for j in range(1, left_strength + 1):
                    if float(df_raw.loc[p - j, 'low']) <= p_low:
                        is_pivot = False
                        break
                if is_pivot:
                    for j in range(1, right_strength + 1):
                        if float(df_raw.loc[p + j, 'low']) <= p_low:
                            is_pivot = False
                            break
                if is_pivot:
                    atr_val = atr_series.loc[p] if atr_series is not None else 0.0
                    proposed_sl = p_low - buffer_atr_mult * atr_val
                    if proposed_sl > current_sl:
                        current_sl = proposed_sl
                        
            else: # Short trade: look for pivot highs to move stop DOWN
                is_pivot = True
                p_high = float(df_raw.loc[p, 'high'])
                for j in range(1, left_strength + 1):
                    if float(df_raw.loc[p - j, 'high']) >= p_high:
                        is_pivot = False
                        break
                if is_pivot:
                    for j in range(1, right_strength + 1):
                        if float(df_raw.loc[p + j, 'high']) >= p_high:
                            is_pivot = False
                            break
                if is_pivot:
                    atr_val = atr_series.loc[p] if atr_series is not None else 0.0
                    proposed_sl = p_high + buffer_atr_mult * atr_val
                    if proposed_sl < current_sl:
                        current_sl = proposed_sl
                        
        # 2. Check exits
        stop_r = -abs(current_sl - entry_price) / R
        
        if tp_mode == 'tp1':
            if sweep_dir == 1: # Bullish
                hit_stop = low <= current_sl
                hit_tp1 = high >= tp1
                if hit_tp1 and hit_stop:
                    return stop_r, 'stop', duration
                elif hit_stop:
                    return stop_r, 'stop', duration
                elif hit_tp1:
                    return actual_rr, 'tp1', duration
            else: # Bearish
                hit_stop = high >= current_sl
                hit_tp1 = low <= tp1
                if hit_tp1 and hit_stop:
                    return stop_r, 'stop', duration
                elif hit_stop:
                    return stop_r, 'stop', duration
                elif hit_tp1:
                    return actual_rr, 'tp1', duration
                    
        elif tp_mode == 'split':
            if not has_hit_tp1:
                if sweep_dir == 1: # Bullish
                    hit_stop = low <= current_sl
                    hit_tp1 = high >= tp1
                    if hit_tp1 and hit_stop:
                        return stop_r, 'stop', duration
                    elif hit_stop:
                        return stop_r, 'stop', duration
                    elif hit_tp1:
                        has_hit_tp1 = True
                        # Lock in breakeven if breakeven is tighter than current structural stop
                        current_sl = max(entry_price, current_sl)
                        if low <= current_sl:
                            return 0.5 * actual_rr, 'trail_stop', duration
                else: # Bearish
                    hit_stop = high >= current_sl
                    hit_tp1 = low <= tp1
                    if hit_tp1 and hit_stop:
                        return stop_r, 'stop', duration
                    elif hit_stop:
                        return stop_r, 'stop', duration
                    elif hit_tp1:
                        has_hit_tp1 = True
                        current_sl = min(entry_price, current_sl)
                        if high >= current_sl:
                            return 0.5 * actual_rr, 'trail_stop', duration
            else:
                # Looking for TP2 or Trail Stop
                if sweep_dir == 1: # Bullish
                    hit_trail = low <= current_sl
                    hit_tp2 = high >= tp2
                    if hit_tp2 and hit_trail:
                        return 0.5 * actual_rr, 'trail_stop', duration
                    elif hit_trail:
                        # Exited second half at current_sl
                        exit_r = (current_sl - entry_price) / R
                        return 0.5 * actual_rr + 0.5 * exit_r, 'trail_stop', duration
                    elif hit_tp2:
                        return 1.5 * actual_rr, 'tp2', duration
                else: # Bearish
                    hit_trail = high >= current_sl
                    hit_tp2 = low <= tp2
                    if hit_tp2 and hit_trail:
                        return 0.5 * actual_rr, 'trail_stop', duration
                    elif hit_trail:
                        exit_r = (entry_price - current_sl) / R
                        return 0.5 * actual_rr + 0.5 * exit_r, 'trail_stop', duration
                    elif hit_tp2:
                        return 1.5 * actual_rr, 'tp2', duration
                        
    # End of data
    final_close = float(df_raw.iloc[-1]['close'])
    final_return = ((final_close - entry_price) / R) * sweep_dir
    duration = len(df_raw) - 1 - idx
    
    if tp_mode == 'tp1':
        return max(stop_r, min(actual_rr, final_return)), 'end_of_data', duration
    else:
        if has_hit_tp1:
            exit_r = (current_sl - entry_price) / R * sweep_dir
            second_half_r = max(exit_r, min(actual_rr, final_return * 0.5))
            return 0.5 * actual_rr + second_half_r, 'end_of_data', duration
        else:
            return max(stop_r, min(0.5 * actual_rr, final_return)), 'end_of_data', duration

def run_moving_stop_comparison(df_oos, df_raw, atr_series, threshold, tp_mode, multiplier):
    trades_df = df_oos[(df_oos['pred_prob'] > threshold) & (df_oos['ny_session'] == 1.0)].copy()
    
    # 1. Fixed Stop Loss Simulation
    fixed_dollar_pnls = []
    fixed_r_returns = []
    for _, row in trades_df.iterrows():
        # Baseline uses baseline_strategy's simulate_trade_execution
        from baseline_strategy import simulate_trade_execution
        ret, _, _ = simulate_trade_execution(row, df_raw, tp_mode=tp_mode)
        if ret is not None:
            fixed_r_returns.append(ret)
            R = row['trade_R']
            if R is not None and not pd.isna(R):
                fixed_dollar_pnls.append(multiplier * R * ret)
                
    # 2. Moving Stop Loss Simulation
    moving_dollar_pnls = []
    moving_r_returns = []
    for _, row in trades_df.iterrows():
        ret, _, _ = simulate_trade_moving_stop(row, df_raw, tp_mode=tp_mode, left_strength=2, right_strength=2, buffer_atr_mult=0.5, atr_series=atr_series)
        if ret is not None:
            moving_r_returns.append(ret)
            R = row['trade_R']
            if R is not None and not pd.isna(R):
                moving_dollar_pnls.append(multiplier * R * ret)
                
    # Metrics Fixed
    fixed_metrics = calculate_metrics(fixed_r_returns, [0]*len(fixed_r_returns))
    fixed_net_profit = sum(fixed_dollar_pnls)
    fixed_cum = [100000.0] + list(100000.0 + np.cumsum(fixed_dollar_pnls))
    fixed_max_dd = np.max(np.maximum.accumulate(fixed_cum) - fixed_cum)
    
    # Metrics Moving
    moving_metrics = calculate_metrics(moving_r_returns, [0]*len(moving_r_returns))
    moving_net_profit = sum(moving_dollar_pnls)
    moving_cum = [100000.0] + list(100000.0 + np.cumsum(moving_dollar_pnls))
    moving_max_dd = np.max(np.maximum.accumulate(moving_cum) - moving_cum)
    
    return {
        'trades': len(trades_df),
        'fixed_win_rate': fixed_metrics['win_rate'],
        'fixed_exp': fixed_metrics['expectancy'],
        'fixed_profit': fixed_net_profit,
        'fixed_max_dd': fixed_max_dd,
        'moving_win_rate': moving_metrics['win_rate'],
        'moving_exp': moving_metrics['expectancy'],
        'moving_profit': moving_net_profit,
        'moving_max_dd': moving_max_dd
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
    precompute_metrics(df_ml, df_raw, raw_time_to_idx, atr_series)
    
    # Add time helpers for filtering
    df_ml['dt'] = pd.to_datetime(df_ml['time'], unit='ms').dt.tz_localize('UTC').dt.tz_convert('America/New_York')
    df_ml['date'] = df_ml['dt'].dt.date
    df_ml['time_only'] = df_ml['dt'].dt.time
    
    # Sort chronologically
    df_ml = df_ml.sort_values('time').reset_index(drop=True)
    df_ml['sweep_id'] = df_ml['time'] - df_ml['time_since_sweep'] * 300000
    df_ml['setup_num'] = df_ml.groupby('sweep_id').cumcount() + 1
    
    # Run walk-forward to get baseline predictions
    print("\nRunning Walk-Forward Validation...")
    baseline_oos = run_walk_forward_validation(df_ml)
    
    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70]
    modes = ['tp1', 'split']
    multiplier = 32.0 # 16 contracts
    
    print("\n" + "="*145)
    print("COMPARATIVE BACKTEST: FIXED STOP vs STRUCTURE-BASED MOVING STOP (16 Contracts, NY Session Only)")
    print("="*145)
    print(f"| TP Mode | Thresh | Trades | Fixed Win% | Moving Win% | Fixed Exp (R) | Moving Exp (R) | Fixed Profit | Moving Profit | Fixed Max DD | Moving Max DD |")
    print(f"|---------|--------|--------|------------|-------------|---------------|----------------|--------------|---------------|--------------|---------------|")
    
    for mode in modes:
        mode_str = 'TP1 Only' if mode == 'tp1' else 'Split TP1/TP2'
        for th in thresholds:
            res = run_moving_stop_comparison(baseline_oos, df_raw, atr_series, th, mode, multiplier)
            if res['trades'] > 0:
                print(f"| {mode_str:<12} | {th:<6.2f} | {res['trades']:<6} | {res['fixed_win_rate']:<10.1%} | {res['moving_win_rate']:<11.1%} | {res['fixed_exp']:<13.2f} | {res['moving_exp']:<14.2f} | ${res['fixed_profit']:<11,.2f} | ${res['moving_profit']:<12,.2f} | ${res['fixed_max_dd']:<11,.2f} | ${res['moving_max_dd']:<12,.2f} |")
                
    print("="*145)

if __name__ == '__main__':
    main()
