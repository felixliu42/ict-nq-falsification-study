import pandas as pd
import numpy as np
import os
from datetime import time
import lightgbm as lgb
from research_harness import calculate_atr, precompute_metrics
from baseline_strategy import clean_columns, simulate_trade_execution, calculate_metrics

def run_walk_forward_validation(df_ml, min_train_size=30, step_size=5):
    """
    Run expanding window walk-forward validation to generate out-of-sample predicted probabilities.
    """
    features = [
        'liquidity_type', 'liquidity_strength', 'sweep_direction', 'sweep_size',
        'bos_strength', 'fvg_rejected', 'retracement_depth', 'time_since_sweep',
        'ny_session', 'london_session', 'asian_session'
    ]
    
    df_ml = df_ml.sort_values('time').reset_index(drop=True)
    df_ml['pred_prob'] = np.nan
    
    n_rows = len(df_ml)
    if n_rows <= min_train_size:
        # If too small, return empty or fallback
        return pd.DataFrame()
        
    for t in range(min_train_size, n_rows, step_size):
        train_df = df_ml.iloc[:t]
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
        
    df_oos = df_ml.iloc[min_train_size:].copy().reset_index(drop=True)
    return df_oos

def filter_setups(df_ml_exp, sess_cfg='none', enforce_daily_bias=False, bias_reset_mode='entire_day', max_setup=3, allowed_liquidity_types='all', allowed_penetration=['small', 'medium', 'large']):
    filtered_indices = []
    current_bias_date = None
    active_bias = None
    
    for idx, row in df_ml_exp.iterrows():
        t_val = row['time_only']
        # Session Filter
        if sess_cfg == 'morning_only':
            if not (time(9, 30) <= t_val <= time(12, 0)):
                continue
        elif sess_cfg == 'lunch_excluded':
            if not (time(9, 30) <= t_val <= time(16, 0)) or (time(12, 0) <= t_val <= time(13, 30)):
                continue
                
        # Setup Number Filter
        if row['setup_num'] > max_setup:
            continue
            
        # Penetration Filter
        if row['penetration_bucket'] not in allowed_penetration:
            continue
            
        # Liquidity Filter
        l_type = int(row['liquidity_type'])
        l_strength = float(row['liquidity_strength'])
        
        if allowed_liquidity_types == 'daily_only':
            if l_type != 1: continue
        elif allowed_liquidity_types == 'daily_stacked_4h':
            if not (l_type == 1 or (l_type == 3 and l_strength > 1.0)):
                continue
        elif allowed_liquidity_types == 'stacked_only':
            if l_strength <= 1.0: continue
        elif allowed_liquidity_types == 'exclude_singular_1h':
            if l_type == 2 and l_strength == 1.0: continue
            
        # Daily Bias Filter
        if enforce_daily_bias:
            setup_date = row['date']
            sweep_dir = int(row['sweep_direction'])
            
            # Reset bias if new calendar day
            if current_bias_date != setup_date:
                current_bias_date = setup_date
                active_bias = None
                
            # Lunch Reset
            if bias_reset_mode == 'lunch_reset' and t_val >= time(12, 0):
                if active_bias is not None and t_val >= time(12, 0) and (row['time'] - df_ml_exp.loc[filtered_indices[-1], 'time'] if filtered_indices else 0) > 0:
                     active_bias = None
                     
            # HTF Sweep Reset
            if bias_reset_mode == 'htf_sweep' and l_type in [1, 3]:
                active_bias = None
                
            # Reject setups in the opposite direction
            if active_bias is not None and sweep_dir != active_bias:
                continue
                
            # Establish new bias if neutral
            if active_bias is None:
                active_bias = sweep_dir
                
        filtered_indices.append(idx)
        
    return df_ml_exp.loc[filtered_indices].copy().reset_index(drop=True)

def print_variant_report(variant_name, df_oos, df_raw):
    print(f"\n=====================================================================================================")
    print(f"WALK-FORWARD ML BACKTEST RESULTS FOR: {variant_name}")
    print(f"=====================================================================================================")
    
    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    modes = ['tp1', 'split']
    
    results = []
    
    for mode in modes:
        for th in thresholds:
            trades_df = df_oos[df_oos['pred_prob'] > th].copy()
            trade_returns = []
            trade_durations = []
            
            for _, row in trades_df.iterrows():
                ret, outcome, duration = simulate_trade_execution(row, df_raw, tp_mode=mode)
                if ret is not None:
                    trade_returns.append(ret)
                    trade_durations.append(duration)
                    
            metrics = calculate_metrics(trade_returns, trade_durations)
            metrics['threshold'] = th
            metrics['tp_mode'] = 'TP1' if mode == 'tp1' else 'Split'
            results.append(metrics)
            
    print(f"| {'Threshold':<9} | {'TP Mode':<7} | {'Trades':<6} | {'Win Rate':<8} | {'PF':<6} | {'Expectancy (R)':<14} | {'Max DD (R)':<10} | {'Avg Win (R)':<11} | {'Avg Loss (R)':<12} | {'W/L Ratio':<9} | {'Avg Duration':<12} |")
    print(f"|{'-'*11}|{'-'*9}|{'-'*8}|{'-'*10}|{'-'*8}|{'-'*16}|{'-'*12}|{'-'*13}|{'-'*14}|{'-'*11}|{'-'*14}|")
    for r in results:
        pf_str = f"{r['profit_factor']:.2f}" if r['profit_factor'] != np.inf else "Inf"
        wl_str = f"{r['win_loss_ratio']:.2f}" if r['win_loss_ratio'] != np.inf else "Inf"
        if r['total_trades'] == 0:
            pf_str = "N/A"
            wl_str = "N/A"
        print(f"| {r['threshold']:<9.2f} | {r['tp_mode']:<7} | {r['total_trades']:<6} | {r['win_rate']:<8.1%} | {pf_str:<6} | {r['expectancy']:<14.2f} | {r['max_dd']:<10.2f} | {r['avg_win']:<11.2f} | {r['avg_loss']:<12.2f} | {wl_str:<9} | {r['avg_duration']:<12.1f} |")
    print("="*105)
    
    # Per-Session Breakdown at best thresholds
    # Let's find best threshold for TP1 and Split (highest expectancy with trades >= 10)
    best_tp1_th = 0.50
    best_tp1_exp = -99.0
    best_split_th = 0.50
    best_split_exp = -99.0
    
    for r in results:
        if r['total_trades'] >= 8:
            if r['tp_mode'] == 'TP1' and r['expectancy'] > best_tp1_exp:
                best_tp1_exp = r['expectancy']
                best_tp1_th = r['threshold']
            elif r['tp_mode'] == 'Split' and r['expectancy'] > best_split_exp:
                best_split_exp = r['expectancy']
                best_split_th = r['threshold']
                
    print("\nPER-SESSION BREAKDOWN FOR BEST THRESHOLDS:")
    print(f"| {'Threshold':<9} | {'TP Mode':<7} | {'Session':<12} | {'Trades':<6} | {'Win Rate':<8} | {'Expectancy (R)':<14} | {'Avg Win (R)':<11} | {'Avg Loss (R)':<12} | {'W/L Ratio':<9} | {'Avg Duration':<12} |")
    print(f"|{'-'*11}|{'-'*9}|{'-'*14}|{'-'*8}|{'-'*10}|{'-'*16}|{'-'*13}|{'-'*14}|{'-'*11}|{'-'*14}|")
    
    for best_th, mode in [(best_tp1_th, 'tp1'), (best_split_th, 'split')]:
        mode_str = 'TP1' if mode == 'tp1' else 'Split'
        trades_df = df_oos[df_oos['pred_prob'] > best_th].copy()
        
        session_returns = {'NY': [], 'London': [], 'Asian': [], 'Other': []}
        session_durations = {'NY': [], 'London': [], 'Asian': [], 'Other': []}
        
        for _, row in trades_df.iterrows():
            ret, _, duration = simulate_trade_execution(row, df_raw, tp_mode=mode)
            if ret is not None:
                matched_session = False
                if row['ny_session'] == 1:
                    session_returns['NY'].append(ret)
                    session_durations['NY'].append(duration)
                    matched_session = True
                if row['london_session'] == 1:
                    session_returns['London'].append(ret)
                    session_durations['London'].append(duration)
                    matched_session = True
                if row['asian_session'] == 1:
                    session_returns['Asian'].append(ret)
                    session_durations['Asian'].append(duration)
                    matched_session = True
                if not matched_session:
                    session_returns['Other'].append(ret)
                    session_durations['Other'].append(duration)
                    
        for sess, rets in session_returns.items():
            if len(rets) > 0:
                sess_metrics = calculate_metrics(rets, session_durations[sess])
                wl_str = f"{sess_metrics['win_loss_ratio']:.2f}" if sess_metrics['win_loss_ratio'] != np.inf else "Inf"
                if sess_metrics['total_trades'] == 0:
                    wl_str = "N/A"
                print(f"| {best_th:<9.2f} | {mode_str:<7} | {sess:<12} | {sess_metrics['total_trades']:<6} | {sess_metrics['win_rate']:<8.1%} | {sess_metrics['expectancy']:<14.2f} | {sess_metrics['avg_win']:<11.2f} | {sess_metrics['avg_loss']:<12.2f} | {wl_str:<9} | {sess_metrics['avg_duration']:<12.1f} |")
    print("="*115)

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
    
    # --------------------------------------------------
    # Variant 1: Sess: morning_only | Daily Bias Reset: htf_sweep
    # --------------------------------------------------
    print("\nRunning Variant 1 filter...")
    v1_df = filter_setups(
        df_ml,
        sess_cfg='morning_only',
        enforce_daily_bias=True,
        bias_reset_mode='htf_sweep',
        max_setup=3,
        allowed_liquidity_types='all'
    )
    print(f"Variant 1 setups: {len(v1_df)}")
    v1_oos = run_walk_forward_validation(v1_df)
    if not v1_oos.empty:
        print_variant_report("Sess: morning_only | Daily Bias Reset: htf_sweep", v1_oos, df_raw)
    else:
        print("Insufficient setups for Variant 1 walk-forward validation.")
        
    # --------------------------------------------------
    # Variant 2: Sess: morning_only | Stop Mult: 0.90
    # --------------------------------------------------
    print("\nRunning Variant 2 filter...")
    v2_df = filter_setups(
        df_ml,
        sess_cfg='morning_only',
        enforce_daily_bias=False,
        max_setup=3,
        allowed_liquidity_types='all'
    )
    print(f"Variant 2 setups: {len(v2_df)}")
    # Apply Stop Multiplier = 0.90
    for idx, row in v2_df.iterrows():
        R = row['trade_R']
        if R is not None and not pd.isna(R):
            entry_price = row['entry_price']
            expanded_sl = entry_price - row['sweep_direction'] * R * 0.90
            v2_df.loc[idx, 'suggested_sl'] = expanded_sl
            
    v2_oos = run_walk_forward_validation(v2_df)
    if not v2_oos.empty:
        print_variant_report("Sess: morning_only | Stop Mult: 0.90", v2_oos, df_raw)
    else:
        print("Insufficient setups for Variant 2 walk-forward validation.")
        
    # --------------------------------------------------
    # Variant 3: Liq: stacked_only | Max Setups: 1
    # --------------------------------------------------
    print("\nRunning Variant 3 filter...")
    v3_df = filter_setups(
        df_ml,
        sess_cfg='none',
        enforce_daily_bias=False,
        max_setup=1,
        allowed_liquidity_types='stacked_only'
    )
    print(f"Variant 3 setups: {len(v3_df)}")
    v3_oos = run_walk_forward_validation(v3_df)
    if not v3_oos.empty:
        print_variant_report("Liq: stacked_only | Max Setups: 1", v3_oos, df_raw)
    else:
        print("Insufficient setups for Variant 3 walk-forward validation.")

if __name__ == '__main__':
    main()
