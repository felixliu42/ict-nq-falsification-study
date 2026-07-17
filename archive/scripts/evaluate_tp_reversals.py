import pandas as pd
import numpy as np
import os
import lightgbm as lgb
from datetime import time
from research_harness import calculate_atr, precompute_metrics
from baseline_strategy import clean_columns, simulate_trade_execution
from backtest_variants import run_walk_forward_validation

def find_pivot_properties(target_price, entry_idx, sweep_dir, df_raw, atr_series):
    """
    Scan backward from entry to find the creation bar of the target_price,
    and determine if it was a 1H pivot, 4H pivot, or Daily level, and if it was stacked.
    """
    # Find the bar in the past where high (for longs) or low (for shorts) matches target_price
    found_idx = None
    for p in range(entry_idx - 1, max(0, entry_idx - 3000), -1):
        if sweep_dir == 1: # Long trade: target is high
            if abs(float(df_raw.loc[p, 'high']) - target_price) <= 0.50:
                found_idx = p
                break
        else: # Short trade: target is low
            if abs(float(df_raw.loc[p, 'low']) - target_price) <= 0.50:
                found_idx = p
                break
                
    if found_idx is None:
        return 'daily', 1.0 # Default to daily single level if not found
        
    # Check if this bar corresponds to a pivot high/low on 1H or 4H
    # Let's check 1H pivot (strength 12) and 4H pivot (strength 24) on raw data
    is_4h_pivot = True
    is_1h_pivot = True
    
    # 4H check (strength 24)
    if sweep_dir == 1:
        val = float(df_raw.loc[found_idx, 'high'])
        for j in range(1, 25):
            if found_idx - j >= 0 and float(df_raw.loc[found_idx - j, 'high']) >= val:
                is_4h_pivot = False
                break
            if found_idx + j < len(df_raw) and float(df_raw.loc[found_idx + j, 'high']) >= val:
                is_4h_pivot = False
                break
    else:
        val = float(df_raw.loc[found_idx, 'low'])
        for j in range(1, 25):
            if found_idx - j >= 0 and float(df_raw.loc[found_idx - j, 'low']) <= val:
                is_4h_pivot = False
                break
            if found_idx + j < len(df_raw) and float(df_raw.loc[found_idx + j, 'low']) <= val:
                is_4h_pivot = False
                break
                
    # 1H check (strength 12)
    if sweep_dir == 1:
        val = float(df_raw.loc[found_idx, 'high'])
        for j in range(1, 13):
            if found_idx - j >= 0 and float(df_raw.loc[found_idx - j, 'high']) >= val:
                is_1h_pivot = False
                break
            if found_idx + j < len(df_raw) and float(df_raw.loc[found_idx + j, 'high']) >= val:
                is_1h_pivot = False
                break
    else:
        val = float(df_raw.loc[found_idx, 'low'])
        for j in range(1, 13):
            if found_idx - j >= 0 and float(df_raw.loc[found_idx - j, 'low']) <= val:
                is_1h_pivot = False
                break
            if found_idx + j < len(df_raw) and float(df_raw.loc[found_idx + j, 'low']) <= val:
                is_1h_pivot = False
                break
                
    tf = 'daily'
    if is_4h_pivot:
        tf = '4h'
    elif is_1h_pivot:
        tf = '1h'
        
    # Check if stacked: are there other active swing highs/lows within 1.5x ATR at the time of creation?
    # We can scan the past 1000 bars for other swing points within 1.5 * ATR
    atr_val = atr_series.loc[found_idx]
    stacked_count = 1.0
    
    for p in range(found_idx - 100, found_idx + 100):
        if p < 0 or p >= len(df_raw) or p == found_idx:
            continue
        if sweep_dir == 1: # Highs
            if abs(float(df_raw.loc[p, 'high']) - target_price) <= atr_val * 1.5:
                # verify if it is also a pivot
                is_p = True
                v = float(df_raw.loc[p, 'high'])
                for j in range(1, 6):
                    if p - j >= 0 and float(df_raw.loc[p - j, 'high']) >= v:
                        is_p = False
                        break
                if is_p:
                    stacked_count += 0.5
        else: # Lows
            if abs(float(df_raw.loc[p, 'low']) - target_price) <= atr_val * 1.5:
                is_p = True
                v = float(df_raw.loc[p, 'low'])
                for j in range(1, 6):
                    if p - j >= 0 and float(df_raw.loc[p - j, 'low']) <= v:
                        is_p = False
                        break
                if is_p:
                    stacked_count += 0.5
                    
    return tf, stacked_count

def main():
    df_ml = pd.read_csv('demo_ml_dataset.csv')
    df_raw = clean_columns(pd.read_csv('translated_tv_export.csv'))
    
    df_ml['time'] = pd.to_numeric(df_ml['time'], errors='coerce')
    df_raw['time'] = pd.to_numeric(df_raw['time'], errors='coerce')
    
    print("Precomputing raw indexes and ATR...")
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
    
    # Filter NY-session setups with probability > 0.60
    trades_df = baseline_oos[(baseline_oos['pred_prob'] > 0.60) & (baseline_oos['ny_session'] == 1.0)].copy()
    
    print(f"\nAnalyzing target properties for all {len(trades_df)} trades...")
    
    reconstruction_records = []
    for _, row in trades_df.iterrows():
        entry_time = int(row['time'])
        sweep_dir = int(row['sweep_direction'])
        
        # Get entry index
        idx = raw_time_to_idx.get(entry_time)
        if idx is None:
            continue
            
        target_price = float(row['suggested_tp']) if not pd.isna(row['suggested_tp']) else None
        if target_price is None:
            continue
            
        tf, strength = find_pivot_properties(target_price, idx, sweep_dir, df_raw, atr_series)
        
        # Simulate outcome
        ret, exit_type, duration = simulate_trade_execution(row, df_raw, tp_mode='split')
        
        # A trade reaches TP1 if it hits TP1 (meaning it was exit_type == 'tp2' or 'trail_stop')
        # A trade reverses if it hit trail_stop, and continues if it hit tp2.
        hit_tp1 = exit_type in ['tp2', 'trail_stop']
        hit_tp2 = exit_type == 'tp2'
        
        reconstruction_records.append({
            'time': entry_time,
            'target_tf': tf,
            'target_strength': strength,
            'hit_tp1': int(hit_tp1),
            'hit_tp2': int(hit_tp2),
            'reversal': int(hit_tp1 and not hit_tp2),
            'continuation': int(hit_tp1 and hit_tp2)
        })
        
    df_analysis = pd.DataFrame(reconstruction_records)
    
    # Filter only to trades that hit TP1 (where a reversal could happen)
    df_tp1 = df_analysis[df_analysis['hit_tp1'] == 1].copy()
    
    print("\n" + "="*80)
    print("TARGET LIQUIDITY ANALYSIS (Trades that reached TP1)")
    print("="*80)
    
    # 1. Group by Timeframe
    print("\n1. Reversal Probability by Target Timeframe:")
    tf_groups = df_tp1.groupby('target_tf')
    for tf, group in tf_groups:
        n = len(group)
        rev = group['reversal'].sum()
        cont = group['continuation'].sum()
        rev_prob = rev / n if n > 0 else 0.0
        print(f"  - {tf.upper():<7} | Count: {n:<3} | Reversals: {rev:<2} | Continuations: {cont:<2} | Reversal Prob: {rev_prob:.1%}")
        
    # 2. Group by Stacked vs Single
    print("\n2. Reversal Probability by Target Structure (Stacked vs Single):")
    df_tp1['is_stacked'] = df_tp1['target_strength'] > 1.0
    struct_groups = df_tp1.groupby('is_stacked')
    for is_stacked, group in struct_groups:
        label = "Stacked" if is_stacked else "Single"
        n = len(group)
        rev = group['reversal'].sum()
        cont = group['continuation'].sum()
        rev_prob = rev / n if n > 0 else 0.0
        print(f"  - {label:<7} | Count: {n:<3} | Reversals: {rev:<2} | Continuations: {cont:<2} | Reversal Prob: {rev_prob:.1%}")
        
    # 3. Group by combined TF + Stacked
    print("\n3. Combined Timeframe + Structure Leaderboard:")
    df_tp1['group_name'] = df_tp1['target_tf'] + "_" + np.where(df_tp1['is_stacked'], 'Stacked', 'Single')
    combined_groups = df_tp1.groupby('group_name')
    for name, group in combined_groups:
        n = len(group)
        rev = group['reversal'].sum()
        cont = group['continuation'].sum()
        rev_prob = rev / n if n > 0 else 0.0
        print(f"  - {name:<12} | Count: {n:<3} | Reversals: {rev:<2} | Continuations: {cont:<2} | Reversal Prob: {rev_prob:.1%}")
        
    print("="*80)
    
    # 4. LightGBM Reversal Classifier Evaluation
    if len(df_tp1) >= 10:
        print("\nTraining LightGBM Classifier to predict Target Reversals...")
        # Prepare features
        # Target TF: one-hot encode
        df_tp1['tf_1h'] = (df_tp1['target_tf'] == '1h').astype(int)
        df_tp1['tf_4h'] = (df_tp1['target_tf'] == '4h').astype(int)
        df_tp1['tf_daily'] = (df_tp1['target_tf'] == 'daily').astype(int)
        
        features = ['tf_1h', 'tf_4h', 'tf_daily', 'target_strength']
        X = df_tp1[features]
        y = df_tp1['reversal']
        
        clf = lgb.LGBMClassifier(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42, min_child_samples=2, verbosity=-1)
        clf.fit(X, y)
        
        importances = clf.feature_importances_
        print("LightGBM Feature Importances for predicting Reversal at TP1:")
        for feat, imp in zip(features, importances):
            print(f"  - {feat:<20}: {imp}")
    else:
        print("\nNot enough data points to train LightGBM classifier (need at least 10 trades that hit TP1).")

if __name__ == '__main__':
    main()
