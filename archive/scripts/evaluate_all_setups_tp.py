import pandas as pd
import numpy as np
from research_harness import calculate_atr, precompute_metrics
from baseline_strategy import clean_columns, simulate_trade_execution
from evaluate_tp_reversals import find_pivot_properties

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
    
    df_ml = df_ml.sort_values('time').reset_index(drop=True)
    
    print(f"\nAnalyzing target properties for ALL {len(df_ml)} setups in the dataset...")
    
    records = []
    # Loop over all setups (this may take a minute, so we'll do it efficiently)
    for i, row in df_ml.iterrows():
        entry_time = int(row['time'])
        sweep_dir = int(row['sweep_direction'])
        
        idx = raw_time_to_idx.get(entry_time)
        if idx is None:
            continue
            
        target_price = float(row['suggested_tp']) if not pd.isna(row['suggested_tp']) else None
        if target_price is None:
            continue
            
        tf, strength = find_pivot_properties(target_price, idx, sweep_dir, df_raw, atr_series)
        
        # Simulate outcome
        ret, exit_type, duration = simulate_trade_execution(row, df_raw, tp_mode='split')
        
        hit_tp1 = exit_type in ['tp2', 'trail_stop']
        hit_tp2 = exit_type == 'tp2'
        
        records.append({
            'target_tf': tf,
            'target_strength': strength,
            'hit_tp1': int(hit_tp1),
            'hit_tp2': int(hit_tp2),
            'reversal': int(hit_tp1 and not hit_tp2),
            'continuation': int(hit_tp1 and hit_tp2)
        })
        
    df_analysis = pd.DataFrame(records)
    df_tp1 = df_analysis[df_analysis['hit_tp1'] == 1].copy()
    
    print("\n" + "="*80)
    print(f"ROBUST LIQUIDITY ANALYSIS (Sample Size: {len(df_tp1)} trades that reached TP1)")
    print("="*80)
    
    # 1. Group by Timeframe
    print("\n1. Reversal Probability by Target Timeframe:")
    tf_groups = df_tp1.groupby('target_tf')
    for tf, group in tf_groups:
        n = len(group)
        rev = group['reversal'].sum()
        cont = group['continuation'].sum()
        rev_prob = rev / n if n > 0 else 0.0
        print(f"  - {tf.upper():<7} | Count: {n:<5} | Reversals: {rev:<4} | Continuations: {cont:<4} | Reversal Prob: {rev_prob:.1%}")
        
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
        print(f"  - {label:<7} | Count: {n:<5} | Reversals: {rev:<4} | Continuations: {cont:<4} | Reversal Prob: {rev_prob:.1%}")
        
    # 3. Group by combined TF + Stacked
    print("\n3. Combined Timeframe + Structure Leaderboard:")
    df_tp1['group_name'] = df_tp1['target_tf'] + "_" + np.where(df_tp1['is_stacked'], 'Stacked', 'Single')
    combined_groups = df_tp1.groupby('group_name')
    for name, group in combined_groups:
        n = len(group)
        rev = group['reversal'].sum()
        cont = group['continuation'].sum()
        rev_prob = rev / n if n > 0 else 0.0
        print(f"  - {name:<12} | Count: {n:<5} | Reversals: {rev:<4} | Continuations: {cont:<4} | Reversal Prob: {rev_prob:.1%}")
        
    print("="*80)

if __name__ == '__main__':
    main()
