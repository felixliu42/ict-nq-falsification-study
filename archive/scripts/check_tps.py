import pandas as pd
import numpy as np
from research_harness import calculate_atr
from baseline_strategy import clean_columns

def main():
    df_ml = pd.read_csv('demo_ml_dataset.csv')
    df_raw = clean_columns(pd.read_csv('translated_tv_export.csv'))
    
    atr_series = calculate_atr(df_raw, 14)
    raw_times = df_raw['time'].values
    raw_time_to_idx = {t: idx for idx, t in enumerate(raw_times)}
    
    fallback_count = 0
    total_count = 0
    
    rrs_liq = []
    rrs_atr = []
    
    for _, row in df_ml.iterrows():
        idx = raw_time_to_idx.get(row['time'])
        if idx is None or pd.isna(row['suggested_tp']):
            continue
            
        close = float(df_raw.loc[idx, 'close'])
        atr = atr_series.loc[idx]
        target_dist = abs(row['suggested_tp'] - close)
        
        # Calculate risk R
        sweep_idx = idx - int(row['time_since_sweep'])
        if sweep_idx < 0:
            continue
            
        if row['sweep_direction'] == -1:
            sweep_extreme = float(df_raw.loc[sweep_idx, 'high'])
        else:
            sweep_extreme = float(df_raw.loc[sweep_idx, 'low'])
            
        R = abs(close - sweep_extreme)
        if R <= 0:
            continue
            
        total_count += 1
        
        # Check if target is close to 2.0 * ATR
        if abs(target_dist - 2.0 * atr) < 1.0: # within $1 price difference
            fallback_count += 1
            rrs_atr.append(target_dist / R)
        else:
            rrs_liq.append(target_dist / R)
            
    print(f"Total valid setups analyzed: {total_count}")
    print(f"Fallback (ATR) setups: {fallback_count} ({fallback_count/total_count:.1%})")
    print(f"Liquidity pool setups: {total_count - fallback_count} ({(total_count - fallback_count)/total_count:.1%})")
    print(f"Average R:R for ATR targets: {np.mean(rrs_atr):.2f}R")
    print(f"Average R:R for Liquidity targets: {np.mean(rrs_liq):.2f}R")

if __name__ == '__main__':
    main()
