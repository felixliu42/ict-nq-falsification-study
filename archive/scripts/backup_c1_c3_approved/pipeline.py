import pandas as pd
import numpy as np
import os
import argparse

def clean_columns(df):
    """
    Clean column names to handle TradingView suffixes and casing.
    """
    cols = {}
    standard_cols = [
        'time', 'open', 'high', 'low', 'close', 'volume',
        'sweep_direction', 'liquidity_type', 'liquidity_strength',
        'bos_down_strength', 'bearish_fvg_rejected', 'bearish_displacement_size',
        'bos_up_strength', 'bullish_fvg_rejected', 'bullish_displacement_size',
        'retracement_depth', 'distance_to_equilibrium', 'time_since_sweep',
        'ny_session', 'london_session', 'asian_session',
        'suggested_tp', 'suggested_sl'
    ]
    # Sort standard_cols by length descending to match longer strings first (e.g. time_since_sweep before time)
    standard_cols = sorted(standard_cols, key=len, reverse=True)
    
    for col in df.columns:
        col_lower = str(col).lower().strip()
        matched = False
        for std in standard_cols:
            if std in col_lower:
                cols[col] = std
                matched = True
                break
        if not matched:
            cols[col] = col_lower
    return df.rename(columns=cols)

def process_pipeline(input_csv, output_csv=None):
    """
    Load TradingView CSV export, segment sequences, label outcomes,
    calculate excursion metrics, and export processed ML dataset.
    """
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input CSV file not found: {input_csv}")
        
    print(f"Loading dataset from: {input_csv}")
    df = pd.read_csv(input_csv)
    df = clean_columns(df)
    
    # Print columns found for verification
    print("Columns identified in CSV:")
    for col in ['time', 'open', 'high', 'low', 'close', 'sweep_direction', 'valid_setup']:
        if col in df.columns:
            print(f"  - {col}: Detected")
        else:
            print(f"  - {col}: NOT DETECTED (will look for sweep_direction values)")

    # Convert numeric columns
    numeric_cols = ['open', 'high', 'low', 'close', 'sweep_direction', 'time_since_sweep', 
                    'liquidity_type', 'liquidity_strength', 'bos_down_strength', 'bos_up_strength',
                    'bearish_fvg_rejected', 'bullish_fvg_rejected', 'ny_session', 'london_session', 'asian_session',
                    'suggested_tp', 'suggested_sl']
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    # Find entry indices
    # Entry occurs when sweep_direction is not null and it is the start of a sequence
    is_valid = df['sweep_direction'].notna() & (df['sweep_direction'] != 0)
    
    # We define entry as the first bar where setup is valid
    is_new_seq = (
        is_valid & 
        (
            df['sweep_direction'].shift(1).isna() | 
            (df['time_since_sweep'] == 0) | 
            (df['time_since_sweep'] < df['time_since_sweep'].shift(1))
        )
    )
    
    entry_indices = df[is_new_seq].index.tolist()
    print(f"Identified {len(entry_indices)} trade setup sequences.")
    
    processed_records = []
    
    for idx in entry_indices:
        row = df.loc[idx]
        sweep_dir = int(row['sweep_direction'])
        time_since = int(row['time_since_sweep'])
        entry_price = float(row['close'])
        
        # Locate the sweep bar using lookback
        sweep_idx = idx - time_since
        if sweep_idx < 0:
            # Sequence started before chart data limits
            continue
            
        sweep_row = df.loc[sweep_idx]
        
        # Determine sweep extreme and stop loss
        if sweep_dir == -1: # Bearish Setup
            sweep_extreme = float(sweep_row['high'])
        else: # Bullish Setup
            sweep_extreme = float(sweep_row['low'])
            
        # Risk (R)
        R = abs(entry_price - sweep_extreme)
        if R == 0.0:
            continue
            
        # Stop loss
        if 'suggested_sl' in df.columns and not pd.isna(row['suggested_sl']):
            stop_loss = float(row['suggested_sl'])
        else:
            stop_loss = sweep_extreme
            
        # Target
        if 'suggested_tp' in df.columns and not pd.isna(row['suggested_tp']):
            target = float(row['suggested_tp'])
        else:
            target = entry_price + 2.0 * R * sweep_dir
            
        # Scan forward for outcome starting AFTER the entry bar
        y = 0
        exit_idx = None
        mfe_price = entry_price
        mae_price = entry_price
        
        for t in range(idx + 1, len(df)):
            t_row = df.loc[t]
            t_high = float(t_row['high'])
            t_low = float(t_row['low'])
            
            # Excursion tracking
            if sweep_dir == 1: # Bullish
                mfe_price = max(mfe_price, t_high)
                mae_price = min(mae_price, t_low)
            else: # Bearish
                mfe_price = min(mfe_price, t_low)
                mae_price = max(mae_price, t_high)
                
            # Check for Target vs. Stop Loss
            if sweep_dir == 1:
                hit_target = t_high >= target
                hit_stop = t_low <= stop_loss
                if hit_target and hit_stop:
                    y = 0 # Dual touch is stop-out (conservative)
                    exit_idx = t
                    break
                elif hit_stop:
                    y = 0
                    exit_idx = t
                    break
                elif hit_target:
                    y = 1
                    exit_idx = t
                    break
            else:
                hit_target = t_low <= target
                hit_stop = t_high >= stop_loss
                if hit_target and hit_stop:
                    y = 0
                    exit_idx = t
                    break
                elif hit_stop:
                    y = 0
                    exit_idx = t
                    break
                elif hit_target:
                    y = 1
                    exit_idx = t
                    break
                    
        if exit_idx is None:
            # Reached end of history without hitting target or stop
            exit_idx = len(df) - 1
            
        # Excursions in R-multiples
        if sweep_dir == 1:
            mfe_r = (mfe_price - entry_price) / R
            mae_r = (entry_price - mae_price) / R
        else:
            mfe_r = (entry_price - mfe_price) / R
            mae_r = (mae_price - entry_price) / R
            
        time_to_target = exit_idx - idx
        
        # Calculate Sweep Size (upper/lower wick size)
        sweep_open = float(sweep_row['open'])
        sweep_close = float(sweep_row['close'])
        sweep_high = float(sweep_row['high'])
        sweep_low = float(sweep_row['low'])
        if sweep_dir == -1:
            sweep_size = (sweep_high - max(sweep_open, sweep_close)) / sweep_close
        else:
            sweep_size = (min(sweep_open, sweep_close) - sweep_low) / sweep_close
            
        # Align directional structure features
        bos_strength = float(row['bos_down_strength']) if sweep_dir == -1 else float(row['bos_up_strength'])
        fvg_rejected = float(row['bearish_fvg_rejected']) if sweep_dir == -1 else float(row['bullish_fvg_rejected'])
        
        # Fill missing features with 0
        if pd.isna(bos_strength): bos_strength = 0.0
        if pd.isna(fvg_rejected): fvg_rejected = 0.0
        
        # Compile record
        rec = {
            'time': int(row['time']),
            'liquidity_type': int(row['liquidity_type']),
            'liquidity_strength': float(row['liquidity_strength']),
            'sweep_direction': sweep_dir,
            'sweep_size': sweep_size,
            'bos_strength': bos_strength,
            'fvg_rejected': fvg_rejected,
            'retracement_depth': float(row['retracement_depth']),
            'time_since_sweep': time_since,
            'ny_session': int(row['ny_session']) if not pd.isna(row['ny_session']) else 0,
            'london_session': int(row['london_session']) if not pd.isna(row['london_session']) else 0,
            'asian_session': int(row['asian_session']) if not pd.isna(row['asian_session']) else 0,
            'suggested_tp': target,
            'suggested_sl': stop_loss,
            'max_favorable_excursion': mfe_r,
            'max_adverse_excursion': mae_r,
            'time_to_target': time_to_target,
            'label': y
        }
        
        # Dynamically pass through VIX and custom features
        for col in row.index:
            col_str = str(col).lower()
            if col_str.startswith('vix_') or col_str.startswith('opp_fvg_') or col_str.startswith('same_fvg_'):
                rec[col] = row[col]
                
        processed_records.append(rec)
        
    ml_df = pd.DataFrame(processed_records)
    
    if output_csv:
        ml_df.to_csv(output_csv, index=False)
        print(f"ML dataset exported successfully to: {output_csv} ({len(ml_df)} rows)")
        
    return ml_df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MNQ Trade Setup Dataset Pipeline")
    parser.add_argument('--input', type=str, required=True, help="Path to exported TradingView CSV")
    parser.add_argument('--output', type=str, default="mnq_ml_dataset.csv", help="Path to save processed dataset")
    args = parser.parse_args()
    
    try:
        process_pipeline(args.input, args.output)
    except Exception as e:
        print(f"Pipeline error: {e}")
        exit(1)
