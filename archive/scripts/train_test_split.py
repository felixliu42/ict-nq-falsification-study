import pandas as pd
import numpy as np
import os
import lightgbm as lgb
from datetime import time
from research_harness import calculate_atr, precompute_metrics
from baseline_strategy import clean_columns, simulate_trade_execution, calculate_metrics

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
    
    # Filter for New York Session only
    print("\nFiltering for New York Session setups...")
    df_ny = df_ml[df_ml['ny_session'] == 1.0].copy().reset_index(drop=True)
    n_setups = len(df_ny)
    print(f"Total New York Session setups: {n_setups}")
    
    # Chronological split: 70% Train, 30% Test
    split_idx = int(n_setups * 0.70)
    train_df = df_ny.iloc[:split_idx].copy().reset_index(drop=True)
    test_df = df_ny.iloc[split_idx:].copy().reset_index(drop=True)
    
    print(f"Training Set: {len(train_df)} setups")
    print(f"Out-of-Sample Test Set: {len(test_df)} setups")
    
    features = [
        'liquidity_type', 'liquidity_strength', 'sweep_direction', 'sweep_size',
        'bos_strength', 'fvg_rejected', 'retracement_depth', 'time_since_sweep',
        'ny_session', 'london_session', 'asian_session'
    ]
    
    # Train the LightGBM model once on the mature training set
    print("Training LightGBM model on training set...")
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
    
    model.fit(X_train, y_train)
    
    # Predict probabilities on the out-of-sample test set
    print("Generating out-of-sample predictions on test set...")
    test_df['pred_prob'] = model.predict_proba(test_df[features])[:, 1]
    
    # Run the backtest threshold sweep on the test set
    print("\n" + "="*145)
    print("PURE OUT-OF-SAMPLE TEST RESULTS (70/30 Split, NY Session Only, 16 Contracts/$32 point)")
    print("="*145)
    
    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    modes = ['tp1', 'split']
    multiplier = 32.0
    
    results = []
    
    for mode in modes:
        for th in thresholds:
            trades_df = test_df[test_df['pred_prob'] > th].copy()
            
            dollar_pnls = []
            r_returns = []
            
            for _, row in trades_df.iterrows():
                ret, outcome, duration = simulate_trade_execution(row, df_raw, tp_mode=mode)
                if ret is not None:
                    r_returns.append(ret)
                    R = row['trade_R']
                    if R is not None and not pd.isna(R):
                        dollar_pnls.append(multiplier * R * ret)
                        
            if not dollar_pnls:
                continue
                
            metrics = calculate_metrics(r_returns, [0]*len(r_returns))
            net_profit = sum(dollar_pnls)
            final_balance = 100000.0 + net_profit
            
            # Drawdown
            cum_bal = [100000.0] + list(100000.0 + np.cumsum(dollar_pnls))
            running_max = np.maximum.accumulate(cum_bal)
            drawdowns = running_max - cum_bal
            max_dd = np.max(drawdowns)
            
            results.append({
                'mode': 'TP1 Only' if mode == 'tp1' else 'Split TP1/TP2',
                'threshold': th,
                'trades': len(dollar_pnls),
                'win_rate': metrics['win_rate'],
                'expectancy': metrics['expectancy'],
                'net_profit': net_profit,
                'final_balance': final_balance,
                'max_dd': max_dd,
                'profit_to_dd': net_profit / max_dd if max_dd > 0 else np.inf
            })
            
    # Display table sorted by final balance
    df_results = pd.DataFrame(results).sort_values('final_balance', ascending=False).reset_index(drop=True)
    print(f"| Rank | {'TP Mode':<13} | {'Thresh':<6} | {'Trades':<6} | {'Win Rate':<8} | {'Exp (R)':<8} | {'Net Profit':<12} | {'Final Balance':<14} | {'Max DD ($)':<12} | {'Profit/DD':<9} |")
    print(f"|------|---------------|--------|--------|----------|---------|--------------|---------------|--------------|-----------|")
    for i, r in df_results.iterrows():
        print(f"| {i+1:<4} | {r['mode']:<13} | {r['threshold']:<6.2f} | {r['trades']:<6} | {r['win_rate']:<8.1%} | {r['expectancy']:<8.2f} | ${r['net_profit']:<11,.2f} | ${r['final_balance']:<12,.2f} | ${r['max_dd']:<11,.2f} | {r['profit_to_dd']:<9.2f} |")

if __name__ == '__main__':
    main()
