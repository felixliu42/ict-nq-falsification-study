import pandas as pd
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt

try:
    import lightgbm as lgb
except ImportError as e:
    print(f"Error importing LightGBM: {e}")
    print("Please install LightGBM using: pip install lightgbm")
    exit(1)

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
        'ny_session', 'london_session', 'asian_session'
    ]
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

def calculate_choppiness_index(df_raw, length=30):
    """
    Calculate the Choppiness Index on raw bar-by-bar data.
    """
    df = df_raw.copy()
    high = df['high']
    low = df['low']
    close = df['close']
    close_prev = close.shift(1)
    
    # True Range
    tr1 = high - low
    tr2 = (high - close_prev).abs()
    tr3 = (low - close_prev).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    tr_sum = tr.rolling(window=length).sum()
    max_high = high.rolling(window=length).max()
    min_low = low.rolling(window=length).min()
    
    diff = max_high - min_low
    diff = diff.replace(0, 1e-8) # avoid division by zero or log10(0)
    
    chop = 100.0 * np.log10(tr_sum / diff) / np.log10(length)
    return chop

def run_fold_walk_forward(df_ml, min_train_size=30, num_folds=3):
    """
    Perform walk-forward validation split into F chronological folds.
    Returns:
        df_oos: out-of-sample dataframe with 'pred_prob' column
        fold_importances: dict mapping fold index to feature importance Series
    """
    print(f"Running Fold-Based Walk-Forward Validation (W={min_train_size}, Folds={num_folds})...")
    
    features = [
        'liquidity_type', 'liquidity_strength', 'sweep_direction', 'sweep_size',
        'bos_strength', 'fvg_rejected', 'retracement_depth', 'time_since_sweep',
        'ny_session', 'london_session', 'asian_session'
    ]
    
    df_ml = df_ml.sort_values('time').reset_index(drop=True)
    df_ml['pred_prob'] = np.nan
    
    n_rows = len(df_ml)
    oos_size = n_rows - min_train_size
    if oos_size <= 0:
        raise ValueError("Dataset too small for walk-forward validation.")
        
    fold_size = oos_size // num_folds
    fold_importances = {}
    
    for f in range(num_folds):
        train_end = min_train_size + f * fold_size
        test_end = min_train_size + (f + 1) * fold_size if f < num_folds - 1 else n_rows
        
        train_df = df_ml.iloc[:train_end]
        test_df = df_ml.iloc[train_end:test_end]
        
        # Train LightGBM model
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
        
        # Record feature importances for this fold
        fold_importances[f"Fold {f+1}"] = pd.Series(model.feature_importances_, index=features)
        
        # Predict out-of-sample probabilities
        probs = model.predict_proba(X_test)[:, 1]
        df_ml.loc[train_end:test_end-1, 'pred_prob'] = probs
        
    df_oos = df_ml.iloc[min_train_size:].copy().reset_index(drop=True)
    return df_oos, fold_importances

def simulate_trade_execution(row, df_raw, tp_mode='tp1'):
    """
    Simulate trade execution for a single setup row against the raw high/low price series.
    Returns:
        r_multiple_return: float
    """
    entry_time = int(row['time'])
    sweep_dir = int(row['sweep_direction'])
    time_since = int(row['time_since_sweep'])
    
    # Find setup bar index in raw data
    raw_indices = df_raw.index[df_raw['time'] == entry_time].tolist()
    if not raw_indices:
        time_diffs = np.abs(df_raw['time'] - entry_time)
        idx = int(np.argmin(time_diffs))
        if time_diffs[idx] > 3600000:
            return None
    else:
        idx = raw_indices[0]
        
    entry_price = float(df_raw.loc[idx, 'close'])
    sweep_idx = idx - time_since
    if sweep_idx < 0:
        return None
        
    if sweep_dir == -1: # Bearish Setup
        sweep_extreme = float(df_raw.loc[sweep_idx, 'high'])
    else: # Bullish Setup
        sweep_extreme = float(df_raw.loc[sweep_idx, 'low'])
        
    R = abs(entry_price - sweep_extreme)
    if R == 0.0:
        return None
        
    tp1 = entry_price + 2.0 * R * sweep_dir
    tp2 = entry_price + 4.0 * R * sweep_dir
    stop_loss = sweep_extreme
    
    has_hit_tp1 = False
    trail_stop = stop_loss
    
    for t in range(idx + 1, len(df_raw)):
        high = float(df_raw.loc[t, 'high'])
        low = float(df_raw.loc[t, 'low'])
        
        if tp_mode == 'tp1':
            if sweep_dir == 1:
                hit_stop = low <= stop_loss
                hit_tp1 = high >= tp1
                if hit_tp1 and hit_stop: return -1.0
                elif hit_stop: return -1.0
                elif hit_tp1: return 2.0
            else:
                hit_stop = high >= stop_loss
                hit_tp1 = low <= tp1
                if hit_tp1 and hit_stop: return -1.0
                elif hit_stop: return -1.0
                elif hit_tp1: return 2.0
                    
        elif tp_mode == 'split':
            if not has_hit_tp1:
                if sweep_dir == 1:
                    hit_stop = low <= stop_loss
                    hit_tp1 = high >= tp1
                    if hit_tp1 and hit_stop: return -1.0
                    elif hit_stop: return -1.0
                    elif hit_tp1:
                        has_hit_tp1 = True
                        trail_stop = entry_price
                        if low <= trail_stop: return 1.0
                else:
                    hit_stop = high >= stop_loss
                    hit_tp1 = low <= tp1
                    if hit_tp1 and hit_stop: return -1.0
                    elif hit_stop: return -1.0
                    elif hit_tp1:
                        has_hit_tp1 = True
                        trail_stop = entry_price
                        if high >= trail_stop: return 1.0
            else:
                if sweep_dir == 1:
                    hit_trail = low <= trail_stop
                    hit_tp2 = high >= tp2
                    if hit_tp2 and hit_trail: return 1.0
                    elif hit_trail: return 1.0
                    elif hit_tp2: return 3.0
                else:
                    hit_trail = high >= trail_stop
                    hit_tp2 = low <= tp2
                    if hit_tp2 and hit_trail: return 1.0
                    elif hit_trail: return 1.0
                    elif hit_tp2: return 3.0
                        
    # End of history fallback
    final_close = float(df_raw.iloc[-1]['close'])
    final_return = ((final_close - entry_price) / R) * sweep_dir
    
    if tp_mode == 'tp1':
        return max(-1.0, min(2.0, final_return))
    else:
        if has_hit_tp1:
            second_half_r = max(0.0, min(2.0, final_return * 0.5))
            return 1.0 + second_half_r
        else:
            return max(-1.0, min(1.0, final_return))

def calculate_metrics(returns):
    """
    Compute metrics for a returns series.
    """
    total_trades = len(returns)
    if total_trades == 0:
        return {'trades': 0, 'win_rate': 0.0, 'pf': 0.0, 'expectancy': 0.0}
    win_rate = len([r for r in returns if r > 0]) / total_trades
    gains = sum([r for r in returns if r > 0])
    losses = sum([abs(r) for r in returns if r < 0])
    pf = gains / losses if losses > 0 else (gains if gains > 0 else 1.0)
    expectancy = sum(returns) / total_trades
    return {'trades': total_trades, 'win_rate': win_rate, 'pf': pf, 'expectancy': expectancy}

def main():
    parser = argparse.ArgumentParser(description="LightGBM System Optimization Framework")
    parser.add_argument('--dataset', type=str, default="demo_ml_dataset.csv", help="ML dataset path")
    parser.add_argument('--raw', type=str, default="demo_tv_export.csv", help="Raw price series path")
    parser.add_argument('--min-train', type=int, default=30, help="Initial training size")
    parser.add_argument('--folds', type=int, default=3, help="Chronological folds for feature stability")
    parser.add_argument('--chop-lookback', type=int, default=30, help="Choppiness Index window")
    parser.add_argument('--outdir', type=str, default=".", help="Output directory")
    args = parser.parse_args()
    
    if not os.path.exists(args.dataset):
        raise FileNotFoundError(f"ML dataset not found: {args.dataset}")
    if not os.path.exists(args.raw):
        raise FileNotFoundError(f"Raw TradingView CSV not found: {args.raw}")
        
    print(f"Loading raw price series: {args.raw}")
    df_raw = pd.read_csv(args.raw)
    df_raw = clean_columns(df_raw)
    
    # Calculate Choppiness Index on raw data
    print(f"Calculating Choppiness Index (lookback={args.chop_lookback})...")
    df_raw['chop'] = calculate_choppiness_index(df_raw, length=args.chop_lookback)
    
    print(f"Loading ML dataset: {args.dataset}")
    df_ml = pd.read_csv(args.dataset)
    df_ml['time'] = pd.to_numeric(df_ml['time'], errors='coerce')
    df_raw['time'] = pd.to_numeric(df_raw['time'], errors='coerce')
    
    # Map chop value to ML dataset setups
    chop_map = dict(zip(df_raw['time'], df_raw['chop']))
    df_ml['chop'] = df_ml['time'].map(chop_map)
    df_ml = df_ml.dropna(subset=['chop']).reset_index(drop=True)
    
    # Determine Trend vs Chop regime based on median
    median_chop = df_ml['chop'].median()
    df_ml['regime'] = np.where(df_ml['chop'] < median_chop, 'Trend', 'Chop')
    print(f"Median Choppiness: {median_chop:.2f}")
    print(f"Regime breakdown: Trend={len(df_ml[df_ml['regime']=='Trend'])}, Chop={len(df_ml[df_ml['regime']=='Chop'])}")
    
    # Walk-Forward Validation
    df_oos, fold_imps = run_fold_walk_forward(df_ml, min_train_size=args.min_train, num_folds=args.folds)
    
    # Perform sweeps
    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    modes = ['tp1', 'split']
    
    optimization_data = []
    equity_curves = {'tp1': {}, 'split': {}}
    
    for mode in modes:
        equity_curves[mode] = {'Trend': {}, 'Chop': {}, 'Combined': {}}
        for th in thresholds:
            # Predict filter
            trades_df = df_oos[df_oos['pred_prob'] > th].copy()
            
            # Simulate
            trend_returns = []
            chop_returns = []
            combined_returns = []
            
            for _, row in trades_df.iterrows():
                ret = simulate_trade_execution(row, df_raw, tp_mode=mode)
                if ret is not None:
                    combined_returns.append(ret)
                    if row['regime'] == 'Trend':
                        trend_returns.append(ret)
                    else:
                        chop_returns.append(ret)
                        
            # Metrics
            trend_metrics = calculate_metrics(trend_returns)
            chop_metrics = calculate_metrics(chop_returns)
            combined_metrics = calculate_metrics(combined_returns)
            
            optimization_data.append({
                'tp_mode': mode,
                'threshold': th,
                'trend_trades': trend_metrics['trades'],
                'trend_wr': trend_metrics['win_rate'],
                'trend_exp': trend_metrics['expectancy'],
                'chop_trades': chop_metrics['trades'],
                'chop_wr': chop_metrics['win_rate'],
                'chop_exp': chop_metrics['expectancy'],
                'comb_trades': combined_metrics['trades'],
                'comb_wr': combined_metrics['win_rate'],
                'comb_exp': combined_metrics['expectancy']
            })
            
            # Save equity curves
            equity_curves[mode]['Trend'][th] = [0] + list(np.cumsum(trend_returns))
            equity_curves[mode]['Chop'][th] = [0] + list(np.cumsum(chop_returns))
            equity_curves[mode]['Combined'][th] = [0] + list(np.cumsum(combined_returns))
            
    # 1. Print Regime Optimization Table
    print("\n" + "="*95)
    print("REGIME OPTIMIZATION SUMMARY (TREND VS. CHOP)")
    print("="*95)
    print(f"| {'Threshold':<9} | {'TP Mode':<10} | {'Trend Trades':<12} | {'Trend Exp (R)':<13} | {'Chop Trades':<11} | {'Chop Exp (R)':<12} | {'Comb Exp (R)':<12} |")
    print(f"|{'-'*11}|{'-'*12}|{'-'*14}|{'-'*15}|{'-'*13}|{'-'*14}|{'-'*14}|")
    for r in optimization_data:
        m_str = 'TP1' if r['tp_mode'] == 'tp1' else 'Split'
        print(f"| {r['threshold']:<9.2f} | {m_str:<10} | {r['trend_trades']:<12} | {r['trend_exp']:<13.2f} | {r['chop_trades']:<11} | {r['chop_exp']:<12.2f} | {r['comb_exp']:<12.2f} |")
    print("="*95)
    
    # 2. Print Feature Stability Analysis
    print("\n" + "="*80)
    print("FEATURE IMPORTANCE STABILITY ACROSS WALK-FORWARD FOLDS")
    print("="*80)
    
    imp_df = pd.DataFrame(fold_imps)
    # Coefficient of Variation = Standard Deviation / Mean
    imp_df['Mean'] = imp_df.mean(axis=1)
    imp_df['StdDev'] = imp_df.std(axis=1)
    imp_df['CV'] = imp_df['StdDev'] / (imp_df['Mean'] + 1e-8)
    
    # Define stability status
    def get_status(cv):
        if cv < 0.40: return "High Stability"
        elif cv < 0.80: return "Moderate Stability"
        else: return "Unstable (Overfitting Risk)"
        
    imp_df['Status'] = imp_df['CV'].apply(get_status)
    imp_df_sorted = imp_df.sort_values('CV')
    
    print(f"| {'Feature Name':<20} | " + " | ".join([f"{k:<8}" for k in fold_imps.keys()]) + " | {'Mean':<6} | {'CV':<6} | {'Status':<25} |")
    print(f"|{'-'*22}|" + "|".join([f"{'-'*10}" for _ in fold_imps.keys()]) + f"|{'-'*8}|{'-'*8}|{'-'*27}|")
    for feat, row in imp_df_sorted.iterrows():
        fold_vals = " | ".join([f"{row[k]:<8.1f}" for k in fold_imps.keys()])
        print(f"| {feat:<20} | {fold_vals} | {row['Mean']:<6.1f} | {row['CV']:<6.2f} | {row['Status']:<25} |")
    print("="*80)
    
    # Find Best Configuration (highest combined expectancy)
    best_config = max(optimization_data, key=lambda x: x['comb_exp'])
    best_th = best_config['threshold']
    best_mode = best_config['tp_mode']
    print(f"\nOptimal Configuration Found: Threshold={best_th:.2f}, TP Mode={best_mode.upper()} (Expectancy: {best_config['comb_exp']:.2f}R)")
    
    # 3. Print Per-Session Regime Breakdown for Optimal Config
    print("\n" + "="*80)
    print(f"SESSION SENSITIVITY BREAKDOWN FOR OPTIMAL SYSTEM (Th={best_th:.2f}, {best_mode.upper()})")
    print("="*80)
    print(f"| {'Session':<12} | {'Regime':<8} | {'Trades':<8} | {'Win Rate':<10} | {'Expectancy (R)':<14} |")
    print(f"|{'-'*14}|{'-'*10}|{'-'*10}|{'-'*12}|{'-'*16}|")
    
    trades_opt = df_oos[df_oos['pred_prob'] > best_th].copy()
    session_regime_returns = {}
    for sess in ['ny_session', 'london_session', 'asian_session']:
        sess_name = sess.split('_')[0].upper()
        session_regime_returns[sess_name] = {'Trend': [], 'Chop': []}
        
    for _, row in trades_opt.iterrows():
        ret = simulate_trade_execution(row, df_raw, tp_mode=best_mode)
        if ret is not None:
            reg = row['regime']
            if row['ny_session'] == 1:
                session_regime_returns['NY'][reg].append(ret)
            if row['london_session'] == 1:
                session_regime_returns['LONDON'][reg].append(ret)
            if row['asian_session'] == 1:
                session_regime_returns['ASIAN'][reg].append(ret)
                
    for sess, regimes in session_regime_returns.items():
        for reg, rets in regimes.items():
            if len(rets) > 0:
                metrics = calculate_metrics(rets)
                print(f"| {sess:<12} | {reg:<8} | {metrics['trades']:<8} | {metrics['win_rate']:<10.1%} | {metrics['expectancy']:<14.2f} |")
    print("="*80)
    
    # Generate Multi-panel Plot
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Threshold vs Expectancy (TP1)
    plt.subplot(2, 2, 1)
    tp1_data = [d for d in optimization_data if d['tp_mode'] == 'tp1']
    th_vals = [d['threshold'] for d in tp1_data]
    plt.plot(th_vals, [d['trend_exp'] for d in tp1_data], marker='o', label='Trend Regime', color='tab:green', linewidth=2)
    plt.plot(th_vals, [d['chop_exp'] for d in tp1_data], marker='x', label='Chop Regime', color='tab:red', linewidth=2)
    plt.plot(th_vals, [d['comb_exp'] for d in tp1_data], marker='s', label='Combined', color='tab:blue', linewidth=2.5, linestyle='--')
    plt.title("Threshold Sweep: TP1 Only (+2R)")
    plt.xlabel("Probability Threshold")
    plt.ylabel("Expectancy (R)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot 2: Threshold vs Expectancy (Split TP)
    plt.subplot(2, 2, 2)
    split_data = [d for d in optimization_data if d['tp_mode'] == 'split']
    plt.plot(th_vals, [d['trend_exp'] for d in split_data], marker='o', label='Trend Regime', color='tab:green', linewidth=2)
    plt.plot(th_vals, [d['chop_exp'] for d in split_data], marker='x', label='Chop Regime', color='tab:red', linewidth=2)
    plt.plot(th_vals, [d['comb_exp'] for d in split_data], marker='s', label='Combined', color='tab:blue', linewidth=2.5, linestyle='--')
    plt.title("Threshold Sweep: Split TP1/TP2 (+2R/+4R)")
    plt.xlabel("Probability Threshold")
    plt.ylabel("Expectancy (R)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot 3: Feature Importance Stability Across Folds
    plt.subplot(2, 2, 3)
    # Plot top 6 features for layout legibility
    top_features = imp_df_sorted.index[:6]
    for feat in top_features:
        fold_labels = [k for k in fold_imps.keys()]
        fold_scores = [imp_df.loc[feat, k] for k in fold_labels]
        plt.plot(fold_labels, fold_scores, marker='o', label=f"{feat} (CV: {imp_df.loc[feat, 'CV']:.2f})", linewidth=2)
    plt.title("Feature Importance Stability Across Time (Top 6)")
    plt.xlabel("Chronological Folds")
    plt.ylabel("Importance (Split Count)")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8, loc='upper left')
    
    # Plot 4: Optimal Config Equity Curve Stability
    plt.subplot(2, 2, 4)
    y_trend = equity_curves[best_mode]['Trend'][best_th]
    y_chop = equity_curves[best_mode]['Chop'][best_th]
    y_comb = equity_curves[best_mode]['Combined'][best_th]
    
    plt.plot(range(len(y_trend)), y_trend, label=f"Trend Regime (Trades: {len(y_trend)-1})", color='tab:green', linewidth=2)
    plt.plot(range(len(y_chop)), y_chop, label=f"Chop Regime (Trades: {len(y_chop)-1})", color='tab:red', linewidth=2)
    plt.plot(range(len(y_comb)), y_comb, label=f"Combined System (Trades: {len(y_comb)-1})", color='tab:blue', linewidth=2.5, linestyle='--')
    plt.title(f"Equity Curves: Threshold > {best_th:.2f} ({best_mode.upper()})")
    plt.xlabel("Trade Count")
    plt.ylabel("Cumulative Returns (R)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plot_path = os.path.join(args.outdir, "optimization_results.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\nMulti-panel optimization plots saved successfully to: {plot_path}")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"Optimization error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
