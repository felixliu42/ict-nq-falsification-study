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
        'ny_session', 'london_session', 'asian_session',
        'suggested_tp', 'suggested_sl'
    ]
    # Sort standard_cols by length descending to match longer strings first
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

def run_walk_forward_validation(df_ml, min_train_size=30, step_size=5):
    """
    Run an expanding window walk-forward validation to generate out-of-sample predicted probabilities.
    """
    print(f"Running Walk-Forward Validation (W={min_train_size}, N={step_size})...")
    
    features = [
        'liquidity_type', 'liquidity_strength', 'sweep_direction', 'sweep_size',
        'bos_strength', 'fvg_rejected', 'retracement_depth', 'time_since_sweep',
        'ny_session', 'london_session', 'asian_session'
    ]
    
    # Ensure dataframe is sorted by time
    df_ml = df_ml.sort_values('time').reset_index(drop=True)
    df_ml['pred_prob'] = np.nan
    
    n_rows = len(df_ml)
    if n_rows <= min_train_size:
        raise ValueError(f"Dataset has {n_rows} rows, which is <= min_train_size ({min_train_size}). Cannot perform walk-forward validation.")
        
    for t in range(min_train_size, n_rows, step_size):
        train_df = df_ml.iloc[:t]
        test_end = min(t + step_size, n_rows)
        test_df = df_ml.iloc[t:test_end]
        
        # Define conservative LightGBM model to prevent overfitting
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
        
        # Predict out-of-sample probabilities
        probs = model.predict_proba(X_test)[:, 1]
        df_ml.loc[t:test_end-1, 'pred_prob'] = probs
        
    # Drop rows that were in the initial training window (they have no OOS prediction)
    df_oos = df_ml.iloc[min_train_size:].copy().reset_index(drop=True)
    print(f"Generated {len(df_oos)} out-of-sample predictions.")
    return df_oos

def simulate_trade_execution(row, df_raw, tp_mode='tp1'):
    """
    Simulate trade execution for a single setup row against the raw high/low price series.
    Returns:
        r_multiple_return: float
        hit_type: str ('tp1', 'tp2', 'stop', 'trail_stop', 'end_of_data')
        duration: int (number of bars from entry to exit)
    """
    entry_time = int(row['time'])
    sweep_dir = int(row['sweep_direction'])
    time_since = int(row['time_since_sweep'])
    
    # Find setup bar index in raw data
    raw_indices = df_raw.index[df_raw['time'] == entry_time].tolist()
    if not raw_indices:
        # Fallback to absolute closest time if exact timestamp matches are missing
        time_diffs = np.abs(df_raw['time'] - entry_time)
        idx = int(np.argmin(time_diffs))
        if time_diffs[idx] > 3600000: # Max 1 hour difference
            return None, 'missing_data', None
    else:
        idx = raw_indices[0]
        
    entry_price = float(df_raw.loc[idx, 'close'])
    sweep_idx = idx - time_since
    if sweep_idx < 0:
        return None, 'before_history_limit', None
        
    # Get sweep extreme price (acts as Stop Loss)
    if sweep_dir == -1: # Bearish Setup
        sweep_extreme = float(df_raw.loc[sweep_idx, 'high'])
    else: # Bullish Setup
        sweep_extreme = float(df_raw.loc[sweep_idx, 'low'])
        
    R = abs(entry_price - sweep_extreme)
    if R == 0.0:
        return None, 'zero_risk', None
        
    # Retrieve stop_loss from row['suggested_sl'] if available, otherwise fall back to sweep_extreme
    if 'suggested_sl' in row and not pd.isna(row['suggested_sl']):
        stop_loss = float(row['suggested_sl'])
    else:
        stop_loss = sweep_extreme
        
    # Retrieve tp1 from row['suggested_tp'] if available, otherwise fall back to entry_price + 2.0 * R * sweep_dir
    if 'suggested_tp' in row and not pd.isna(row['suggested_tp']):
        tp1 = float(row['suggested_tp'])
    else:
        tp1 = entry_price + 2.0 * R * sweep_dir
        
    # Calculate tp2 = entry_price + 2.0 * reward * sweep_dir (where reward = abs(tp1 - entry_price)) for split mode
    reward = abs(tp1 - entry_price)
    tp2 = entry_price + 2.0 * reward * sweep_dir
    
    # Generalize returns in R-multiples using the actual R:R of the trade instead of hardcoded '2.0' and '3.0' constants
    actual_rr = abs(tp1 - entry_price) / R
    stop_r = -abs(stop_loss - entry_price) / R
    
    # Scan forward bar-by-bar starting after the entry bar
    has_hit_tp1 = False
    trail_stop = stop_loss
    
    for t in range(idx + 1, len(df_raw)):
        high = float(df_raw.loc[t, 'high'])
        low = float(df_raw.loc[t, 'low'])
        close = float(df_raw.loc[t, 'close'])
        
        duration = t - idx
        
        if tp_mode == 'tp1':
            if sweep_dir == 1: # Bullish
                hit_stop = low <= stop_loss
                hit_tp1 = high >= tp1
                if hit_tp1 and hit_stop:
                    return stop_r, 'stop', duration # Conservative double touch
                elif hit_stop:
                    return stop_r, 'stop', duration
                elif hit_tp1:
                    return actual_rr, 'tp1', duration
            else: # Bearish
                hit_stop = high >= stop_loss
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
                    hit_stop = low <= stop_loss
                    hit_tp1 = high >= tp1
                    if hit_tp1 and hit_stop:
                        return stop_r, 'stop', duration
                    elif hit_stop:
                        return stop_r, 'stop', duration
                    elif hit_tp1:
                        has_hit_tp1 = True
                        trail_stop = entry_price # Trail to break-even
                        # Check if trailed stop was hit on same bar
                        if low <= trail_stop:
                            return 0.5 * actual_rr, 'trail_stop', duration # 50% exited at TP1, 50% at BE
                else: # Bearish
                    hit_stop = high >= stop_loss
                    hit_tp1 = low <= tp1
                    if hit_tp1 and hit_stop:
                        return stop_r, 'stop', duration
                    elif hit_stop:
                        return stop_r, 'stop', duration
                    elif hit_tp1:
                        has_hit_tp1 = True
                        trail_stop = entry_price # Trail to break-even
                        if high >= trail_stop:
                            return 0.5 * actual_rr, 'trail_stop', duration
            else:
                # Looking for TP2 or Trail Stop
                if sweep_dir == 1: # Bullish
                    hit_trail = low <= trail_stop
                    hit_tp2 = high >= tp2
                    if hit_tp2 and hit_trail:
                        return 0.5 * actual_rr, 'trail_stop', duration # Conservative trail hit
                    elif hit_trail:
                        return 0.5 * actual_rr, 'trail_stop', duration
                    elif hit_tp2:
                        return 1.5 * actual_rr, 'tp2', duration # 50% at TP1, 50% at TP2
                else: # Bearish
                    hit_trail = high >= trail_stop
                    hit_tp2 = low <= tp2
                    if hit_tp2 and hit_trail:
                        return 0.5 * actual_rr, 'trail_stop', duration
                    elif hit_trail:
                        return 0.5 * actual_rr, 'trail_stop', duration
                    elif hit_tp2:
                        return 1.5 * actual_rr, 'tp2', duration
                        
    # End of chart history reached
    final_close = float(df_raw.iloc[-1]['close'])
    final_return = ((final_close - entry_price) / R) * sweep_dir
    duration = len(df_raw) - 1 - idx
    
    if tp_mode == 'tp1':
        # Bound return between stop_r and actual_rr
        return max(stop_r, min(actual_rr, final_return)), 'end_of_data', duration
    else:
        if has_hit_tp1:
            # First half at TP1 (worth 0.5 * actual_rr total), second half at final close (bounded by trail stop 0R and TP2 (worth actual_rr))
            second_half_r = max(0.0, min(actual_rr, final_return * 0.5))
            return 0.5 * actual_rr + second_half_r, 'end_of_data', duration
        else:
            return max(stop_r, min(0.5 * actual_rr, final_return)), 'end_of_data', duration

def get_trade_R(row, df_raw):
    """
    Retrieve or calculate the trade risk (R) in price units.
    """
    entry_time = int(row['time'])
    sweep_dir = int(row['sweep_direction'])
    time_since = int(row['time_since_sweep'])
    
    # Find setup bar index in raw data
    raw_indices = df_raw.index[df_raw['time'] == entry_time].tolist()
    if not raw_indices:
        # Fallback to absolute closest time if exact timestamp matches are missing
        time_diffs = np.abs(df_raw['time'] - entry_time)
        idx = int(np.argmin(time_diffs))
        if time_diffs[idx] > 3600000: # Max 1 hour difference
            return None
    else:
        idx = raw_indices[0]
        
    entry_price = float(df_raw.loc[idx, 'close'])
    sweep_idx = idx - time_since
    if sweep_idx < 0:
        return None
        
    # Get sweep extreme price (acts as Stop Loss)
    if sweep_dir == -1: # Bearish Setup
        sweep_extreme = float(df_raw.loc[sweep_idx, 'high'])
    else: # Bullish Setup
        sweep_extreme = float(df_raw.loc[sweep_idx, 'low'])
        
    return abs(entry_price - sweep_extreme)

def calculate_metrics(returns, durations):
    """
    Calculate performance metrics on a series of trade returns and durations.
    """
    total_trades = len(returns)
    if total_trades == 0:
        return {
            'total_trades': 0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'expectancy': 0.0,
            'max_dd': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'win_loss_ratio': 0.0,
            'avg_duration': 0.0
        }
        
    win_rate = len([r for r in returns if r > 0]) / total_trades
    gains = sum([r for r in returns if r > 0])
    losses = sum([abs(r) for r in returns if r < 0])
    profit_factor = gains / losses if losses > 0 else (gains if gains > 0 else 1.0)
    expectancy = sum(returns) / total_trades
    
    # Calculate Max Drawdown in R-multiples
    cum_returns = [0] + list(np.cumsum(returns))
    running_max = np.maximum.accumulate(cum_returns)
    drawdowns = running_max - cum_returns
    max_dd = np.max(drawdowns)
    
    # Calculate extra metrics
    pos_returns = [r for r in returns if r > 0]
    neg_returns = [abs(r) for r in returns if r < 0]
    
    avg_win = np.mean(pos_returns) if len(pos_returns) > 0 else 0.0
    avg_loss = np.mean(neg_returns) if len(neg_returns) > 0 else 0.0
    
    win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else (np.inf if avg_win > 0 else 0.0)
    avg_duration = np.mean(durations) if len(durations) > 0 else 0.0
    
    return {
        'total_trades': total_trades,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'expectancy': expectancy,
        'max_dd': max_dd,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'win_loss_ratio': win_loss_ratio,
        'avg_duration': avg_duration
    }

def run_backtest(df_oos, df_raw, output_dir=".", no_ml=False):
    """
    Run the backtest threshold sweep and per-session breakdown.
    """
    if no_ml:
        thresholds = [0.0]
    else:
        thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
        
    modes = ['tp1', 'split']
    
    results = []
    equity_curves = {}
    balance_curves = {}
    
    for mode in modes:
        equity_curves[mode] = {}
        for th in thresholds:
            # Filter setups
            if no_ml:
                trades_df = df_oos.copy()
            else:
                trades_df = df_oos[df_oos['pred_prob'] > th].copy()
            
            trade_returns = []
            trade_durations = []
            session_data = []
            dollar_pnls = []
            
            for _, row in trades_df.iterrows():
                ret, outcome, duration = simulate_trade_execution(row, df_raw, tp_mode=mode)
                if ret is not None:
                    trade_returns.append(ret)
                    trade_durations.append(duration)
                    
                    if no_ml:
                        R = get_trade_R(row, df_raw)
                        if R is not None:
                            dollar_pnls.append(16.0 * R * ret)
                            
                    session_data.append({
                        'return': ret,
                        'duration': duration,
                        'ny_session': row['ny_session'],
                        'london_session': row['london_session'],
                        'asian_session': row['asian_session']
                    })
                    
            metrics = calculate_metrics(trade_returns, trade_durations)
            metrics['threshold'] = th
            metrics['tp_mode'] = 'TP1 Only (+2R)' if mode == 'tp1' else 'Split TP1/TP2 (+2R/+4R)'
            results.append(metrics)
            
            # Save equity curve (cumulative return over time)
            equity_curves[mode][th] = [0] + list(np.cumsum(trade_returns))
            
            if no_ml:
                balance_curves[mode] = [100000.0] + list(100000.0 + np.cumsum(dollar_pnls))
            
    # Print Consolidated Results Table
    if no_ml:
        print("\n" + "="*135)
        print("CONSOLIDATED BACKTEST RESULTS (RULES-BASED MODE)")
        print("="*135)
        print(f"| {'TP Mode':<25} | {'Total Trades':<12} | {'Win Rate':<8} | {'Profit Factor':<13} | {'Expectancy (R)':<14} | {'Max DD (R)':<10} | {'Avg Win (R)':<11} | {'Avg Loss (R)':<12} | {'W/L Ratio':<9} | {'Avg Duration':<12} |")
        print(f"|{'-'*27}|{'-'*14}|{'-'*10}|{'-'*15}|{'-'*16}|{'-'*12}|{'-'*13}|{'-'*14}|{'-'*11}|{'-'*14}|")
        for r in results:
            pf_str = f"{r['profit_factor']:.2f}" if r['profit_factor'] != np.inf else "Inf"
            wl_str = f"{r['win_loss_ratio']:.2f}" if r['win_loss_ratio'] != np.inf else "Inf"
            if r['total_trades'] == 0:
                pf_str = "N/A"
                wl_str = "N/A"
            print(f"| {r['tp_mode']:<25} | {r['total_trades']:<12} | {r['win_rate']:<8.1%} | {pf_str:<13} | {r['expectancy']:<14.2f} | {r['max_dd']:<10.2f} | {r['avg_win']:<11.2f} | {r['avg_loss']:<12.2f} | {wl_str:<9} | {r['avg_duration']:<12.1f} |")
        print("="*135)
    else:
        print("\n" + "="*145)
        print("CONSOLIDATED BACKTEST RESULTS (THRESHOLD SWEEP)")
        print("="*145)
        print(f"| {'Threshold':<9} | {'TP Mode':<25} | {'Total Trades':<12} | {'Win Rate':<8} | {'Profit Factor':<13} | {'Expectancy (R)':<14} | {'Max DD (R)':<10} | {'Avg Win (R)':<11} | {'Avg Loss (R)':<12} | {'W/L Ratio':<9} | {'Avg Duration':<12} |")
        print(f"|{'-'*11}|{'-'*27}|{'-'*14}|{'-'*10}|{'-'*15}|{'-'*16}|{'-'*12}|{'-'*13}|{'-'*14}|{'-'*11}|{'-'*14}|")
        for r in results:
            pf_str = f"{r['profit_factor']:.2f}" if r['profit_factor'] != np.inf else "Inf"
            wl_str = f"{r['win_loss_ratio']:.2f}" if r['win_loss_ratio'] != np.inf else "Inf"
            if r['total_trades'] == 0:
                pf_str = "N/A"
                wl_str = "N/A"
            print(f"| {r['threshold']:<9.2f} | {r['tp_mode']:<25} | {r['total_trades']:<12} | {r['win_rate']:<8.1%} | {pf_str:<13} | {r['expectancy']:<14.2f} | {r['max_dd']:<10.2f} | {r['avg_win']:<11.2f} | {r['avg_loss']:<12.2f} | {wl_str:<9} | {r['avg_duration']:<12.1f} |")
        print("="*145)
    
    # Print Per-Session Breakdown
    if no_ml:
        print("\n" + "="*110)
        print("PER-SESSION BREAKDOWN (RULES-BASED MODE)")
        print("="*110)
        print(f"| {'TP Mode':<10} | {'Session':<12} | {'Trades':<6} | {'Win Rate':<8} | {'Expectancy (R)':<14} | {'Avg Win (R)':<11} | {'Avg Loss (R)':<12} | {'W/L Ratio':<9} | {'Avg Duration':<12} |")
        print(f"|{'-'*12}|{'-'*14}|{'-'*8}|{'-'*10}|{'-'*16}|{'-'*13}|{'-'*14}|{'-'*11}|{'-'*14}|")
        
        for mode in modes:
            mode_str = 'TP1' if mode == 'tp1' else 'Split'
            trades_df = df_oos.copy()
            
            # Simulate trades and tag session
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
                    print(f"| {mode_str:<10} | {sess:<12} | {sess_metrics['total_trades']:<6} | {sess_metrics['win_rate']:<8.1%} | {sess_metrics['expectancy']:<14.2f} | {sess_metrics['avg_win']:<11.2f} | {sess_metrics['avg_loss']:<12.2f} | {wl_str:<9} | {sess_metrics['avg_duration']:<12.1f} |")
        print("="*110)
    else:
        print("\n" + "="*120)
        print("PER-SESSION BREAKDOWN")
        print("="*120)
        print(f"| {'Threshold':<9} | {'TP Mode':<10} | {'Session':<12} | {'Trades':<6} | {'Win Rate':<8} | {'Expectancy (R)':<14} | {'Avg Win (R)':<11} | {'Avg Loss (R)':<12} | {'W/L Ratio':<9} | {'Avg Duration':<12} |")
        print(f"|{'-'*11}|{'-'*12}|{'-'*14}|{'-'*8}|{'-'*10}|{'-'*16}|{'-'*13}|{'-'*14}|{'-'*11}|{'-'*14}|")
        
        for mode in modes:
            mode_str = 'TP1' if mode == 'tp1' else 'Split'
            for th in thresholds:
                trades_df = df_oos[df_oos['pred_prob'] > th].copy()
                
                # Simulate trades and tag session
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
                        print(f"| {th:<9.2f} | {mode_str:<10} | {sess:<12} | {sess_metrics['total_trades']:<6} | {sess_metrics['win_rate']:<8.1%} | {sess_metrics['expectancy']:<14.2f} | {sess_metrics['avg_win']:<11.2f} | {sess_metrics['avg_loss']:<12.2f} | {wl_str:<9} | {sess_metrics['avg_duration']:<12.1f} |")
        print("="*120)
        
    # Plot Cumulative Equity Curves
    if no_ml:
        plt.figure(figsize=(12, 5))
        
        # Subplot 1: TP1 Only
        plt.subplot(1, 2, 1)
        y_tp1 = equity_curves['tp1'][0.0]
        plt.plot(range(len(y_tp1)), y_tp1, color='blue', linewidth=2, label=f"TP1 Only (Trades: {len(y_tp1)-1})")
        plt.title("Cumulative Equity Curve: TP1 Only (+2R)")
        plt.xlabel("Trade Count")
        plt.ylabel("Cumulative Returns (R)")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(fontsize=9, loc='upper left')
        
        # Subplot 2: Split TP1/TP2
        plt.subplot(1, 2, 2)
        y_split = equity_curves['split'][0.0]
        plt.plot(range(len(y_split)), y_split, color='green', linewidth=2, label=f"Split TP1/TP2 (Trades: {len(y_split)-1})")
        plt.title("Cumulative Equity Curve: Split TP1/TP2 (+2R/+4R)")
        plt.xlabel("Trade Count")
        plt.ylabel("Cumulative Returns (R)")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(fontsize=9, loc='upper left')
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, "backtest_equity_curve.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"\nEquity curve plot saved successfully to: {plot_path}")
        
        # Plot Dollar-Based Account Balance Curves
        import matplotlib.ticker as mticker
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(balance_curves['tp1'])), balance_curves['tp1'], label=f"TP1 Only (Final: ${balance_curves['tp1'][-1]:,.2f})", color='blue', linewidth=2)
        plt.plot(range(len(balance_curves['split'])), balance_curves['split'], label=f"Split TP1/TP2 (Final: ${balance_curves['split'][-1]:,.2f})", color='green', linewidth=2)
        plt.title("Dollar-Based Account Balance Curve (Starting Balance: $100,000)")
        plt.xlabel("Trade Count")
        plt.ylabel("Account Balance ($)")
        plt.gca().yaxis.set_major_formatter(mticker.StrMethodFormatter('${x:,.0f}'))
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(fontsize=10, loc='upper left')
        plt.tight_layout()
        bal_plot_path = os.path.join(output_dir, "balance_curve.png")
        plt.savefig(bal_plot_path, dpi=150)
        plt.close()
        print(f"Account balance curve plot saved successfully to: {bal_plot_path}")
    else:
        plt.figure(figsize=(14, 6))
        
        # Subplot 1: TP1 Only
        plt.subplot(1, 2, 1)
        colors = plt.cm.viridis(np.linspace(0, 0.9, len(thresholds)))
        for i, th in enumerate(thresholds):
            y = equity_curves['tp1'][th]
            plt.plot(range(len(y)), y, label=f"Prob > {th:.2f} (Trades: {len(y)-1})", color=colors[i], linewidth=2)
        plt.title("Cumulative Equity Curve: TP1 Only (+2R)")
        plt.xlabel("Trade Count")
        plt.ylabel("Cumulative Returns (R)")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(fontsize=9, loc='upper left')
        
        # Subplot 2: Split TP1/TP2
        plt.subplot(1, 2, 2)
        for i, th in enumerate(thresholds):
            y = equity_curves['split'][th]
            plt.plot(range(len(y)), y, label=f"Prob > {th:.2f} (Trades: {len(y)-1})", color=colors[i], linewidth=2)
        plt.title("Cumulative Equity Curve: Split TP1/TP2 (+2R/+4R)")
        plt.xlabel("Trade Count")
        plt.ylabel("Cumulative Returns (R)")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(fontsize=9, loc='upper left')
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, "backtest_equity_curve.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"\nEquity curve plot saved successfully to: {plot_path}")

def main():
    parser = argparse.ArgumentParser(description="LightGBM Trade Setup Backtester")
    parser.add_argument('--dataset', type=str, default="demo_ml_dataset.csv", help="Path to processed ML dataset CSV")
    parser.add_argument('--raw', type=str, default="demo_tv_export.csv", help="Path to raw TradingView export CSV")
    parser.add_argument('--min-train-size', type=int, default=30, help="Initial training size for walk-forward validation")
    parser.add_argument('--step-size', type=int, default=5, help="Step size for walk-forward validation")
    parser.add_argument('--outdir', type=str, default=".", help="Directory to save output files and plots")
    parser.add_argument('--no-ml', action='store_true', help="Bypass LightGBM machine learning step (pure rules-based mode)")
    args = parser.parse_args()
    
    if not os.path.exists(args.dataset):
        raise FileNotFoundError(f"ML dataset not found: {args.dataset}")
    if not os.path.exists(args.raw):
        raise FileNotFoundError(f"Raw TradingView CSV not found: {args.raw}")
        
    print(f"Loading ML dataset: {args.dataset}")
    df_ml = pd.read_csv(args.dataset)
    
    print(f"Loading raw TradingView data: {args.raw}")
    df_raw = pd.read_csv(args.raw)
    df_raw = clean_columns(df_raw)
    
    # Check that required time columns match types
    df_ml['time'] = pd.to_numeric(df_ml['time'], errors='coerce')
    df_raw['time'] = pd.to_numeric(df_raw['time'], errors='coerce')
    
    if args.no_ml:
        df_oos = df_ml
        run_backtest(df_oos, df_raw, args.outdir, no_ml=True)
    else:
        # Run Walk-Forward Validation
        df_oos = run_walk_forward_validation(df_ml, min_train_size=args.min_train_size, step_size=args.step_size)
        run_backtest(df_oos, df_raw, args.outdir, no_ml=False)

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"Backtester error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
