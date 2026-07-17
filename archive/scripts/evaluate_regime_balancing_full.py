import os
import sys
import pandas as pd
import numpy as np
import lightgbm as lgb
from backtest import simulate_trade_execution, get_trade_R, clean_columns
from evaluate_compounding_regimes import calculate_daily_sharpe

YEARS = [2020, 2021, 2022, 2023, 2024, 2025]

def calculate_regime_features(df_raw):
    """
    Calculate ATR ratio (volatility regime) and SMA position (trend regime) on the raw series.
    """
    print("Calculating raw price regime features...")
    # Calculate True Range (TR)
    h = df_raw['high'].values
    l = df_raw['low'].values
    c = df_raw['close'].values
    n = len(df_raw)
    
    tr = np.zeros(n)
    tr[0] = h[0] - l[0]
    for i in range(1, n):
        tr[i] = max(h[i] - l[i], abs(h[i] - c[i-1]), abs(l[i] - c[i-1]))
        
    df_raw['tr'] = tr
    # 14-period ATR
    df_raw['atr_14'] = df_raw['tr'].rolling(14).mean()
    # 500-period ATR (long term volatility baseline)
    df_raw['atr_500'] = df_raw['tr'].rolling(500).mean()
    # Volatility Ratio
    df_raw['vol_ratio'] = df_raw['atr_14'] / df_raw['atr_500']
    
    # 500-period SMA Position
    df_raw['sma_500'] = df_raw['close'].rolling(500).mean()
    df_raw['ma_pos'] = df_raw['close'] / df_raw['sma_500']
    
    # Fill NaNs
    df_raw['vol_ratio'] = df_raw['vol_ratio'].fillna(1.0)
    df_raw['ma_pos'] = df_raw['ma_pos'].fillna(1.0)
    
    return df_raw

def run_walk_forward(df_ml, window_size, features):
    """
    Run chronological rolling walk-forward validation.
    If window_size is large (e.g. > 1,000,000), it represents millisecond duration.
    Otherwise, it represents the number of setup rows.
    """
    df_ml = df_ml.sort_values('time').reset_index(drop=True)
    df_ml['pred_prob'] = np.nan
    n_rows = len(df_ml)
    step_size = 5
    
    if window_size > 1000000:
        # Chronological window in milliseconds
        window_ms = window_size
        # df_ml is sorted by time above, so we can slice the training window
        # with searchsorted instead of scanning the full frame every step.
        # This selects exactly the same rows as the previous boolean mask
        # (time >= test_start_time - window_ms) & (time < test_start_time).
        times = df_ml['time'].to_numpy()
        min_time = times[0]
        # Always use a 1-year warm-up period to keep out-of-sample start aligned
        warmup_ms = 365 * 24 * 3600 * 1000
        start_time_ms = min_time + warmup_ms

        start_idx = int(np.searchsorted(times, start_time_ms, side='left'))
        if start_idx >= n_rows:
            raise ValueError("No setups found after chronological warmup size")

        for t in range(start_idx, n_rows, step_size):
            test_end = min(t + step_size, n_rows)
            test_df = df_ml.iloc[t:test_end]
            test_start_time = times[t]

            train_lo = int(np.searchsorted(times, test_start_time - window_ms, side='left'))
            train_hi = int(np.searchsorted(times, test_start_time, side='left'))
            train_df = df_ml.iloc[train_lo:train_hi]
            if len(train_df) < 50:
                train_df = df_ml.iloc[max(0, t - 200) : t]
                
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
            
        df_oos = df_ml.iloc[start_idx:].copy().reset_index(drop=True)
    else:
        # Legacy setup-count window
        for t in range(window_size, n_rows, step_size):
            train_df = df_ml.iloc[t - window_size : t]
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
            
        df_oos = df_ml.iloc[window_size:].copy().reset_index(drop=True)
        
    return df_oos

def evaluate_backtest(df_oos, df_raw_combined, threshold):
    """
    Simulate trade outcomes and calculate overall metrics.
    """
    trades_df = df_oos[(df_oos['pred_prob'] > threshold) & (df_oos['ny_session'] == 1.0)].copy()
    
    trade_outcomes = []
    for _, row in trades_df.iterrows():
        ret, outcome, duration = simulate_trade_execution(row, df_raw_combined, tp_mode='split')
        R = get_trade_R(row, df_raw_combined)
        if ret is not None and R is not None:
            dt = pd.to_datetime(row['time'], unit='ms').tz_localize('UTC').tz_convert('America/New_York').date()
            trade_outcomes.append({
                'ret': ret,
                'R': R,
                'date': dt
            })
            
    if len(trade_outcomes) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
        
    flat_balances = [100000.0]
    comp_balances = [100000.0]
    
    flat_pnls = []
    comp_pnls = []
    flat_dates = []
    comp_dates = []
    
    for t in trade_outcomes:
        # Flat
        p_flat = 32.0 * t['R'] * t['ret']
        flat_pnls.append(p_flat)
        flat_balances.append(flat_balances[-1] + p_flat)
        flat_dates.append(t['date'])
        
        # Comp
        current_bal = comp_balances[-1]
        risk_amt = current_bal * 0.02
        n_contracts = risk_amt / (t['R'] * 2.0)
        n_contracts = max(1, int(round(n_contracts)))
        
        p_comp = n_contracts * (t['R'] * 2.0) * t['ret']
        comp_pnls.append(p_comp)
        comp_balances.append(current_bal + p_comp)
        comp_dates.append(t['date'])
        
    flat_ret = (flat_balances[-1] - 100000.0) / 100000.0 * 100
    comp_ret = (comp_balances[-1] - 100000.0) / 100000.0 * 100
    
    flat_cum_max = np.maximum.accumulate(flat_balances)
    flat_dd = np.max((flat_cum_max - flat_balances) / flat_cum_max * 100)
    
    comp_cum_max = np.maximum.accumulate(comp_balances)
    comp_dd = np.max((comp_cum_max - comp_balances) / comp_cum_max * 100)
    
    flat_sharpe = calculate_daily_sharpe(flat_pnls, flat_dates, "2021-01-01", "2025-12-31", is_compounded=False, initial_balance=100000.0)
    comp_sharpe = calculate_daily_sharpe(comp_pnls, comp_dates, "2021-01-01", "2025-12-31", is_compounded=True, initial_balance=100000.0)
    
    return flat_ret, comp_ret, flat_dd, comp_dd, flat_sharpe, comp_sharpe, len(trade_outcomes)

def main():
    print("Loading and combining datasets for all 6 years...")
    raw_dfs = []
    ml_dfs = []
    
    for year in YEARS:
        raw_path = f"data/MNQ_{year}/translated_tv_export.csv"
        ml_path = f"data/MNQ_{year}/demo_ml_dataset.csv"
        
        raw_dfs.append(pd.read_csv(raw_path))
        ml_dfs.append(pd.read_csv(ml_path))
        
    df_raw_combined = pd.concat(raw_dfs, ignore_index=True)
    df_ml_combined = pd.concat(ml_dfs, ignore_index=True)
    
    df_raw_combined = clean_columns(df_raw_combined)
    df_raw_combined['time'] = pd.to_numeric(df_raw_combined['time'], errors='coerce')
    df_raw_combined = df_raw_combined.sort_values('time').drop_duplicates(subset=['time']).reset_index(drop=True)
    
    df_ml_combined['time'] = pd.to_numeric(df_ml_combined['time'], errors='coerce')
    df_ml_combined = df_ml_combined.sort_values('time').drop_duplicates(subset=['time']).reset_index(drop=True)
    
    # Calculate raw series regime features
    df_raw_combined = calculate_regime_features(df_raw_combined)
    
    # Merge regime features into the ML dataset using pd.merge_asof
    print("Merging regime features into ML dataset...")
    df_ml_combined = pd.merge_asof(
        df_ml_combined,
        df_raw_combined[['time', 'vol_ratio', 'ma_pos']],
        on='time',
        direction='backward'
    )
    
    # -----------------------------------------------------------------
    # STRATEGY 1A: 3-Month Rolling Window (1000 setups)
    # -----------------------------------------------------------------
    print("\n--- Running Strategy 1A: 3-Month Window (1000 setups) ---")
    base_features = [
        'liquidity_type', 'liquidity_strength', 'sweep_direction', 'sweep_size',
        'bos_strength', 'fvg_rejected', 'retracement_depth', 'time_since_sweep',
        'ny_session', 'london_session', 'asian_session'
    ]
    df_oos_1a = run_walk_forward(df_ml_combined, window_size=1000, features=base_features)
    
    # -----------------------------------------------------------------
    # STRATEGY 1B: 6-Month Rolling Window (1800 setups)
    # -----------------------------------------------------------------
    print("\n--- Running Strategy 1B: 6-Month Window (1800 setups) ---")
    df_oos_1b = run_walk_forward(df_ml_combined, window_size=1800, features=base_features)
    
    # -----------------------------------------------------------------
    # STRATEGY 2: 1-Year Window (3500 setups) with Regime Features
    # -----------------------------------------------------------------
    print("\n--- Running Strategy 2: 1-Year Window + Explicit Regime Features ---")
    regime_features = base_features + ['vol_ratio', 'ma_pos']
    df_oos_2 = run_walk_forward(df_ml_combined, window_size=3500, features=regime_features)
    
    # Sweep thresholds and record findings
    # We sweep 0.25 to 0.35 for Strat 1A/1B, and 0.30 to 0.40 for Strat 2
    thresholds = [0.25, 0.30, 0.35]
    
    print("\n" + "="*125)
    print("REGIME BALANCING COMPARATIVE LEADERBOARD (Overall 5-Year continuous metrics 2021-2026)")
    print("="*125)
    print("| Sizing / Strategy | Threshold | Trades | Total Return (%) | Max Drawdown (%) | Daily Sharpe |")
    print("|-------------------|-----------|--------|------------------|------------------|--------------|")
    
    for th in thresholds:
        # Strat 1A (3M)
        f_ret, c_ret, f_dd, c_dd, f_sh, c_sh, trs = evaluate_backtest(df_oos_1a, df_raw_combined, th)
        if trs > 0:
            print(f"| Flat / 3M Window  | {th:.2f}      | {trs:<6} | {f_ret:>16.1f}% | {f_dd:>16.1f}% | {f_sh:>12.2f} |")
            print(f"| Comp / 3M Window  | {th:.2f}      | {trs:<6} | {c_ret:>16.1f}% | {c_dd:>16.1f}% | {c_sh:>12.2f} |")
            print("|-------------------|-----------|--------|------------------|------------------|--------------|")
            
        # Strat 1B (6M)
        f_ret, c_ret, f_dd, c_dd, f_sh, c_sh, trs = evaluate_backtest(df_oos_1b, df_raw_combined, th)
        if trs > 0:
            print(f"| Flat / 6M Window  | {th:.2f}      | {trs:<6} | {f_ret:>16.1f}% | {f_dd:>16.1f}% | {f_sh:>12.2f} |")
            print(f"| Comp / 6M Window  | {th:.2f}      | {trs:<6} | {c_ret:>16.1f}% | {c_dd:>16.1f}% | {c_sh:>12.2f} |")
            print("|-------------------|-----------|--------|------------------|------------------|--------------|")
            
        # Strat 2 (1Y + Regime Features)
        f_ret, c_ret, f_dd, c_dd, f_sh, c_sh, trs = evaluate_backtest(df_oos_2, df_raw_combined, th)
        if trs > 0:
            print(f"| Flat / 1Y + Regime| {th:.2f}      | {trs:<6} | {f_ret:>16.1f}% | {f_dd:>16.1f}% | {f_sh:>12.2f} |")
            print(f"| Comp / 1Y + Regime| {th:.2f}      | {trs:<6} | {c_ret:>16.1f}% | {c_dd:>16.1f}% | {c_sh:>12.2f} |")
            print("|-------------------|-----------|--------|------------------|------------------|--------------|")
            
    print("="*125)

if __name__ == '__main__':
    main()
