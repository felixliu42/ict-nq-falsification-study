"""
Walk-forward validation for the baseline strategy (used by run_baseline_backtest.py).

The baseline configuration calls run_walk_forward with
window_size = 180 * 24 * 3600 * 1000 (a 6-month chronological training window
in milliseconds, with a fixed 1-year warm-up before the first prediction).

The old regime-balancing comparison experiment that used to live in this file
(evaluate_backtest / main, with its hardcoded 2021-2025 Sharpe date range) has
been archived to archive/scripts/evaluate_regime_balancing_full.py.
"""
import pandas as pd
import numpy as np
import lightgbm as lgb

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
        # This selects exactly the same rows as a boolean mask
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
