import pandas as pd
import numpy as np
import os
import json
import argparse
from datetime import datetime, time
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Import basic elements from baseline
from baseline_strategy import clean_columns, simulate_trade_execution

def calculate_atr(df_raw, period=14):
    """
    Calculate Wilder's Average True Range (ATR) matching TradingView ta.atr(14).
    """
    high = df_raw['high']
    low = df_raw['low']
    close = df_raw['close']
    close_prev = close.shift(1)
    
    tr = pd.concat([
        high - low,
        (high - close_prev).abs(),
        (low - close_prev).abs()
    ], axis=1).max(axis=1)
    
    # Calculate Wilder's RMA
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    return atr

def get_trade_R(row, df_raw):
    """
    Calculate the baseline risk R for a setup row.
    """
    if 'trade_R' in row and not pd.isna(row['trade_R']):
        return row['trade_R']
        
    entry_time = int(row['time'])
    sweep_dir = int(row['sweep_direction'])
    time_since = int(row['time_since_sweep'])
    
    raw_indices = df_raw.index[df_raw['time'] == entry_time].tolist()
    if not raw_indices:
        time_diffs = np.abs(df_raw['time'] - entry_time)
        idx = int(np.argmin(time_diffs))
    else:
        idx = raw_indices[0]
        
    entry_price = float(df_raw.loc[idx, 'close'])
    sweep_idx = idx - time_since
    if sweep_idx < 0:
        return None
        
    if sweep_dir == -1:
        sweep_extreme = float(df_raw.loc[sweep_idx, 'high'])
    else:
        sweep_extreme = float(df_raw.loc[sweep_idx, 'low'])
        
    return abs(entry_price - sweep_extreme)

def precompute_metrics(df_ml, df_raw, raw_time_to_idx, atr_series):
    """
    Precompute trade risk (R), sweep penetration bucket, and entry price for each setup.
    This runs once globally to avoid costly repeated DataFrame searches.
    """
    penetration_buckets = []
    trade_Rs = []
    entry_prices = []
    
    raw_times = df_raw['time'].values
    
    for _, row in df_ml.iterrows():
        entry_time = int(row['time'])
        time_since = int(row['time_since_sweep'])
        sweep_dir = int(row['sweep_direction'])
        
        # Locate sweep bar index in df_raw
        raw_idx = raw_time_to_idx.get(entry_time)
        if raw_idx is None:
            # Fallback binary search if exact match is not found
            pos = np.searchsorted(raw_times, entry_time)
            if pos == 0:
                raw_idx = 0
            elif pos >= len(raw_times):
                raw_idx = len(raw_times) - 1
            else:
                raw_idx = pos if abs(raw_times[pos] - entry_time) < abs(raw_times[pos - 1] - entry_time) else pos - 1
                
        entry_price = float(df_raw.loc[raw_idx, 'close'])
        entry_prices.append(entry_price)
        
        sweep_idx = raw_idx - time_since
        if sweep_idx < 0:
            penetration_buckets.append('unknown')
            trade_Rs.append(np.nan)
            continue
            
        # Wick size calculation
        r_open = float(df_raw.loc[sweep_idx, 'open'])
        r_high = float(df_raw.loc[sweep_idx, 'high'])
        r_low = float(df_raw.loc[sweep_idx, 'low'])
        r_close = float(df_raw.loc[sweep_idx, 'close'])
        
        if sweep_dir == -1: # Bearish sweep
            wick_size = r_high - max(r_open, r_close)
            sweep_extreme = r_high
        else: # Bullish sweep
            wick_size = min(r_open, r_close) - r_low
            sweep_extreme = r_low
            
        atr_val = atr_series.loc[sweep_idx]
        if atr_val > 0.0:
            p_val = wick_size / atr_val
        else:
            p_val = 0.0
            
        if p_val < 0.5:
            penetration_buckets.append('small')
        elif p_val < 1.5:
            penetration_buckets.append('medium')
        else:
            penetration_buckets.append('large')
            
        trade_Rs.append(abs(entry_price - sweep_extreme))
        
    df_ml['penetration_bucket'] = penetration_buckets
    df_ml['trade_R'] = trade_Rs
    df_ml['entry_price'] = entry_prices

def run_experiment(df_ml, df_raw, config, atr_series):
    """
    Simulate trades for a single configuration and return the results.
    """
    # 1. Localize timezone to America/New_York (EST/EDT) for session filtering
    df_ml_exp = df_ml.copy()
    df_ml_exp['dt'] = pd.to_datetime(df_ml_exp['time'], unit='ms').dt.tz_localize('UTC').dt.tz_convert('America/New_York')
    df_ml_exp['date'] = df_ml_exp['dt'].dt.date
    df_ml_exp['time_only'] = df_ml_exp['dt'].dt.time
    
    # Identify unique sweeps using the sweep bar timestamp
    # 300000 ms = 5 minutes
    df_ml_exp['sweep_id'] = df_ml_exp['time'] - df_ml_exp['time_since_sweep'] * 300000
    
    # Sort chronologically
    df_ml_exp = df_ml_exp.sort_values('time').reset_index(drop=True)
    
    # Track sequence number per sweep_id
    df_ml_exp['setup_num'] = df_ml_exp.groupby('sweep_id').cumcount() + 1
    
    # Precompute metrics locally if not already precomputed (e.g. in unit tests)
    if 'penetration_bucket' not in df_ml_exp or 'trade_R' not in df_ml_exp or 'entry_price' not in df_ml_exp:
        raw_times = df_raw['time'].values
        raw_time_to_idx = {t: idx for idx, t in enumerate(raw_times)}
        precompute_metrics(df_ml_exp, df_raw, raw_time_to_idx, atr_series)
    
    # 2. Filter setups based on config parameters
    filtered_indices = []
    
    # State variables for Daily Bias tracking
    current_bias_date = None
    active_bias = None
    
    # Session config
    sess_cfg = config.get('session_filter', 'none')
    
    # Daily bias config
    use_daily_bias = config.get('enforce_daily_bias', False)
    bias_reset = config.get('bias_reset_mode', 'entire_day') # entire_day, lunch_reset, htf_sweep
    
    # Liquidity config
    liq_cfg = config.get('allowed_liquidity_types', 'all')
    
    # Max setup sequence config
    max_setup = config.get('max_setup_number', 3)
    
    # Penetration bucket config
    allowed_penetration = config.get('allowed_penetration', ['small', 'medium', 'large'])
    
    for idx, row in df_ml_exp.iterrows():
        # A. Session Filter
        t_val = row['time_only']
        if sess_cfg == 'morning_only':
            if not (time(9, 30) <= t_val <= time(12, 0)):
                continue
        elif sess_cfg == 'lunch_excluded':
            if not (time(9, 30) <= t_val <= time(16, 0)) or (time(12, 0) <= t_val <= time(13, 30)):
                continue
                
        # B. Setup Number Filter
        if row['setup_num'] > max_setup:
            continue
            
        # C. Penetration Filter
        if row['penetration_bucket'] not in allowed_penetration:
            continue
            
        # D. Liquidity Hierarchy Filter
        l_type = int(row['liquidity_type'])
        l_strength = float(row['liquidity_strength'])
        
        if liq_cfg == 'daily_only':
            if l_type != 1: continue
        elif liq_cfg == 'daily_stacked_4h':
            if not (l_type == 1 or (l_type == 3 and l_strength > 1.0)):
                continue
        elif liq_cfg == 'stacked_only':
            if l_strength <= 1.0: continue
        elif liq_cfg == 'exclude_singular_1h':
            if l_type == 2 and l_strength == 1.0: continue
            
        # E. Daily Bias Filter
        if use_daily_bias:
            setup_date = row['date']
            sweep_dir = int(row['sweep_direction'])
            
            # Reset bias if new calendar day
            if current_bias_date != setup_date:
                current_bias_date = setup_date
                active_bias = None
                
            # Lunch Reset
            if bias_reset == 'lunch_reset' and t_val >= time(12, 0):
                if active_bias is not None and t_val >= time(12, 0) and (row['time'] - df_ml_exp.loc[filtered_indices[-1], 'time'] if filtered_indices else 0) > 0:
                     active_bias = None
                     
            # HTF Sweep Reset
            if bias_reset == 'htf_sweep' and l_type in [1, 3]:
                active_bias = None
                
            # Reject setups in the opposite direction
            if active_bias is not None and sweep_dir != active_bias:
                continue
                
            # Establish new bias if neutral
            if active_bias is None:
                active_bias = sweep_dir
                
        filtered_indices.append(idx)
        
    trades_df = df_ml_exp.loc[filtered_indices].copy()
    
    # 3. Simulate exits using Stop Expansion Multipliers
    stop_mult = config.get('stop_multiplier', 1.0)
    tp_mode = config.get('tp_mode', 'split')
    
    trade_returns = []
    trade_durations = []
    trade_dollar_pnls = []
    
    # Session breakdown tags
    session_returns = {'NY': [], 'London': [], 'Asian': [], 'Other': []}
    session_durations = {'NY': [], 'London': [], 'Asian': [], 'Other': []}
    
    # Liquidity breakdown tags
    liq_returns = {1: [], 2: [], 3: []}
    
    # Setup number breakdown tags
    setup_num_returns = {1: [], 2: [], 3: [], 'other': []}
    
    # Direction breakdown tags
    dir_returns = {1: [], -1: []}
    
    for _, row in trades_df.iterrows():
        # Modify stop loss level if multiplier is not 1.0
        modified_row = row.copy()
        if stop_mult != 1.0:
            # Recompute stop loss with expansion
            R = row['trade_R']
            if R is not None and not pd.isna(R):
                entry_price = row['entry_price']
                expanded_sl = entry_price - row['sweep_direction'] * R * stop_mult
                modified_row['suggested_sl'] = expanded_sl
                
        # Simulate execution
        ret, outcome, duration = simulate_trade_execution(modified_row, df_raw, tp_mode=tp_mode)
        if ret is not None:
            trade_returns.append(ret)
            trade_durations.append(duration)
            
            # Dollar PnL (16 shares)
            R = row['trade_R']
            if R is not None and not pd.isna(R):
                trade_dollar_pnls.append(16.0 * R * ret)
                
            # Session tags
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
                
            # Liquidity tags
            l_t = int(row['liquidity_type'])
            if l_t in liq_returns:
                liq_returns[l_t].append(ret)
                
            # Setup number tags
            s_n = int(row['setup_num'])
            if s_n in setup_num_returns:
                setup_num_returns[s_n].append(ret)
            else:
                setup_num_returns['other'].append(ret)
                
            # Direction tags
            s_d = int(row['sweep_direction'])
            if s_d in dir_returns:
                dir_returns[s_d].append(ret)
                liq_returns[l_t].append(ret)
                
            # Setup number tags
            s_n = int(row['setup_num'])
            if s_n in setup_num_returns:
                setup_num_returns[s_n].append(ret)
            else:
                setup_num_returns['other'].append(ret)
                
            # Direction tags
            s_d = int(row['sweep_direction'])
            if s_d in dir_returns:
                dir_returns[s_d].append(ret)
                
    # Calculate performance metrics
    total_trades = len(trade_returns)
    if total_trades == 0:
        return {
            'total_trades': 0, 'win_rate': 0.0, 'profit_factor': 0.0, 'expectancy': 0.0,
            'max_dd': 0.0, 'sharpe': 0.0, 'avg_trades_day': 0.0, 'avg_duration': 0.0,
            'final_balance': 100000.0, 'dollar_pnls': [], 'returns': [], 'durations': [],
            'session_breakdown': {}, 'liq_breakdown': {}, 'setup_breakdown': {}, 'dir_breakdown': {}
        }
        
    wins = [r for r in trade_returns if r > 0]
    losses = [abs(r) for r in trade_returns if r < 0]
    
    win_rate = len(wins) / total_trades
    gains = sum(wins)
    total_losses = sum(losses)
    profit_factor = gains / total_losses if total_losses > 0 else (gains if gains > 0 else 1.0)
    expectancy = sum(trade_returns) / total_trades
    
    # Max DD in R
    cum_returns = [0] + list(np.cumsum(trade_returns))
    running_max = np.maximum.accumulate(cum_returns)
    drawdowns = running_max - cum_returns
    max_dd = np.max(drawdowns)
    
    # Sharpe Ratio (trade-level)
    std_ret = np.std(trade_returns)
    sharpe = (np.mean(trade_returns) / std_ret) if std_ret > 0 else 0.0
    
    # Average Trades per Day
    total_days = df_ml_exp['date'].nunique()
    avg_trades_day = total_trades / total_days if total_days > 0 else 0.0
    avg_duration = np.mean(trade_durations)
    
    # Final Balance
    final_balance = 100000.0 + sum(trade_dollar_pnls)
    
    return {
        'total_trades': total_trades,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'expectancy': expectancy,
        'max_dd': max_dd,
        'sharpe': sharpe,
        'avg_trades_day': avg_trades_day,
        'avg_duration': avg_duration,
        'final_balance': final_balance,
        'dollar_pnls': trade_dollar_pnls,
        'returns': trade_returns,
        'durations': trade_durations,
        # Breakdowns
        'session_breakdown': session_returns,
        'liq_breakdown': liq_returns,
        'setup_breakdown': setup_num_returns,
        'dir_breakdown': dir_returns
    }

def main():
    parser = argparse.ArgumentParser(description="MNQ Liquidity Strategy Research Harness")
    parser.add_argument('--dataset', type=str, default="demo_ml_dataset.csv", help="ML dataset path")
    parser.add_argument('--raw', type=str, default="demo_tv_export.csv", help="Raw TV export path")
    parser.add_argument('--outdir', type=str, default=".", help="Output directory")
    args = parser.parse_args()
    
    if not os.path.exists(args.dataset) or not os.path.exists(args.raw):
        print("Missing dataset files.")
        return
        
    df_ml = pd.read_csv(args.dataset)
    df_raw = pd.read_csv(args.raw)
    df_raw = clean_columns(df_raw)
    
    df_ml['time'] = pd.to_numeric(df_ml['time'], errors='coerce')
    df_raw['time'] = pd.to_numeric(df_raw['time'], errors='coerce')
    
    print("Calculating True Range and ATR(14) series...")
    atr_series = calculate_atr(df_raw, 14)
    
    print("Precomputing raw index mapping...")
    raw_times = df_raw['time'].values
    raw_time_to_idx = {t: idx for idx, t in enumerate(raw_times)}
    
    print("Precomputing sweep penetration buckets and trade risks globally...")
    precompute_metrics(df_ml, df_raw, raw_time_to_idx, atr_series)
    
    # Define experiment configurations (Shallow parameter sweeps)
    # Target: 24 configurations to avoid combinatorial explosion
    configs = []
    
    # Base Configuration
    base_cfg = {
        'name': 'Baseline (No Filters)',
        'session_filter': 'none',
        'enforce_daily_bias': False,
        'stop_multiplier': 1.0,
        'allowed_liquidity_types': 'all',
        'max_setup_number': 3,
        'allowed_penetration': ['small', 'medium', 'large'],
        'tp_mode': 'split'
    }
    configs.append(base_cfg)
    
    # Sweep 1: Session Filters x Stop Multipliers
    session_filters = ['morning_only', 'lunch_excluded']
    stop_multipliers = [0.90, 1.05, 1.10]
    for sf in session_filters:
        for sm in stop_multipliers:
            configs.append({
                'name': f"Sess: {sf} | Stop Mult: {sm:.2f}",
                'session_filter': sf,
                'enforce_daily_bias': False,
                'stop_multiplier': sm,
                'allowed_liquidity_types': 'all',
                'max_setup_number': 3,
                'allowed_penetration': ['small', 'medium', 'large'],
                'tp_mode': 'split'
            })
            
    # Sweep 2: Session Filters x Daily Directional Bias Resets
    bias_resets = ['entire_day', 'lunch_reset', 'htf_sweep']
    for sf in ['morning_only', 'lunch_excluded']:
        for br in bias_resets:
            configs.append({
                'name': f"Sess: {sf} | Daily Bias Reset: {br}",
                'session_filter': sf,
                'enforce_daily_bias': True,
                'bias_reset_mode': br,
                'stop_multiplier': 1.0,
                'allowed_liquidity_types': 'all',
                'max_setup_number': 3,
                'allowed_penetration': ['small', 'medium', 'large'],
                'tp_mode': 'split'
            })
            
    # Sweep 3: Liquidity Filters x Setup Sequence Limits
    liq_filters = ['daily_only', 'daily_stacked_4h', 'stacked_only', 'exclude_singular_1h']
    max_setup_limits = [1, 2]
    for lf in liq_filters:
        for ms in max_setup_limits:
            configs.append({
                'name': f"Liq: {lf} | Max Setups: {ms}",
                'session_filter': 'none',
                'enforce_daily_bias': False,
                'stop_multiplier': 1.0,
                'allowed_liquidity_types': lf,
                'max_setup_number': ms,
                'allowed_penetration': ['small', 'medium', 'large'],
                'tp_mode': 'split'
            })
            
    # Sweep 4: Sweep Penetration Bucket Filters
    penetration_filters = [
        ['medium', 'large'], # Exclude small
        ['small', 'medium']  # Exclude large
    ]
    for pf in penetration_filters:
        configs.append({
            'name': f"Penetration Allowed: {pf}",
            'session_filter': 'none',
            'enforce_daily_bias': False,
            'stop_multiplier': 1.0,
            'allowed_liquidity_types': 'all',
            'max_setup_number': 3,
            'allowed_penetration': pf,
            'tp_mode': 'split'
        })
        
    print(f"Loaded {len(configs)} configuration sweeps.")
    
    results_summary = []
    
    # Create directories
    os.makedirs(os.path.join(args.outdir, 'results'), exist_ok=True)
    os.makedirs(os.path.join(args.outdir, 'analysis'), exist_ok=True)
    
    plt.figure(figsize=(14, 8))
    
    for cfg in configs:
        name = cfg['name']
        print(f"Running Experiment: {name}...")
        res = run_experiment(df_ml, df_raw, cfg, atr_series)
        
        # Save trade logs to results/
        trade_log = pd.DataFrame({
            'return': res['returns'],
            'duration': res['durations'],
            'dollar_pnl': res['dollar_pnls']
        })
        log_name = name.replace(":", "").replace("|", "").replace(" ", "_").lower()
        trade_log.to_csv(os.path.join(args.outdir, 'results', f"log_{log_name}.csv"), index=False)
        
        # Calculate drawdown-adjusted return
        # Adjusting return: expectancy / max drawdown
        dd_adj = res['expectancy'] / res['max_dd'] if res['max_dd'] > 0 else 0.0
        
        results_summary.append({
            'name': name,
            'total_trades': res['total_trades'],
            'win_rate': res['win_rate'],
            'profit_factor': res['profit_factor'],
            'expectancy': res['expectancy'],
            'max_dd': res['max_dd'],
            'sharpe': res['sharpe'],
            'avg_trades_day': res['avg_trades_day'],
            'avg_duration': res['avg_duration'],
            'final_balance': res['final_balance'],
            'dd_adjusted_return': dd_adj,
            'session_breakdown': res['session_breakdown'],
            'liq_breakdown': res['liq_breakdown'],
            'setup_breakdown': res['setup_breakdown'],
            'dir_breakdown': res['dir_breakdown']
        })
        
        # Plot running balance
        if res['total_trades'] > 0:
            cum_bal = [100000.0] + list(100000.0 + np.cumsum(res['dollar_pnls']))
            plt.plot(range(len(cum_bal)), cum_bal, label=f"{name[:30]}", alpha=0.7, linewidth=1.5)
            
    # Format and Save comparative balance plot
    plt.title("Comparative Account Balance Curves (Start: $100k, 16 Shares/Trade)", fontsize=14, fontweight='bold')
    plt.xlabel("Trade Count", fontsize=12)
    plt.ylabel("Account Balance ($)", fontsize=12)
    plt.gca().yaxis.set_major_formatter(mticker.StrMethodFormatter('${x:,.0f}'))
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=8, loc='best', ncol=2)
    plt.tight_layout()
    bal_plot_path = os.path.join(args.outdir, 'analysis', 'comparative_balance_curves.png')
    plt.savefig(bal_plot_path, dpi=150)
    plt.close()
    print(f"Comparative plot saved to: {bal_plot_path}")
    
    # Sort Leaderboard
    leaderboard_df = pd.DataFrame(results_summary)
    
    # Sort by expectancy descending, then DD adjusted return descending
    leaderboard_df = leaderboard_df.sort_values(by=['expectancy', 'dd_adjusted_return'], ascending=False).reset_index(drop=True)
    
    # Print Leaderboard Markdown Table
    print("\n" + "="*145)
    print("EXPERIMENTAL SWEEP LEADERBOARD")
    print("="*145)
    print(f"| Rank | {'Configuration Name':<45} | {'Trades':<6} | {'Win Rate':<8} | {'PF':<6} | {'Expectancy (R)':<14} | {'Max DD (R)':<10} | {'Sharpe':<8} | {'Trades/Day':<10} | {'Final Balance':<14} |")
    print(f"|{'-'*6}|{'-'*47}|{'-'*8}|{'-'*10}|{'-'*8}|{'-'*16}|{'-'*12}|{'-'*10}|{'-'*12}|{'-'*16}|")
    for i, r in leaderboard_df.iterrows():
        pf_str = f"{r['profit_factor']:.2f}" if r['profit_factor'] != np.inf else "Inf"
        if r['total_trades'] == 0:
            pf_str = "N/A"
        print(f"| {i+1:<4} | {r['name']:<45} | {r['total_trades']:<6} | {r['win_rate']:<8.1%} | {pf_str:<6} | {r['expectancy']:<14.2f} | {r['max_dd']:<10.2f} | {r['sharpe']:<8.2f} | {r['avg_trades_day']:<10.2f} | ${r['final_balance']:<13,.2f} |")
    print("="*145)
    
    # Save leaderboard results to analysis/
    leaderboard_df.drop(columns=['session_breakdown', 'liq_breakdown', 'setup_breakdown', 'dir_breakdown']).to_csv(
        os.path.join(args.outdir, 'analysis', 'leaderboard.csv'), index=False
    )
    
    # Write details breakdown file to analysis/breakdowns.txt
    breakdown_file_path = os.path.join(args.outdir, 'analysis', 'breakdowns.txt')
    with open(breakdown_file_path, 'w', encoding='utf-8') as f:
        f.write("EXPERIMENT BREAKDOWNS DETAILS\n")
        f.write("="*80 + "\n\n")
        for r in results_summary:
            f.write(f"Experiment: {r['name']}\n")
            f.write(f"Total Trades: {r['total_trades']} | Win Rate: {r['win_rate']:.1%} | Expectancy: {r['expectancy']:.2f}R\n")
            f.write("-" * 40 + "\n")
            
            # Session Breakdown
            f.write("Session Breakdown (Expectancy / Trades):\n")
            for sess, rets in r['session_breakdown'].items():
                s_expectancy = np.mean(rets) if len(rets) > 0 else 0.0
                f.write(f"  - {sess}: {s_expectancy:.2f}R ({len(rets)} trades)\n")
                
            # Liquidity Breakdown
            f.write("Liquidity Type Breakdown:\n")
            for l_t, rets in r['liq_breakdown'].items():
                l_name = "Daily" if l_t == 1 else ("1H" if l_t == 2 else "4H")
                l_expectancy = np.mean(rets) if len(rets) > 0 else 0.0
                f.write(f"  - {l_name}: {l_expectancy:.2f}R ({len(rets)} trades)\n")
                
            # Setup Number Breakdown
            f.write("Setup Sequence Number Breakdown:\n")
            for s_n, rets in r['setup_breakdown'].items():
                s_expectancy = np.mean(rets) if len(rets) > 0 else 0.0
                f.write(f"  - Setup #{s_n}: {s_expectancy:.2f}R ({len(rets)} trades)\n")
                
            # Direction Breakdown
            f.write("Trade Direction Breakdown:\n")
            for s_d, rets in r['dir_breakdown'].items():
                d_name = "LONG" if s_d == 1 else "SHORT"
                d_expectancy = np.mean(rets) if len(rets) > 0 else 0.0
                f.write(f"  - {d_name}: {d_expectancy:.2f}R ({len(rets)} trades)\n")
            f.write("\n" + "="*80 + "\n\n")
            
    print(f"Breakdowns report saved to: {breakdown_file_path}")

if __name__ == '__main__':
    main()
