import os
import sys
import pandas as pd
import numpy as np
import databento as db
import subprocess
from backtest import simulate_trade_execution, get_trade_R, clean_columns
from evaluate_compounding_regimes import calculate_daily_sharpe, calculate_monthly_sharpe

API_KEY = os.environ.get("DATABENTO_API_KEY", "")
YEARS = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]

def download_data(symbol, year, output_path):
    """
    Download 1-minute OHLCV data from Databento for a specific symbol-year.
    """
    start_date = f"{year}-06-25"
    end_date = f"{year+1}-06-25"
    
    print(f"Downloading {symbol} for {year} ({start_date} to {end_date})...")
    try:
        client = db.Historical(key=API_KEY)
        data = client.timeseries.get_range(
            dataset="GLBX.MDP3",
            schema="ohlcv-1m",
            symbols=[symbol],
            stype_in="continuous",
            start=start_date,
            end=end_date,
        )
        df = data.to_df()
        
        if df.empty:
            print(f"Warning: No data returned for {symbol} in {year}")
            return False
            
        # Format columns as expected by the pipeline
        if "ts_event" in df.columns:
            ts_series = df["ts_event"]
        else:
            ts_series = df.index
            
        df["Time"] = pd.to_datetime(ts_series).astype("int64") // 1_000_000
        df = df.rename(columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume"
        })
        
        output_cols = ["Time", "Open", "High", "Low", "Close", "Volume"]
        df_output = df[output_cols]
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_output.to_csv(output_path, index=False)
        print(f"Successfully saved raw data to {output_path}")
        return True
    except Exception as e:
        print(f"Error downloading {symbol} for {year}: {e}", file=sys.stderr)
        return False

def process_year(symbol, year, folder_prefix, db_symbol):
    """
    Ensure raw data is downloaded, translated, and labeled for the given symbol-year.
    """
    raw_path = f"data/{folder_prefix}_{year}/raw_data.csv"
    translated_path = f"data/{folder_prefix}_{year}/translated_tv_export.csv"
    ml_path = f"data/{folder_prefix}_{year}/demo_ml_dataset.csv"
    
    # 1. Download if missing
    if not os.path.exists(raw_path):
        success = download_data(db_symbol, year, raw_path)
        if not success:
            return False
    else:
        print(f"Raw data for {folder_prefix} {year} already exists locally.")
        
    # 2. Run translation if missing
    if not os.path.exists(translated_path):
        print(f"Translating indicators for {folder_prefix} {year}...")
        # Call pine_translator.py CLI
        cmd = [
            sys.executable,
            "pine_translator.py",
            "--input", raw_path,
            "--output", translated_path
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print(f"Translation failed for {folder_prefix} {year}:\n{res.stderr}", file=sys.stderr)
            return False
    else:
        print(f"Translated indicators for {folder_prefix} {year} already exist.")
        
    # 3. Run labeling pipeline if missing
    if not os.path.exists(ml_path):
        print(f"Running labeling pipeline for {folder_prefix} {year}...")
        # Call pipeline.py CLI
        cmd = [
            sys.executable,
            "pipeline.py",
            "--input", translated_path,
            "--output", ml_path
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print(f"Pipeline failed for {folder_prefix} {year}:\n{res.stderr}", file=sys.stderr)
            return False
    else:
        print(f"ML dataset for {folder_prefix} {year} already exists.")
        
    return True

def run_symbol_backtest(folder_prefix, multiplier, flat_contracts, risk_cap, threshold):
    """
    Run 6-Month rolling walk-forward backtest and print yearly monthly Sharpe breakdown.
    """
    raw_dfs = []
    ml_dfs = []
    
    for year in YEARS:
        raw_path = f"data/{folder_prefix}_{year}/raw_data.csv"
        ml_path = f"data/{folder_prefix}_{year}/demo_ml_dataset.csv"
        raw_dfs.append(pd.read_csv(raw_path))
        ml_dfs.append(pd.read_csv(ml_path))
        
    df_raw_combined = pd.concat(raw_dfs, ignore_index=True)
    df_ml_combined = pd.concat(ml_dfs, ignore_index=True)
    
    df_raw_combined = clean_columns(df_raw_combined)
    df_raw_combined['time'] = pd.to_numeric(df_raw_combined['time'], errors='coerce')
    df_raw_combined = df_raw_combined.sort_values('time').drop_duplicates(subset=['time']).reset_index(drop=True)
    
    df_ml_combined['time'] = pd.to_numeric(df_ml_combined['time'], errors='coerce')
    df_ml_combined = df_ml_combined.sort_values('time').drop_duplicates(subset=['time']).reset_index(drop=True)
    
    # Run 6-Month rolling walk-forward validation (W=1800)
    from evaluate_regime_balancing import run_walk_forward
    base_features = [
        'liquidity_type', 'liquidity_strength', 'sweep_direction', 'sweep_size',
        'bos_strength', 'fvg_rejected', 'retracement_depth', 'time_since_sweep',
        'ny_session', 'london_session', 'asian_session'
    ]
    print(f"\nRunning 6-Month rolling walk-forward validation for {folder_prefix} (W=1800)...")
    df_oos = run_walk_forward(df_ml_combined, window_size=1800, features=base_features)
    
    df_oos['dt'] = pd.to_datetime(df_oos['time'], unit='ms').dt.tz_localize('UTC').dt.tz_convert('America/New_York')
    df_oos['year_num'] = df_oos['dt'].dt.year
    
    # Filter for setups > threshold and NY session
    trades_df = df_oos[(df_oos['pred_prob'] > threshold) & (df_oos['ny_session'] == 1.0)].copy()
    
    trade_outcomes = []
    skipped = 0
    for _, row in trades_df.iterrows():
        R = get_trade_R(row, df_raw_combined)
        if R is not None:
            if R > risk_cap:
                skipped += 1
                continue
                
            ret, outcome, duration = simulate_trade_execution(row, df_raw_combined, tp_mode='split')
            if ret is not None:
                dt = pd.to_datetime(row['time'], unit='ms').tz_localize('UTC').tz_convert('America/New_York').date()
                trade_outcomes.append({
                    'ret': ret,
                    'R': R,
                    'date': dt,
                    'year': row['year_num']
                })
                
    print(f"\nSimulated {len(trade_outcomes)} trades. Skipped {skipped} due to R > {risk_cap} points.")
    
    print("\n" + "="*115)
    print(f"{folder_prefix} 11-YEAR BACKTEST: 6-Month Window, {risk_cap}-Pt Risk Cap (Threshold: {threshold:.2f}, Split TP)")
    print("="*115)
    print("| Year   | Trades | Flat Return | Comp Return | Flat Max DD | Comp Max DD | Flat Sharpe (M) | Comp Sharpe (M) |")
    print("|--------|--------|-------------|-------------|-------------|-------------|-----------------|-----------------|")
    
    flat_balances = [100000.0]
    comp_balances = [100000.0]
    active_years = sorted(list(set([t['year'] for t in trade_outcomes])))
    
    results = []
    
    for yr in active_years:
        yr_trades = [t for t in trade_outcomes if t['year'] == yr]
        n_trades = len(yr_trades)
        
        if n_trades == 0:
            continue
            
        flat_yr_pnls = []
        comp_yr_pnls = []
        flat_yr_dates = []
        comp_yr_dates = []
        
        flat_bal_start = flat_balances[-1]
        comp_bal_start = comp_balances[-1]
        
        flat_yr_balances = [flat_bal_start]
        comp_yr_balances = [comp_bal_start]
        
        for t in yr_trades:
            # Flat
            p_flat = flat_contracts * multiplier * t['R'] * t['ret']
            flat_yr_pnls.append(p_flat)
            flat_balances.append(flat_balances[-1] + p_flat)
            flat_yr_balances.append(flat_yr_balances[-1] + p_flat)
            flat_yr_dates.append(t['date'])
            
            # Comp
            current_bal = comp_balances[-1]
            risk_amt = current_bal * 0.02
            n_contracts = risk_amt / (t['R'] * multiplier)
            n_contracts = max(1, int(round(n_contracts)))
            
            p_comp = n_contracts * (t['R'] * multiplier) * t['ret']
            comp_yr_pnls.append(p_comp)
            comp_balances.append(current_bal + p_comp)
            comp_yr_balances.append(comp_yr_balances[-1] + p_comp)
            comp_yr_dates.append(t['date'])
            
        # Returns
        flat_ret = (flat_yr_balances[-1] - flat_bal_start) / flat_bal_start * 100
        comp_ret = (comp_yr_balances[-1] - comp_bal_start) / comp_bal_start * 100
        
        # Max Drawdown
        flat_running_max = np.maximum.accumulate(flat_yr_balances)
        flat_dds = (flat_running_max - flat_yr_balances) / flat_running_max * 100
        flat_max_dd = np.max(flat_dds)
        
        comp_running_max = np.maximum.accumulate(comp_yr_balances)
        comp_dds = (comp_running_max - comp_yr_balances) / comp_running_max * 100
        comp_max_dd = np.max(comp_dds)
        
        # Sharpe
        start_date = f"{yr}-01-01"
        end_date = f"{yr}-12-31"
        flat_sharpe = calculate_monthly_sharpe(flat_yr_pnls, flat_yr_dates, start_date, end_date, is_compounded=False, initial_balance=flat_bal_start)
        comp_sharpe = calculate_monthly_sharpe(comp_yr_pnls, comp_yr_dates, start_date, end_date, is_compounded=True, initial_balance=comp_bal_start)
        
        print(f"| {yr:<6} | {n_trades:<6} | {flat_ret:>10.1f}% | {comp_ret:>10.1f}% | {flat_max_dd:>10.1f}% | {comp_max_dd:>10.1f}% | {flat_sharpe:>15.2f} | {comp_sharpe:>15.2f} |")
        
        results.append({
            'trades': n_trades, 'flat_ret': flat_ret, 'comp_ret': comp_ret,
            'flat_dd': flat_max_dd, 'comp_dd': comp_max_dd,
            'flat_sharpe': flat_sharpe, 'comp_sharpe': comp_sharpe
        })
        
    avg_trades = np.mean([r['trades'] for r in results])
    avg_flat_ret = np.mean([r['flat_ret'] for r in results])
    avg_comp_ret = np.mean([r['comp_ret'] for r in results])
    avg_flat_dd = np.mean([r['flat_dd'] for r in results])
    avg_comp_dd = np.mean([r['comp_dd'] for r in results])
    avg_flat_sharpe = np.mean([r['flat_sharpe'] for r in results])
    avg_comp_sharpe = np.mean([r['comp_sharpe'] for r in results])
    
    print("|--------|--------|-------------|-------------|-------------|-------------|-----------------|-----------------|")
    print(f"| AVERAGE| {avg_trades:<6.1f} | {avg_flat_ret:>10.1f}% | {avg_comp_ret:>10.1f}% | {avg_flat_dd:>10.1f}% | {avg_comp_dd:>10.1f}% | {avg_flat_sharpe:>15.2f} | {avg_comp_sharpe:>15.2f} |")
    print("="*115)
    
    # Save overall summary statistics to file
    summary_path = f"results_{folder_prefix}.txt"
    with open(summary_path, "w") as f:
        f.write(f"Symbol: {folder_prefix}\n")
        f.write(f"Average Trades: {avg_trades:.1f}\n")
        f.write(f"Average Flat Return: {avg_flat_ret:.1f}%\n")
        f.write(f"Average Compounded Return: {avg_comp_ret:.1f}%\n")
        f.write(f"Average Flat Max DD: {avg_flat_dd:.1f}%\n")
        f.write(f"Average Compounded Max DD: {avg_comp_dd:.1f}%\n")
        f.write(f"Average Flat Sharpe: {avg_flat_sharpe:.2f}\n")
        f.write(f"Average Compounded Sharpe: {avg_comp_sharpe:.2f}\n")
    print(f"Summary saved to {summary_path}")

def main():
    print("Starting Multi-Year and Multi-Symbol Pipeline...")
    
    # 1. Process Nasdaq (NQ/MNQ) for 2015-2025
    print("\n" + "="*50)
    print("PROCESSING NASDAQ (NQ/MNQ)...")
    print("="*50)
    for year in YEARS:
        # Check if year is already done under old structure
        old_raw_path = f"data/MNQ_{year}/translated_tv_export.csv"
        target_raw = f"data/MNQ_{year}/raw_data.csv"
        
        # If we have the old translated export but no raw_data.csv, copy it or re-download
        # To ensure NQ price consistency for all 11 years, we download NQ.c.0 for all years
        process_year(symbol="NQ.c.0", year=year, folder_prefix="MNQ", db_symbol="NQ.c.0")
        
    # 2. Process S&P 500 (ES) for 2015-2025
    print("\n" + "="*50)
    print("PROCESSING S&P 500 (ES)...")
    print("="*50)
    for year in YEARS:
        process_year(symbol="ES.c.0", year=year, folder_prefix="ES", db_symbol="ES.c.0")
        
    # 3. Run Nasdaq Backtest
    # Flat contracts: 16 (equals 16 * $2 = $32/point), risk cap: 60 points, multiplier: 2.0
    run_symbol_backtest(folder_prefix="MNQ", multiplier=2.0, flat_contracts=16, risk_cap=60.0, threshold=0.25)
    
    # 4. Run ES Backtest
    # Flat contracts: 2 (equals 2 * $50 = $100/point), risk cap: 12 points, multiplier: 50.0
    run_symbol_backtest(folder_prefix="ES", multiplier=50.0, flat_contracts=2, risk_cap=12.0, threshold=0.25)

if __name__ == '__main__':
    main()
