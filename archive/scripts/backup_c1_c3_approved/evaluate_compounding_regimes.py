import os
import pandas as pd
import numpy as np
from backtest import simulate_trade_execution, get_trade_R, run_walk_forward_validation, clean_columns

YEARS_CONFIG = {
    2020: ("2020-06-25", "2021-06-25"),
    2021: ("2021-06-25", "2022-06-25"),
    2022: ("2022-06-25", "2023-06-25"),
    2023: ("2023-06-25", "2024-06-25"),
    2024: ("2024-06-25", "2025-06-25"),
    2025: ("2025-06-25", "2026-06-25")
}

def calculate_daily_sharpe(dollar_pnls, trade_dates, start_date_str, end_date_str, is_compounded=False, initial_balance=100000.0):
    """
    Calculate annualized Sharpe ratio using daily returns.
    """
    # Create date range of trading days (excluding weekends)
    all_days = pd.date_range(start=start_date_str, end=end_date_str, freq='B')
    daily_df = pd.DataFrame(index=all_days)
    daily_df['pnl'] = 0.0
    
    # Map trade PnLs to their dates
    trade_df = pd.DataFrame({'date': pd.to_datetime(trade_dates), 'pnl': dollar_pnls})
    # Group by date to handle multiple trades on same day
    daily_grouped = trade_df.groupby('date')['pnl'].sum()
    
    daily_df = daily_df.join(daily_grouped, how='left', rsuffix='_grouped')
    daily_df['pnl'] = daily_df['pnl_grouped'].fillna(0.0)
    
    # Calculate daily returns
    if not is_compounded:
        daily_df['return'] = daily_df['pnl'] / initial_balance
    else:
        # Calculate daily balance and return dynamically
        balances = []
        current_bal = initial_balance
        returns = []
        for pnl in daily_df['pnl'].values:
            ret = pnl / current_bal
            returns.append(ret)
            current_bal += pnl
        daily_df['return'] = returns
        
    daily_returns = daily_df['return'].values
    mean_ret = np.mean(daily_returns)
    std_ret = np.std(daily_returns)
    
    if std_ret > 0:
        # Annualize by multiplying by sqrt(252) trading days
        return (mean_ret / std_ret) * np.sqrt(252)
    return 0.0

def calculate_monthly_sharpe(dollar_pnls, trade_dates, start_date_str, end_date_str, is_compounded=False, initial_balance=100000.0):
    """
    Calculate annualized Sharpe ratio using monthly returns.
    """
    # Create date range of calendar months
    start_dt = pd.to_datetime(start_date_str)
    end_dt = pd.to_datetime(end_date_str)
    
    # Generate PeriodIndex for all months in the date range
    all_months = pd.period_range(start=start_dt, end=end_dt, freq='M')
    monthly_df = pd.DataFrame(index=all_months)
    monthly_df['pnl'] = 0.0
    
    # Map trade PnLs to Month Periods
    trade_df = pd.DataFrame({
        'period': pd.to_datetime(trade_dates).to_period('M'),
        'pnl': dollar_pnls
    })
    
    # Group by month and sum
    monthly_grouped = trade_df.groupby('period')['pnl'].sum()
    monthly_df = monthly_df.join(monthly_grouped, how='left', rsuffix='_grouped')
    monthly_df['pnl'] = monthly_df['pnl_grouped'].fillna(0.0)
    
    # Calculate monthly returns
    if not is_compounded:
        monthly_df['return'] = monthly_df['pnl'] / initial_balance
    else:
        # Calculate monthly balance and return dynamically
        current_bal = initial_balance
        returns = []
        for pnl in monthly_df['pnl'].values:
            ret = pnl / current_bal
            returns.append(ret)
            current_bal += pnl
        monthly_df['return'] = returns
        
    monthly_returns = monthly_df['return'].values
    mean_ret = np.mean(monthly_returns)
    std_ret = np.std(monthly_returns)
    
    if std_ret > 0:
        # Annualize by multiplying by sqrt(12) months
        return (mean_ret / std_ret) * np.sqrt(12)
    return 0.0

def run_year_comparison(year, df_ml, df_raw):
    # Run walk-forward validation
    df_oos = run_walk_forward_validation(df_ml, min_train_size=30, step_size=5)
    trades_df = df_oos[(df_oos['pred_prob'] > 0.60) & (df_oos['ny_session'] == 1.0)].copy()
    
    # Pre-simulate trades to get outcomes
    trade_outcomes = []
    for _, row in trades_df.iterrows():
        ret, outcome, duration = simulate_trade_execution(row, df_raw, tp_mode='split')
        R = get_trade_R(row, df_raw)
        if ret is not None and R is not None:
            # We also record the date of the trade
            dt = pd.to_datetime(row['time'], unit='ms').tz_localize('UTC').tz_convert('America/New_York').date()
            trade_outcomes.append({'ret': ret, 'R': R, 'date': dt})
            
    total_trades = len(trade_outcomes)
    if total_trades == 0:
        return {
            'year': year, 'trades': 0,
            'flat_return': 0.0, 'flat_dd': 0.0, 'flat_sharpe': 0.0,
            'comp_return': 0.0, 'comp_dd': 0.0, 'comp_sharpe': 0.0
        }
        
    # --- 1. FLAT CONTRACT SIZING (16 Contracts) ---
    flat_pnls = []
    flat_balances = [100000.0]
    flat_dates = []
    
    for t in trade_outcomes:
        # 16 contracts of MNQ (multiplier of 32.0, since 1 contract is $2 per point)
        pnl = 32.0 * t['R'] * t['ret']
        flat_pnls.append(pnl)
        flat_balances.append(flat_balances[-1] + pnl)
        flat_dates.append(t['date'])
        
    flat_final_ret = (flat_balances[-1] - 100000.0) / 100000.0 * 100
    
    # Flat Drawdown
    flat_running_max = np.maximum.accumulate(flat_balances)
    flat_dds = (flat_running_max - flat_balances) / flat_running_max * 100
    flat_max_dd = np.max(flat_dds)
    
    start_date, end_date = YEARS_CONFIG[year]
    flat_sharpe = calculate_daily_sharpe(flat_pnls, flat_dates, start_date, end_date, is_compounded=False)
    
    # --- 2. COMPOUNDED POSITION SIZING (2.0% Risk) ---
    comp_pnls = []
    comp_balances = [100000.0]
    comp_dates = []
    
    for t in trade_outcomes:
        current_bal = comp_balances[-1]
        risk_amt = current_bal * 0.02
        
        # Sizing formula: risk_amt / (R * multiplier_per_contract)
        # For MNQ, 1 contract is $2 per point.
        n_contracts = risk_amt / (t['R'] * 2.0)
        # Round to nearest integer (must be at least 1 contract)
        n_contracts = max(1, int(round(n_contracts)))
        
        # Calculate actual trade profit
        pnl = n_contracts * (t['R'] * 2.0) * t['ret']
        comp_pnls.append(pnl)
        comp_balances.append(current_bal + pnl)
        comp_dates.append(t['date'])
        
    comp_final_ret = (comp_balances[-1] - 100000.0) / 100000.0 * 100
    
    # Comp Drawdown
    comp_running_max = np.maximum.accumulate(comp_balances)
    comp_dds = (comp_running_max - comp_balances) / comp_running_max * 100
    comp_max_dd = np.max(comp_dds)
    
    comp_sharpe = calculate_daily_sharpe(comp_pnls, comp_dates, start_date, end_date, is_compounded=True)
    
    return {
        'year': year,
        'trades': total_trades,
        'flat_return': flat_final_ret,
        'flat_dd': flat_max_dd,
        'flat_sharpe': flat_sharpe,
        'comp_return': comp_final_ret,
        'comp_dd': comp_max_dd,
        'comp_sharpe': comp_sharpe
    }

def main():
    results = []
    for year in sorted(YEARS_CONFIG.keys()):
        export_path = f"data/MNQ_{year}/translated_tv_export.csv"
        ml_dataset_path = f"data/MNQ_{year}/demo_ml_dataset.csv"
        
        df_ml = pd.read_csv(ml_dataset_path)
        df_raw = clean_columns(pd.read_csv(export_path))
        
        df_ml['time'] = pd.to_numeric(df_ml['time'], errors='coerce')
        df_raw['time'] = pd.to_numeric(df_raw['time'], errors='coerce')
        
        metrics = run_year_comparison(year, df_ml, df_raw)
        results.append(metrics)
        
    # Calculate Averages
    avg_trades = np.mean([r['trades'] for r in results])
    
    avg_flat_ret = np.mean([r['flat_return'] for r in results])
    avg_flat_dd = np.mean([r['flat_dd'] for r in results])
    avg_flat_sharpe = np.mean([r['flat_sharpe'] for r in results])
    
    avg_comp_ret = np.mean([r['comp_return'] for r in results])
    avg_comp_dd = np.mean([r['comp_dd'] for r in results])
    avg_comp_sharpe = np.mean([r['comp_sharpe'] for r in results])
    
    print("\n" + "="*115)
    print("REGIME COMPARING SIZING MODELS leaderboard (NY Session, 0.60 Threshold, Split TP1/TP2)")
    print("="*115)
    print("| Year   | Trades | Flat Return | Comp Return | Flat Max DD | Comp Max DD | Flat Sharpe | Comp Sharpe |")
    print("|--------|--------|-------------|-------------|-------------|-------------|-------------|-------------|")
    for r in results:
        print(f"| {r['year']:<6} | {r['trades']:<6} | {r['flat_return']:>10.1f}% | {r['comp_return']:>10.1f}% | {r['flat_dd']:>10.1f}% | {r['comp_dd']:>10.1f}% | {r['flat_sharpe']:>11.2f} | {r['comp_sharpe']:>11.2f} |")
    print("|--------|--------|-------------|-------------|-------------|-------------|-------------|-------------|")
    print(f"| AVERAGE| {avg_trades:<6.1f} | {avg_flat_ret:>10.1f}% | {avg_comp_ret:>10.1f}% | {avg_flat_dd:>10.1f}% | {avg_comp_dd:>10.1f}% | {avg_flat_sharpe:>11.2f} | {avg_comp_sharpe:>11.2f} |")
    print("="*115)

if __name__ == '__main__':
    main()
