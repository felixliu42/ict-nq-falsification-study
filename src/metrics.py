"""
Sharpe ratio utilities used by the baseline walk-forward validation
(run_baseline_backtest.py).

The old expanding-window sizing experiment that used to live in this file
(run_year_comparison / main, 0.60 threshold study) has been archived to
archive/scripts/evaluate_compounding_regimes_full.py.
"""
import pandas as pd
import numpy as np

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
