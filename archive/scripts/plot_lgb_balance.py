import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# Import functions from backtest.py
from backtest import clean_columns, run_walk_forward_validation, simulate_trade_execution, get_trade_R

def main():
    # Resolve paths relative to the script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, "demo_ml_dataset.csv")
    raw_path = os.path.join(script_dir, "demo_tv_export.csv")
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"ML dataset not found: {dataset_path}")
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw TradingView CSV not found: {raw_path}")
        
    print(f"Loading ML dataset: {dataset_path}")
    df_ml = pd.read_csv(dataset_path)
    
    print(f"Loading raw TradingView data: {raw_path}")
    df_raw = pd.read_csv(raw_path)
    
    # Clean columns to handle TradingView suffixes and casing
    df_raw = clean_columns(df_raw)
    
    # Ensure time columns are numeric
    df_ml['time'] = pd.to_numeric(df_ml['time'], errors='coerce')
    df_raw['time'] = pd.to_numeric(df_raw['time'], errors='coerce')
    
    # Run expanding window walk-forward validation (min_train_size=30, step_size=5)
    df_oos = run_walk_forward_validation(df_ml, min_train_size=30, step_size=5)
    
    # Sort out-of-sample data chronologically by time
    df_oos = df_oos.sort_values('time').reset_index(drop=True)
    
    # Calculate the total out-of-sample trading days
    total_oos_trading_days = pd.to_datetime(df_oos['time'], unit='ms').dt.date.nunique()
    print(f"Total out-of-sample trading days: {total_oos_trading_days}")
    
    # Evaluate thresholds of 0.5, 0.6, 0.7, and 0.8 for both TP modes ('tp1' and 'split')
    thresholds = [0.5, 0.6, 0.7, 0.8]
    tp_modes = ['tp1', 'split']
    
    results_table = []
    curves = {mode: {} for mode in tp_modes}
    
    for mode in tp_modes:
        for th in thresholds:
            # Filter trades where pred_prob > threshold
            trades_df = df_oos[df_oos['pred_prob'] > th].copy()
            
            trade_returns = []
            dollar_pnls = []
            trade_times = []
            
            for _, row in trades_df.iterrows():
                # Simulate trade execution
                ret, outcome, duration = simulate_trade_execution(row, df_raw, tp_mode=mode)
                if ret is not None:
                    # Calculate trade R
                    R = get_trade_R(row, df_raw)
                    if R is not None:
                        # Dollar PnL = 16.0 * R * ret
                        dollar_pnl = 16.0 * R * ret
                        trade_returns.append(ret)
                        dollar_pnls.append(dollar_pnl)
                        trade_times.append(row['time'])
            
            # Calculate the running account balance starting at $100,000
            initial_balance = 100000.0
            current_balance = initial_balance
            balances = [initial_balance]
            
            # Start dates curve from the beginning of the out-of-sample period
            dates = [pd.to_datetime(df_oos.iloc[0]['time'], unit='ms')]
            
            for t_time, pnl in zip(trade_times, dollar_pnls):
                current_balance += pnl
                balances.append(current_balance)
                dates.append(pd.to_datetime(t_time, unit='ms'))
                
            # Store balance curves for plotting
            curves[mode][th] = (dates, balances)
            
            # Calculate metrics
            total_trades = len(trade_returns)
            if total_trades > 0:
                win_rate = len([r for r in trade_returns if r > 0]) / total_trades
                expectancy = sum(trade_returns) / total_trades
            else:
                win_rate = 0.0
                expectancy = 0.0
                
            # Average trades per day: len(trades_df) / total_oos_trading_days
            avg_trades_per_day = len(trades_df) / total_oos_trading_days
            
            results_table.append({
                'threshold': th,
                'tp_mode': mode,
                'total_trades': total_trades,
                'win_rate': win_rate,
                'expectancy': expectancy,
                'avg_trades_per_day': avg_trades_per_day,
                'final_balance': current_balance
            })
            
    # Print formatted Markdown table to stdout
    print("\n| Threshold | TP Mode | Total Trades | Win Rate | Expectancy (R) | Avg Trades/Day | Final Balance |")
    print("|-----------|---------|--------------|----------|----------------|----------------|---------------|")
    for r in results_table:
        print(f"| {r['threshold']:.1f}       | {r['tp_mode']:<7} | {r['total_trades']:<12} | {r['win_rate']:<8.1%} | {r['expectancy']:<14.3f} | {r['avg_trades_per_day']:<14.3f} | ${r['final_balance']:,.2f} |")
        
    # Generate a two-panel side-by-side plot saved as lgb_balance_curve.png
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Distinct, professional colors for the thresholds
    colors = {
        0.5: '#1f77b4',  # Steel Blue
        0.6: '#ff7f0e',  # Safety Orange
        0.7: '#2ca02c',  # Forest Green
        0.8: '#d62728'   # Crimson Red
    }
    
    # Left Panel: TP1 Only
    ax1.axhline(y=100000.0, color='gray', linestyle='--', alpha=0.5, label='Initial Balance ($100k)')
    for th in thresholds:
        dates, balances = curves['tp1'][th]
        ax1.step(dates, balances, label=f"Prob > {th:.1f} (Final: ${balances[-1]:,.2f})", 
                 color=colors[th], where='post', linewidth=1.8)
    ax1.set_title("Account Balance Curve: TP1 Only (+2R)", fontsize=13, fontweight='bold', pad=10)
    ax1.set_xlabel("Date/Time", fontsize=11)
    ax1.set_ylabel("Account Balance ($)", fontsize=11)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, p: f"${x:,.0f}"))
    ax1.legend(fontsize=9, loc='best')
    
    # Right Panel: Split TP1/TP2
    ax2.axhline(y=100000.0, color='gray', linestyle='--', alpha=0.5, label='Initial Balance ($100k)')
    for th in thresholds:
        dates, balances = curves['split'][th]
        ax2.step(dates, balances, label=f"Prob > {th:.1f} (Final: ${balances[-1]:,.2f})", 
                 color=colors[th], where='post', linewidth=1.8)
    ax2.set_title("Account Balance Curve: Split TP1/TP2 (+2R/+4R)", fontsize=13, fontweight='bold', pad=10)
    ax2.set_xlabel("Date/Time", fontsize=11)
    ax2.set_ylabel("Account Balance ($)", fontsize=11)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, p: f"${x:,.0f}"))
    ax2.legend(fontsize=9, loc='best')
    
    plt.suptitle("LightGBM Walk-Forward Account Balance Simulation (Starting Balance: $100,000)", fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    plot_path = os.path.join(script_dir, "lgb_balance_curve.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\nAccount balance curve plot saved successfully to: {plot_path}")

if __name__ == '__main__':
    main()
