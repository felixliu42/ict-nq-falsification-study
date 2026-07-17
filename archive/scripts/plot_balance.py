import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime

# Import functions from backtest.py
from backtest import clean_columns, run_walk_forward_validation, simulate_trade_execution

def main():
    dataset_path = "demo_ml_dataset.csv"
    raw_path = "demo_tv_export.csv"
    initial_balance = 100000.0
    shares = 16
    
    if not os.path.exists(dataset_path) or not os.path.exists(raw_path):
        print("Dataset or raw file missing. Please make sure demo data is generated.")
        return
        
    print(f"Loading datasets...")
    df_ml = pd.read_csv(dataset_path)
    df_raw = pd.read_csv(raw_path)
    df_raw = clean_columns(df_raw)
    
    df_ml['time'] = pd.to_numeric(df_ml['time'], errors='coerce')
    df_raw['time'] = pd.to_numeric(df_raw['time'], errors='coerce')
    
    # Run Walk-Forward Validation
    df_oos = run_walk_forward_validation(df_ml, min_train_size=30, step_size=5)
    
    # Sort out-of-sample data chronologically by time
    df_oos = df_oos.sort_values('time').reset_index(drop=True)
    
    # Define configurations to plot
    configs = [
        {'threshold': 0.50, 'tp_mode': 'tp1', 'label': 'Prob > 0.50 (TP1 Only)', 'color': '#2b7bba'},
        {'threshold': 0.50, 'tp_mode': 'split', 'label': 'Prob > 0.50 (Split TP1/TP2)', 'color': '#f28e2b'},
        {'threshold': 0.70, 'tp_mode': 'tp1', 'label': 'Prob > 0.70 (TP1 Only)', 'color': '#86bcb6'},
        {'threshold': 0.70, 'tp_mode': 'split', 'label': 'Prob > 0.70 (Split TP1/TP2)', 'color': '#e15759'},
    ]
    
    plt.figure(figsize=(12, 7))
    
    # Draw initial balance line
    plt.axhline(y=initial_balance, color='gray', linestyle='--', alpha=0.5, label='Initial Balance ($100k)')
    
    for config in configs:
        th = config['threshold']
        mode = config['tp_mode']
        
        # Filter trades matching threshold
        trades_df = df_oos[df_oos['pred_prob'] > th].copy()
        
        balances = [initial_balance]
        dates = [pd.to_datetime(df_oos.iloc[0]['time'], unit='ms')]
        
        current_balance = initial_balance
        
        for _, row in trades_df.iterrows():
            # Get trade details
            entry_time = int(row['time'])
            sweep_dir = int(row['sweep_direction'])
            time_since = int(row['time_since_sweep'])
            
            # Find setup bar index in raw data to calculate R
            raw_indices = df_raw.index[df_raw['time'] == entry_time].tolist()
            if not raw_indices:
                # Fallback to closest
                time_diffs = np.abs(df_raw['time'] - entry_time)
                idx = int(np.argmin(time_diffs))
            else:
                idx = raw_indices[0]
                
            entry_price = float(df_raw.loc[idx, 'close'])
            sweep_idx = idx - time_since
            if sweep_idx < 0:
                continue
                
            if sweep_dir == -1:
                sweep_extreme = float(df_raw.loc[sweep_idx, 'high'])
            else:
                sweep_extreme = float(df_raw.loc[sweep_idx, 'low'])
                
            R = abs(entry_price - sweep_extreme)
            if R == 0.0:
                continue
                
            # Simulate execution
            ret, outcome, duration = simulate_trade_execution(row, df_raw, tp_mode=mode)
            if ret is not None:
                # Calculate PnL: dollar_pnl = shares * R * ret
                dollar_pnl = shares * R * ret
                current_balance += dollar_pnl
                
                balances.append(current_balance)
                dates.append(pd.to_datetime(entry_time, unit='ms'))
                
        # If no trades occurred, keep initial balance line
        if len(balances) > 1:
            plt.step(dates, balances, label=f"{config['label']} (Final: ${current_balance:,.2f})", 
                     color=config['color'], where='post', linewidth=2)
            print(f"Config: Th={th}, Mode={mode} -> Final Balance: ${current_balance:,.2f} (Total Trades: {len(balances)-1})")
            
    plt.title("Account Balance Curve Simulation\nInitial Balance: $100,000 | 16 Shares per Trade", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Date/Time", fontsize=12)
    plt.ylabel("Account Balance ($)", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # Format Y axis as currency
    import matplotlib.ticker as mtick
    plt.gca().yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, p: f"${x:,.0f}"))
    
    plt.legend(fontsize=10, loc='best')
    plt.tight_layout()
    
    # Save locally
    output_png = "balance_curve.png"
    plt.savefig(output_png, dpi=150)
    plt.close()
    
    print(f"Balance curve plot generated successfully and saved to {output_png}")

if __name__ == '__main__':
    main()
