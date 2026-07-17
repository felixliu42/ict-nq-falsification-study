import pandas as pd
import numpy as np
import os

def generate_demo_dataset(filename="demo_tv_export.csv", n_bars=72000, n_setups=5700):
    np.random.seed(42)
    
    # Input validation
    if n_bars < 100:
        raise ValueError("n_bars must be at least 100 to allow injecting setups with a margin of 50 bars at the beginning and end.")
    if n_setups > (n_bars - 100):
        raise ValueError(f"n_setups ({n_setups}) cannot exceed the available range of bars ({n_bars - 100}).")
        
    print(f"Generating synthetic TradingView export with {n_bars} bars and {n_setups} setups...")
    
    # 1. Generate a realistic random walk for price
    price = 15000.0
    prices = []
    for _ in range(n_bars):
        change = np.random.normal(0.5, 15.0) # upward drift, volatility
        price += change
        prices.append(price)
        
    prices = np.array(prices)
    
    # Open, High, Low, Close
    close_val = prices
    open_val = np.roll(prices, 1)
    open_val[0] = close_val[0] - np.random.normal(0, 5)
    
    high_val = np.maximum(open_val, close_val) + np.abs(np.random.normal(10, 8, n_bars))
    low_val = np.minimum(open_val, close_val) - np.abs(np.random.normal(10, 8, n_bars))
    
    # Time array in milliseconds (5-minute interval start times)
    start_time = 1781700000000 # arbitrary start time
    time_val = [start_time + i * 5 * 60 * 1000 for i in range(n_bars)]
    
    # Construct base dataframe
    df = pd.DataFrame({
        'Time': time_val,
        'Open': open_val,
        'High': high_val,
        'Low': low_val,
        'Close': close_val,
        'Volume': np.random.randint(500, 3000, n_bars)
    })
    
    # 2. Inject indicator features on random bars (about 8% of bars will represent setups)
    # The columns have mock TV indicator suffixes
    suffix = " (MNQ Liquidity Feature Engine (ML Export))"
    df[f'sweep_direction{suffix}'] = np.nan
    df[f'liquidity_type{suffix}'] = np.nan
    df[f'liquidity_strength{suffix}'] = np.nan
    df[f'bos_up_strength{suffix}'] = np.nan
    df[f'bos_down_strength{suffix}'] = np.nan
    df[f'bullish_fvg_rejected{suffix}'] = np.nan
    df[f'bearish_fvg_rejected{suffix}'] = np.nan
    df[f'retracement_depth{suffix}'] = np.nan
    df[f'distance_to_equilibrium{suffix}'] = np.nan
    df[f'time_since_sweep{suffix}'] = np.nan
    
    # Session features
    df[f'ny_session{suffix}'] = np.random.choice([0.0, 1.0], n_bars, p=[0.6, 0.4])
    df[f'london_session{suffix}'] = np.random.choice([0.0, 1.0], n_bars, p=[0.7, 0.3])
    df[f'asian_session{suffix}'] = np.random.choice([0.0, 1.0], n_bars, p=[0.7, 0.3])

    # Inject setups trade sequences spaced out
    setup_bars = np.sort(np.random.choice(range(50, n_bars - 50), n_setups, replace=False))
    
    for bar in setup_bars:
        sweep_dir = np.random.choice([-1.0, 1.0])
        time_since = np.random.randint(1, 10)
        
        # Inject matching features
        df.loc[bar, f'sweep_direction{suffix}'] = sweep_dir
        df.loc[bar, f'liquidity_type{suffix}'] = np.random.choice([1.0, 2.0, 3.0])
        df.loc[bar, f'liquidity_strength{suffix}'] = np.round(np.random.uniform(0.2, 1.0), 2)
        df.loc[bar, f'retracement_depth{suffix}'] = np.round(np.random.uniform(0.1, 0.9), 2)
        df.loc[bar, f'distance_to_equilibrium{suffix}'] = np.round(np.random.uniform(-0.01, 0.01), 6)
        df.loc[bar, f'time_since_sweep{suffix}'] = float(time_since)
        
        # Structure features based on sweep direction
        if sweep_dir == -1.0: # Bearish
            df.loc[bar, f'bos_down_strength{suffix}'] = np.random.choice([0.0, 1.0])
            df.loc[bar, f'bearish_fvg_rejected{suffix}'] = np.random.choice([0.0, 1.0])
        else: # Bullish
            df.loc[bar, f'bos_up_strength{suffix}'] = np.random.choice([0.0, 1.0])
            df.loc[bar, f'bullish_fvg_rejected{suffix}'] = np.random.choice([0.0, 1.0])
            
        # Ensure that the sweep candle extreme is valid (prevent huge gaps)
        sweep_bar = bar - time_since
        if sweep_dir == -1.0:
            # Bearish sweep: make the sweep candle high equal to or higher than current high
            df.loc[sweep_bar, 'High'] = max(df.loc[sweep_bar, 'High'], df.loc[bar, 'High'] + np.random.randint(5, 30))
        else:
            # Bullish sweep: make the sweep candle low equal to or lower than current low
            df.loc[sweep_bar, 'Low'] = min(df.loc[sweep_bar, 'Low'], df.loc[bar, 'Low'] - np.random.randint(5, 30))

    df.to_csv(filename, index=False)
    print(f"Demo TradingView CSV export written to: {filename}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Generate synthetic TradingView export CSV for testing.")
    parser.add_argument('--bars', type=int, default=72000, help='Number of bars to generate (default: 72000)')
    parser.add_argument('--setups', type=int, default=5700, help='Number of trade setup sequences to inject (default: 5700)')
    parser.add_argument('--output', type=str, default="demo_tv_export.csv", help='Output CSV filename (default: demo_tv_export.csv)')
    
    args = parser.parse_args()
    generate_demo_dataset(filename=args.output, n_bars=args.bars, n_setups=args.setups)

