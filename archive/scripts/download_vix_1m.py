import os
import sys
import pandas as pd
import databento as db

API_KEY = os.environ.get("DATABENTO_API_KEY", "")
# Databento CFE data starts on 2018-11-04
YEAR_RANGES = [
    (2018, "2018-11-04", "2019-06-25"),
    (2019, "2019-06-25", "2020-06-25"),
    (2020, "2020-06-25", "2021-06-25"),
    (2021, "2021-06-25", "2022-06-25"),
    (2022, "2022-06-25", "2023-06-25"),
    (2023, "2023-06-25", "2024-06-25"),
    (2024, "2024-06-25", "2025-06-25"),
    (2025, "2025-06-25", "2026-06-25")
]

def download_vix_year(year, start_date, end_date, output_path):
    print(f"Downloading VIX 1m futures for {year} ({start_date} to {end_date})...")
    try:
        client = db.Historical(key=API_KEY)
        data = client.timeseries.get_range(
            dataset="XCBF.PITCH",
            schema="ohlcv-1m",
            symbols=["VX.c.0"],
            stype_in="continuous",
            start=start_date,
            end=end_date,
        )
        df = data.to_df()
        
        if df.empty:
            print(f"Warning: No data returned for VIX in {year}")
            return False
            
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
        print(f"Successfully saved VIX raw data to {output_path}")
        return True
    except Exception as e:
        print(f"Error downloading VIX for {year}: {e}", file=sys.stderr)
        return False

def main():
    print("Starting 1m VIX Futures Data Download...")
    for year, start, end in YEAR_RANGES:
        out_path = f"data/VIX_1M_{year}/raw_data.csv"
        if os.path.exists(out_path):
            print(f"VIX 1m data for {year} already exists locally.")
            continue
        download_vix_year(year, start, end, out_path)
    print("VIX Futures data download process finished.")

if __name__ == '__main__':
    main()
