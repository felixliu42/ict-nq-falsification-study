"""
This script downloads historical 1-minute OHLCV futures data for the MNQ continuous contract
from Databento, formats it, and saves it to a CSV file.

Obtaining the Databento API Key:
-------------------------------+
1. Go to the Databento Portal: https://databento.com
2. Log in to your account.
3. Navigate to "API Keys" in the sidebar/portal menu.
4. Copy your API key (it starts with "db-").

Setting the API Key:
--------------------+
There are two ways to configure the API key for this script:
1. Environment Variable (Recommended):
   Set the environment variable DATABENTO_API_KEY. On Windows, you can do this in PowerShell:
       $env:DATABENTO_API_KEY="your-api-key"
   Or in Command Prompt:
       set DATABENTO_API_KEY=your-api-key
2. Direct script modification:
   Replace the placeholder value of `DATABENTO_API_KEY` in the script below:
       API_KEY = "your-api-key"
"""

import os
import sys
import argparse
import pandas as pd
import databento as db

# Attempt to import specific exception classes from Databento
try:
    from databento.common.exceptions import BentoClientError, BentoServerError, BentoError
except ImportError:
    BentoClientError = getattr(db, 'BentoClientError', Exception)
    BentoServerError = getattr(db, 'BentoServerError', Exception)
    BentoError = getattr(db, 'BentoError', Exception)

# Attempt to import requests exception classes for network errors
try:
    import requests
    RequestException = requests.exceptions.RequestException
    TimeoutException = requests.exceptions.Timeout
except ImportError:
    RequestException = Exception
    TimeoutException = Exception

# API Key Placeholder (replace with your key if not using the DATABENTO_API_KEY environment variable)
API_KEY_PLACEHOLDER = "your-api-key"

def get_api_key():
    """
    Retrieve the Databento API key from the environment variable or the placeholder.
    """
    key = os.environ.get("DATABENTO_API_KEY")
    if key:
        return key.strip()
    
    if API_KEY_PLACEHOLDER and API_KEY_PLACEHOLDER != "YOUR_API_KEY":
        return API_KEY_PLACEHOLDER.strip()
        
    return None

def main():
    parser = argparse.ArgumentParser(
        description="Download historical 1-minute OHLCV data for MNQ continuous contract from Databento."
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2025-06-25",
        help="Start date in YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS format (default: 2025-06-25)"
    )
    parser.add_argument(
        "--end",
        type=str,
        default="2026-06-25",
        help="End date in YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS format (default: 2026-06-25)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="mnq_raw_data.csv",
        help="Output CSV file path (default: mnq_raw_data.csv)"
    )
    
    args = parser.parse_args()
    
    # 1. Retrieve API key
    api_key = get_api_key()
    if not api_key:
        print("Error: Databento API key is missing.", file=sys.stderr)
        print("Please obtain an API key from https://databento.com and set the environment variable:", file=sys.stderr)
        print("  On Windows PowerShell: $env:DATABENTO_API_KEY=\"db-...\"", file=sys.stderr)
        print("  On Command Prompt:     set DATABENTO_API_KEY=db-...", file=sys.stderr)
        print("Or set it directly in the script by replacing API_KEY_PLACEHOLDER.", file=sys.stderr)
        sys.exit(1)
        
    print("API key loaded successfully.")
    print(f"Querying Databento GLBX.MDP3 dataset for MNQ.c.0...")
    print(f"Start:  {args.start}")
    print(f"End:    {args.end}")
    
    # 2. Query Databento Historical Client
    try:
        client = db.Historical(key=api_key)
        
        # Download 1-minute OHLCV data for continuous contract MNQ.c.0
        # dataset: GLBX.MDP3
        # schema: ohlcv-1m
        # stype_in: continuous (since continuous contract symbol is used)
        data = client.timeseries.get_range(
            dataset="GLBX.MDP3",
            schema="ohlcv-1m",
            symbols=["MNQ.c.0"],
            stype_in="continuous",
            start=args.start,
            end=args.end,
        )
        
        # Convert Databento Binary Encoding (DBN) data store to Pandas DataFrame
        df = data.to_df()
        
    except BentoClientError as e:
        print(f"Databento Client Error: {e}", file=sys.stderr)
        print("Please check that the requested symbol, dates, dataset, or API key are correct.", file=sys.stderr)
        sys.exit(1)
    except BentoServerError as e:
        print(f"Databento Server Error: {e}", file=sys.stderr)
        print("An error occurred on Databento servers. Please try again later.", file=sys.stderr)
        sys.exit(1)
    except BentoError as e:
        print(f"Databento API Error: {e}", file=sys.stderr)
        sys.exit(1)
    except TimeoutException as e:
        print(f"Network Timeout: {e}", file=sys.stderr)
        print("The request to Databento timed out. Please check your internet connection.", file=sys.stderr)
        sys.exit(1)
    except RequestException as e:
        print(f"Network Connection Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        err_msg = str(e).lower()
        if "timeout" in err_msg or "timed out" in err_msg:
            print(f"Network Timeout: {e}", file=sys.stderr)
        elif "connection" in err_msg or "connect" in err_msg:
            print(f"Network Connection Error: {e}", file=sys.stderr)
        else:
            print(f"An unexpected error occurred during the Databento request: {e}", file=sys.stderr)
        sys.exit(1)
        
    # 3. Check for empty dataframe
    if df.empty:
        print("Warning: The request returned no data. An empty CSV will be created.", file=sys.stderr)
        df_output = pd.DataFrame(columns=["Time", "Open", "High", "Low", "Close", "Volume"])
    else:
        print(f"Successfully retrieved {len(df)} records from Databento.")
        
        # 4. Convert timestamp to millisecond Unix epoch integers named 'Time'
        # By default, to_df() returns DatetimeIndex named 'ts_event' (represented in nanoseconds since epoch)
        if "ts_event" in df.columns:
            ts_series = df["ts_event"]
        else:
            ts_series = df.index
            
        # Convert ts_event to datetime, get epoch in nanoseconds, convert to milliseconds via floor division
        df["Time"] = pd.to_datetime(ts_series).astype("int64") // 1_000_000
        
        # 5. Rename price columns to Open, High, Low, Close, Volume
        df = df.rename(columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume"
        })
        
        # Keep only required columns and match order expected by the pipeline
        output_cols = ["Time", "Open", "High", "Low", "Close", "Volume"]
        
        # Ensure all columns are present
        missing_cols = [c for c in output_cols if c not in df.columns]
        if missing_cols:
            print(f"Error: Required columns {missing_cols} are missing from the retrieved data.", file=sys.stderr)
            sys.exit(1)
            
        df_output = df[output_cols]
        
    # 6. Save formatted DataFrame to CSV
    try:
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            
        df_output.to_csv(args.output, index=False)
        print(f"Data saved successfully to: {args.output}")
        print(f"Format summary:\n{df_output.head(3)}")
    except Exception as e:
        print(f"Error saving output file: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
