import urllib.request
import json
import os
import pandas as pd
import datetime

# January 1, 2015 (1420070400) to July 10, 2026 (1783641600)
url = "https://query1.finance.yahoo.com/v8/finance/chart/%5EVIX?period1=1420070400&period2=1783641600&interval=1d"

def main():
    print("Requesting daily VIX data from Yahoo Finance API...")
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    req = urllib.request.Request(url, headers=headers)
    
    try:
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode('utf-8'))
            
        chart = data.get("chart", {})
        result = chart.get("result", [{}])[0]
        timestamps = result.get("timestamp", [])
        indicators = result.get("indicators", {})
        quote = indicators.get("quote", [{}])[0]
        closes = quote.get("close", [])
        
        if not timestamps or not closes:
            print("Error: Could not retrieve timestamps or closes from Yahoo API response.")
            return
            
        dates = []
        clean_closes = []
        
        for ts, close in zip(timestamps, closes):
            if ts is None or close is None:
                continue
            date_str = datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc).strftime("%Y-%m-%d")
            dates.append(date_str)
            clean_closes.append(round(close, 4))
            
        df = pd.DataFrame({
            "Date": dates,
            "VIX_Close": clean_closes
        })
        
        # Sort and deduplicate
        df = df.sort_values("Date").drop_duplicates(subset=["Date"]).reset_index(drop=True)
        
        os.makedirs("data", exist_ok=True)
        output_path = "data/vix_daily.csv"
        df.to_csv(output_path, index=False)
        print(f"Successfully downloaded {len(df)} days of VIX data. Saved to {output_path}")
        
    except Exception as e:
        print(f"Error fetching VIX data: {e}")

if __name__ == '__main__':
    main()
