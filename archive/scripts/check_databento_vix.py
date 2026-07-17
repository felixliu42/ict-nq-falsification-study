import os
import databento as db
import sys

API_KEY = os.environ.get("DATABENTO_API_KEY", "")

def main():
    print("Testing VX.c.0 query on XCBF.PITCH...")
    try:
        client = db.Historical(key=API_KEY)
        
        # Test query for a single day: 2026-06-01
        data = client.timeseries.get_range(
            dataset="XCBF.PITCH",
            schema="ohlcv-1m",
            symbols=["VX.c.0"],
            stype_in="continuous",
            start="2026-06-01",
            end="2026-06-02",
        )
        df = data.to_df()
        if not df.empty:
            print(f"Success! Retrieved {len(df)} bars of 1-minute VIX futures data.")
            print(df.head())
        else:
            print("Query returned empty dataframe.")
            
    except Exception as e:
        print(f"Error querying VIX futures: {e}", file=sys.stderr)

if __name__ == '__main__':
    main()
