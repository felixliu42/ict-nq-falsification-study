import json
import os
from pathlib import Path

BASELINE_JSON = Path(__file__).resolve().parent.parent / "docs" / "baseline" / "baseline_performance.json"

def load_baseline():
    """
    Load the saved baseline record (the 17.72% / 0.96 avg-yearly-Sharpe model
    as originally reported — see docs/VALIDATION_RESULTS.md for the corrected,
    cost-adjusted view of these numbers).
    """
    with open(BASELINE_JSON, "r") as f:
        return json.load(f)

if __name__ == "__main__":
    data = load_baseline()
    print(f"Loaded Baseline: {data['name']}")
    print(f"Avg Sharpe: {data['results']['average']['sharpe']}")
    print(f"Avg Return: {data['results']['average']['return_pct']}%")
