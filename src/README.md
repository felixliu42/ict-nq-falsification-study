# MNQ Liquidity Feature Engine (ML Export)

This repository contains a pure **feature generation engine** built in TradingView Pine Script v6, along with an offline **Python Dataset Pipeline** to process, label, and prepare the generated features for machine learning models (e.g., LightGBM, XGBoost, or Neural Networks).

**CRITICAL RULE**: This script is a pure math engine. It contains **no trading logic, strategy orders, or visual dashboards**, and it has **zero future leakage/repainting**. All calculations are strictly bar-by-bar using historical offsets.

---

## Part 1: Pine Script Features (14 Outputs when `valid_setup == 1`)

To prevent noise, ML features are only output in the Data Window (CSV) and alerts when `valid_setup == 1`. On all other bars, they plot as `na` (nulls).

| Feature Name | Type | Value Range | Description |
|---|---|---|---|
| **`sweep_direction`** | Integer | `-1` or `+1` | Setup bias: `-1` (bearish, buy-side swept), `+1` (bullish, sell-side swept). |
| **`liquidity_type`** | Integer | `1`, `2`, `3` | Type of liquidity swept: `1` (Daily session / PDH/PDL), `2` (1H swing pivot), `3` (4H swing pivot). |
| **`liquidity_strength`** | Float | $[0.0, 1.0]$ | Volatility-adjusted stacked clustering strength of the swept pool. |
| **`bos_down_strength`** | Float | `0.0`, `1.0`, or `null` | `1.0` if a bearish Break of Structure occurred on this bar (only when `sweep_direction = -1`). |
| **`bearish_fvg_rejected`** | Float | `0.0`, `1.0`, or `null` | `1.0` if a bearish Fair Value Gap rejection occurred on this bar (only when `sweep_direction = -1`). |
| **`bearish_displacement_size`** | Float | $[0.0, \infty)$ or `null` | Size of bearish candle body normalized by ATR: $\frac{\text{open} - \text{close}}{\text{ATR}(14)}$ (only when `sweep_direction = -1`). |
| **`bos_up_strength`** | Float | `0.0`, `1.0`, or `null` | `1.0` if a bullish Break of Structure occurred on this bar (only when `sweep_direction = +1`). |
| **`bullish_fvg_rejected`** | Float | `0.0`, `1.0`, or `null` | `1.0` if a bullish Fair Value Gap rejection occurred on this bar (only when `sweep_direction = +1`). |
| **`bullish_displacement_size`** | Float | $[0.0, \infty)$ or `null` | Size of bullish candle body normalized by ATR: $\frac{\text{close} - \text{open}}{\text{ATR}(14)}$ (only when `sweep_direction = +1`). |
| **`retracement_depth`** | Float | $[0.0, 1.0]$ | Fibonacci retracement level of the close relative to the impulse dealing range. |
| **`distance_to_equilibrium`** | Float | $(-\infty, \infty)$ | Volatility-normalized distance to midpoint, signed in the sweep direction: $\text{sweep\_direction} \times \frac{\text{close} - \text{equilibrium}}{\text{close}}$. |
| **`time_since_sweep`** | Integer | $[0, \infty)$ | Number of bars elapsed since the sweep candle. |
| **`ny_session`** | Binary | `0` or `1` | 1 if current bar is in New York hours (`08:00 - 17:00` Eastern Time). |
| **`london_session`** | Binary | `0` or `1` | 1 if current bar is in London hours (`03:00 - 12:00` Eastern Time). |
| **`asian_session`** | Binary | `0` or `1` | 1 if current bar is in Asian hours (`19:00 - 03:00` Eastern Time). |

---

## Part 2: Python Dataset Pipeline (`pipeline.py`)

The `pipeline.py` script processes historical CSV files exported from TradingView. It segments the data into trade sequences, extracts features, and labels each trade sequence based on a strict risk-reward outcome.

### Setup Ingestion & Invalidation Logic
1. **Entry Identification**: Identifies the exact bar where `valid_setup` transitions to `1`.
2. **Stop Loss (SL)**: Set at the absolute extreme high/low of the sweep candle (`sweep_extreme` located `time_since_sweep` bars ago).
3. **Target Price (+2R)**: Set at $2.0 \times \text{Risk}$ in the direction of the setup:
   - Bullish: $\text{Target} = \text{Entry Price} + 2.0 \times (\text{Entry Price} - \text{SL})$
   - Bearish: $\text{Target} = \text{Entry Price} - 2.0 \times (\text{SL} - \text{Entry Price})$
4. **Labeling Rule (`label`)**:
   - `y = 1` (success) if price touches Target before touching SL.
   - `y = 0` (failure) if price touches SL first, touches both on the same bar (dual touch), or if the historical data ends before either is reached.
5. **Excursion Metrics**:
   - `max_favorable_excursion`: Max price reached in the trade direction before exit, normalized by risk ($R$).
   - `max_adverse_excursion`: Max price reached against the trade direction before exit, normalized by risk ($R$).
   - `time_to_target`: Number of bars elapsed between the entry bar and the exit bar.

---

## How to Run the Pipeline

### 1. Export CSV from TradingView
1. Load `mnq_liquidity_feature_engine.pine` on a TradingView chart (e.g. 5m chart).
2. Open the **Data Window** (right-hand sidebar) -> click the three dots -> select **Export Chart Data**.
3. Save the CSV file (e.g. `mnq_chart_data.csv`).

### 2. Run the Pipeline Script
Run the pipeline in your terminal, specifying the input and output paths:
```bash
python pipeline.py --input mnq_chart_data.csv --output mnq_ml_ready.csv
```
This generates a clean training dataset `mnq_ml_ready.csv` containing only setup entry rows with their features and $+2\text{R}/-1\text{R}$ labels.

---

## Developer Tests

To verify pipeline mathematical logic, run the test suite:
```bash
python test_pipeline.py
```
This runs a simulated trade sequence test checking for target hits, stop-outs, MFE/MAE excursions, and feature extractions, asserting expected outputs.

---

## Part 3: Model Training (`train_model.py`)

The `train_model.py` script trains a LightGBM classifier on the processed setup features to predict the probability of success for each completed trade sequence:
$$P(\text{success} \mid \text{complete sweep} \to \text{structure validation sequence})$$

### Execution & Outputs
Run the trainer by pointing to the processed dataset:
```bash
py train_model.py --dataset mnq_ml_dataset.csv --outdir .
```
This trains the model using a chronological train-test split (80% train, 20% test) to prevent data leakage and outputs:
1. **`feature_importance.png`**: Split-count feature importances.
2. **`roc_curve.png`**: ROC curve containing the Area Under the Curve (AUC) score.
3. **`calibration_curve.png`**: Calibration reliability diagram.

---

## Part 4: Backtesting Engine (`backtest.py`)

The backtesting engine executes simulated trades *only* on valid structured sequences filtered by out-of-sample LightGBM probabilities.

### Walk-Forward Validation
To guarantee that the backtest reflects live trading conditions, `backtest.py` implements an expanding window walk-forward validation:
1. **Initial Window**: Train the LightGBM model on the first $W$ trade setups (default: `30`).
2. **Predict Fold**: Predict out-of-sample probabilities for the next $N$ setups (default: `5`).
3. **Expand & Repeat**: Retrain the model on all historical setups and repeat until all setups are predicted.

All trade evaluations are strictly conducted on out-of-sample data.

### Trade Logic & Exits
For each trade where `pred_prob > threshold` (sweeping thresholds from `0.50` to `0.80`), the engine scans the raw price data bar-by-bar starting from the bar *after* the entry bar:
* **TP1 Only**: 100% of position is exited at +2R. Returns: `+2.0R` (TP hit) or `-1.0R` (Stop hit).
* **Split TP1/TP2**: 50% exited at +2R (TP1), and 50% exited at +4R (TP2). Once TP1 is hit, the stop loss for the remaining 50% is trailed to the entry price (break-even). Returns:
  - Hits Stop first: `-1.0R`
  - Hits TP1, then hits trailed stop (break-even): `+1.0R` (50% at +2R, 50% at 0R)
  - Hits TP1, then hits TP2: `+3.0R` (50% at +2R, 50% at +4R)

### Execution
Run the backtester:
```bash
py backtest.py --dataset demo_ml_dataset.csv --raw demo_tv_export.csv --outdir .
```

### Outputs
- **Consolidated Results Table**: Displays win rate, profit factor, expectancy (in R-multiples), and max drawdown (in R-multiples) for all thresholds and exit configurations.
- **Per-Session Breakdown Table**: Breakdown of trades, win rate, and expectancy for New York, London, and Asian sessions.
- **`backtest_equity_curve.png`**: Cumulative equity curves plotted for all thresholds.

---

## Part 5: Optimization & Regime Stability Framework (`optimize.py`)

The `optimize.py` script stress-tests the system by classifying market regimes (Trend vs. Chop) and tracking feature stability across walk-forward folds to prevent overfitting.

### 1. Regime Detection (Choppiness Index)
The script calculates the mathematical **Choppiness Index** on raw bar-by-bar price data (typically over a 30-bar lookback):
- Choppiness below the median value across the dataset is labeled **Trend**.
- Choppiness at or above the median value is labeled **Chop**.

### 2. Overfitting & Feature Stability Metrics
The out-of-sample data is split into $F$ chronological folds (default: `3`). The script trains models on preceding data at each fold and calculates the **Coefficient of Variation (CV)** for each feature's split count:
* **High Stability (CV < 0.40)**: The feature's importance remains consistent across different market periods (robust, key indicator).
* **Moderate Stability (CV 0.40 - 0.80)**: The feature is useful but exhibits some regime sensitivity.
* **Unstable (CV > 0.80)**: The feature is highly dependent on a specific historical period (high risk of overfitting, e.g. session-specific artifacts).

### 3. Execution
Run the optimizer:
```bash
py optimize.py --dataset demo_ml_dataset.csv --raw demo_tv_export.csv --outdir .
```

### 4. Outputs
- **Regime Optimization Table**: Shows expectancy under Trend vs. Chop vs. Combined conditions across all thresholds and exit modes.
- **Feature Stability Table**: Lists each feature's split counts per fold, mean importance, CV, and stability classification.
- **Session Sensitivity Breakdown**: Session performance grouped by Trend vs. Chop for the optimal configuration.
- **`optimization_results.png`**: Multi-panel plot containing:
  1. Threshold vs. Expectancy for TP1 Only.
  2. Threshold vs. Expectancy for Split TP.
  3. Feature Importance stability curves across chronological folds.
  4. Cumulative out-of-sample equity curves for the best configuration split by Trend, Chop, and Combined.


