# HTF Bias Top-Down Strategy - Implementation & Results

Implementation of a strategy specification transcribed from a publicly-taught ICT methodology, with tests featuring four combinations of stop-loss and take-profit placement.

The strategy: Determine 1h and 4h timeframe trends using break of structure, and then wait for price to sweep a 1h/4h/session extreme that could result in a reversal in the bias direction. If 1h and 4h timeframes agree, scale down to 5min, otherwise scale down to 15min. Look for break of structure and then retracement into FVG/OB/EQ/Breaker on the lower timeframe and then scale down to 1min timeframe to enter off of break of structure in the bias direction.

## Results (2015–2026, ~3.8M bars)

| Stop | Target | Trades | Win rate | Sum R | Net Sharpe | Net CAGR | Max DD |
| --- | --- | --- | --- | --- | --- | --- | --- |
| retracement low | split 2R/BE/4R | 631 | 31.1% | −43.6 | −0.36 | −10.3% | 77% |
| retracement low | opposing HTF draw | 637 | 26.5% | −111.2 | −0.24 | ruin | 142% |
| sweep extreme | split 2R/BE/4R | 560 | 33.0% | −19.5 | −0.20 | −3.6% | 52% |
| sweep extreme | opposing HTF draw | 582 | 32.5% | −58.9 | −0.38 | −11.1% | 78% |

At ~55 trades/yr with ~32-pt average risk, costs are only ~4% of risk per trade, showing that the signal simply has no edge rather than being bogged down by costs.

Why it loses:

**The setup stack performs like random entries.** Random entries with the identical exit engine and matched R distribution scored a mean of −5.7R across 5 seeds (range −44 to +54); the strategy's −19.5R sits inside that noise band. The full chain (bias → sweep → BOS → POI → LTF confirmation) adds no measurable selection edge over random timing on this instrument. The confirmation entries actually had a *lower* win rate (33.0%) than random ones (35.1%) - consistent with the project's oldest recurring lesson: every bar spent waiting for confirmation is paid for in entry price.

Decomposition: longs −5.2R, shorts −14.3R (both negative); HTF bias-agree 5m trades −8.5R, HTF bias-oppose 15m trades −11.0R (both negative). No component is carrying hidden edge. The only positive slice is NY-session entries (+14.4R vs −33.9R off-session, ~1.3R/yr) - noted because the old baseline was always NY-only, but at ~349 trades this is statistically indistinguishable from zero and must not be treated as a finding.

## Conclusion

The previous failure of daily-bias approaches (Phase 4) was not merely bad implementation. A native, faithful, cost-aware top-down implementation of a popular ICT playbook also has no positive expectancy on 11 years of NQ data. Data supports prior findings - trend filters block NQ's mean-reversion edge, and confirmation delay destroys R:R.
