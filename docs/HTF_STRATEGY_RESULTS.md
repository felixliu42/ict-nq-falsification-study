# HTF Bias Top-Down Strategy — Implementation & Results

Implementation of a strategy specification transcribed from a publicly-taught ICT methodology (third-party trading educator). Provenance note: because the rules were fixed by an external source before testing, there were no researcher degrees of freedom in the rule design — only stop/target placement (unspecified by the source) required interpretation, and all four defensible interpretations were tested.

The specification: daily bias from 1h/4h BOS-trend (4h trumps 1h; agree → execute 5m, oppose → 15m), wait for a bias-matching HTF liquidity sweep (1h/4h pivot levels, previous-day and session highs/lows), then execution-TF BOS, then retracement into a POI (order block, FVG, breaker, equilibrium), then scale-down LTF BOS entry (5m→1m, 15m→5m). Engine: `src/htf_bias_strategy.py`. All parameters fixed a priori; NQ costs (commission + slippage) and 1%-risk sizing included from the first run; exits simulated on 1-minute bars.

## Results (2015–2026, ~3.8M bars)

Funnel (healthy at every stage — this is not a starved or broken pipeline): 4,839 HTF sweeps → 1,212 BOS confirmations → 1,069 POI touches → ~560–637 entries (~55/yr) depending on variant.

Since the spec leaves stop placement and targets open, all four defensible interpretations were run — none was cherry-picked:

| Stop | Target | Trades | Win rate | Sum R | Net Sharpe | Net CAGR | Max DD |
| --- | --- | --- | --- | --- | --- | --- | --- |
| retracement low | split 2R/BE/4R | 631 | 31.1% | −43.6 | −0.36 | −10.3% | 77% |
| retracement low | opposing HTF draw | 637 | 26.5% | −111.2 | −0.24 | ruin | 142% |
| sweep extreme | split 2R/BE/4R | 560 | 33.0% | −19.5 | −0.20 | −3.6% | 52% |
| sweep extreme | opposing HTF draw | 582 | 32.5% | −58.9 | −0.38 | −11.1% | 78% |

The cost hypothesis from the validation report was confirmed: at ~55 trades/yr with ~32-pt average risk, costs are only ~4% of risk per trade — costs are no longer the problem. The raw signal is.

## Autopsy — why it loses

**The setup stack performs like random entries.** Random entries with the identical exit engine and matched R distribution scored a mean of −5.7R across 5 seeds (range −44 to +54); the strategy's −19.5R sits inside that noise band. The full chain (bias → sweep → BOS → POI → LTF confirmation) adds no measurable selection edge over random timing on this instrument. The confirmation entries actually had a *lower* win rate (33.0%) than random ones (35.1%) — consistent with the project's oldest recurring lesson: every bar spent waiting for confirmation is paid for in entry price.

Decomposition: longs −5.2R, shorts −14.3R (both negative); bias-agree 5m stream −8.5R, bias-oppose 15m stream −11.0R (both negative). No component is carrying hidden edge. The only positive slice is NY-session entries (+14.4R vs −33.9R off-session, ~1.3R/yr) — noted because the old baseline was always NY-only, but at ~349 trades this is statistically indistinguishable from zero and must not be treated as a finding.

**No sensitivity grid was run to "rescue" the result.** With every variant deeply negative and the random benchmark matched, searching pivot widths and window lengths for a positive cell would only manufacture selection bias (see VALIDATION_RESULTS.md).

## Conclusion

The previous failure of daily-bias approaches (Phase 4) was not merely bad implementation. A native, faithful, cost-aware top-down implementation with a healthy setup funnel also has no positive expectancy on 11 years of NQ data. Combined with prior findings — trend filters block NQ's mean-reversion edge, confirmation delay destroys R:R — the evidence is now consistent across three independent implementations: **HTF-bias-gated intraday entries on NQ do not work as specified.**

Caveats for fairness: trend was defined as BOS-of-confirmed-fractal-pivots (a standard but not unique choice); levels were 1h/4h pivots + day/session extremes; other definitions (e.g., displacement-based trend, weekly levels, killzone-restricted entries) were not tested and remain open — but each would need to be pre-registered, run once, and accepted as-is to mean anything.
