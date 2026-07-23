# ICT Trading Bot - Experiment History

A consolidated record of every research phase conducted on the MNQ liquidity-sweep (ICT) strategy, compiled from research logs. Raw result files for each phase are preserved in `archive/results_and_reports/`.

**Current baseline** (the survivor of everything below): FVG + Equilibrium + Breaker confluences, BOS-only reversal confirmation, dealing-range equilibrium bounded by running swing pivots, NY session only, immediate limit entry, walk-forward LightGBM filter with a **6-month chronological training window** (1-year warm-up), **constant 0.25 probability threshold**, 60-point risk cap, split TP (50% at +2R → BE trail, 50% at +4R), fixed 16 MNQ contracts.
**Result (2016–2026 out-of-sample): +17.72% avg annual return, 11.48% avg max DD, +0.96 monthly Sharpe, ~422 trades/yr.** See `docs/baseline/baseline_performance.json`.

---

## Phase 1 - Rules-based parameter sweeps (pre-ML)

Early grid searches with the pure rules engine (`research_harness.py`, results in `archive/results_and_reports/results_research_harness/` and `analysis/leaderboard.csv`). Tested session filters (morning-only, lunch-excluded), stop-loss multipliers (0.90–1.10), liquidity-type filters (daily-only, stacked-only, daily+4H), max setups per pool, and sweep-penetration size filters.

**Verdict: FAILED.** Best configurations reached ~1.08 profit factor with 14–20% win rates and near-zero expectancy; several lost money outright. Conclusion: raw rules could not separate good sweeps from bad ones, leading to the addition of an ML filter.

## Phase 2 - ML pipeline with expanding-window walk-forward

First LightGBM framework (`backtest.py`): expanding walk-forward training window, threshold sweep 0.50–0.80, TP1-only vs split TP. Early sizing study (`evaluate_compounding_regimes` original) compared fixed 16 contracts vs 2% compounding at a 0.60 threshold, year by year 2020–2025.

**Verdict: Partially successful.** Proved the ML filter concept and established split TP as superior, but high thresholds over-filtered and the expanding window went stale. Superseded by rolling windows and lower thresholds.

## Phase 3 - Structural permutation study (2020–2025)

8 permutations of 3 proposed structure changes (`results_permutation_tests.csv`):

| Change | Description | Alone | Best combo |
| --- | --- | --- | --- |
| 1 | BOS-only reversal confirmation (drop inverse-FVG triggers) | 0.39 → 0.45 Sharpe | **1+3: 0.89 Sharpe, +20.0%/yr** |
| 2 | Split `fvg_rejected` into opp/same features | no effect (0.39) | degraded every combo |
| 3 | Dealing-range equilibrium from running swing pivots | 0.39 → **0.73**, DD halved | - |

**Verdict: SUCCESS - Changes 1 & 3 permanently integrated** into `pine_translator.py`, almost doubling Sharpe ratio (pure-ICT baseline was 11.0%/0.39). Change 2 rejected (overfitting/signal dilution).

## Phase 4 - Daily bias & execution timeframe scaling

Three rounds: 4H/1H daily-bias trend features in the model, post-prediction strict bias filtering, and 5m/15m entry-confirmation scaling.

**Verdict: ALL FAILED.** Bias features in the model: Sharpe 0.89 → 0.47 (tree splits on slow-moving features). Post-prediction bias filter: 0.89 → 0.57 (blocks profitable mean-reversion). 5m/15m confirmation gating: collapsed setup count by ~90%, negative Sharpe. NQ's mean-reversion edge requires taking counter-trend sweeps.

## Phase 5 - Confluence study (11 years, 2015–2025)

FVG+EQ baseline vs adding Order Blocks and/or Breaker Blocks (`results_confluences_comparison*.txt`):

| Config | Return | Max DD | Sharpe |
| --- | --- | --- | --- |
| **Baseline + Breaker** 🏆 | +13.28% | 11.98% | **0.67** |
| Baseline (FVG+EQ) | +13.33% | 14.04% | 0.59 |
| Baseline + OB + Breaker | +13.34% | 14.61% | 0.52 |
| Baseline + OB | +12.21% | 16.96% | 0.41 |

**Verdict: Breaker Blocks adopted as default confluence; Order Blocks rejected** (added noise, deeper drawdowns).

Note: I accidentally changed some model features during this phase causing the Sharpe ratio to drop by around 0.2.

## Phase 6 - Sizing & HTF trend filter hypotheses (2015–2025)

On Baseline+Breaker (`results_hypotheses_comparison.txt`): risk-based sizing at 0.5%/1.0% of balance vs fixed 16 contracts; HTF trend filter on/off.

**Verdict: both hypotheses ultimately rejected - fixed 16 contracts, no trend filter.** 0.5% risk sizing was initially promising (stabilized 2016–2017 dramatically, Sharpe 0.64 vs 0.67) but final decision kept fixed sizing in favor of lower drawdown; 1.0% sizing = +19.96%/yr but 29.15% DD, Sharpe 0.50. Trend filter collapsed returns to +5.63% (Sharpe 0.28) by banning mean-reversion entries. *(Note: the 0.5% sizing write-up called it "highly recommended" before the final decision reversed it - worth re-examining if you ever want smoother equity in low-vol years.)*

## Phase 7 - Intraday VIX features (2020–2025)

1-minute VIX futures (VX.c.0) as a single extra model feature, 5 variants (raw close, ratios to 1h/1d/20d/1y MAs). Data preserved in `archive/data/VIX_1M_*` (`results_vix_tests.csv`).

**Verdict: ALL FAILED.** Baseline 0.69 Sharpe beat every VIX variant (best: 20-day ratio at 0.62). ATR-normalized price features already self-adapt to volatility; VIX added regime lag and collinearity.

## Phase 8 - "C1-C3 approved" restoration & 11-year validation

Restored approved config (daily bias off, NY only, FVG+EQ confluences, immediate entries, fixed sizing, 1800-setup training window) and validated 2015–2026 (`results_retest_comparison.txt`; code snapshot in `archive/scripts/backup_c1_c3_approved/`).

**Verdict: became the previous official baseline - +16.0%/yr, 13.1% DD, 0.73 Sharpe**, profitable 8 of 9 out-of-sample years.

## Phase 9 - State-space delayed-entry execution model

Rebuilt the pipeline to monitor setups bar-by-bar up to 15 bars post-sweep, entering when model confidence rose (`results_state_space_comparison.txt`; design notes in the advanced-ML ideas doc).

**Verdict: CATASTROPHIC FAILURE - abandoned.** All thresholds produced negative Sharpe; th=0.25 hit 89.99% max DD. Waiting for confirmation inflates stop distance and turns discount entries into chases; sweep-push resets over-segmented training data.

## Phase 10 - Break-even win-rate gating (2016–2025)

Replaced the fixed probability threshold with a dynamic break-even win-rate gate (required WR = (1+edge)/(Rm+1)) across edge levels, on both immediate and state-space execution (`results_breakeven_gating_comparison.txt`, `results_risk_management_comparison.txt`).

**Verdict: FAILED.** Gating improved raw ML returns vs a naive threshold (1.58% → 9.01%) but every gated variant had negative Sharpe; state-space + tight stops caused position-size blowups. ML models got mixed up again at this stage so the project was reverted to the previous best ML version.

## Phase 11 - Chronological training window sweep ⭐ (2015–2026)

Switched LightGBM training from setup-count windows to time-based windows: 1m/2m/3m/6m/1y (`results_chronological_window_sweep.txt`, report in `baseline_config/`).

| Window | Return | Max DD | Sharpe |
| --- | --- | --- | --- |
| **6 months** 🏆 | **+17.72%** | **11.48%** | **+0.96** |
| 1 year | +14.59% | 11.58% | +0.65 |
| 2 months | +14.33% | 14.00% | +0.73 |
| 3 months | +14.03% | 15.07% | +0.25 |
| 1 month | +3.87% | 28.60% | −0.19 |

**Verdict: SUCCESS - 6-month window adopted; this created the current baseline** (up from 16.0%/0.73). ~220 setups is the sweet spot: enough to generalize, fresh enough to track regime shifts.

## Phase 12 - Dynamic threshold optimization

Per-step threshold optimization maximizing net PnL inside the rolling 6-month training window (`results_dynamic_threshold_optimization.txt`).

**Verdict: FAILED (0.72 vs 0.96 Sharpe).** Protected drawdowns in chop (2017 flipped −7.9% → +5.1%) but stayed too conservative entering trend expansions (2021: +41.1% → +17.0%). Constant 0.25 kept.

## Phase 13 - Constant threshold sweep (0.10–0.45)

Swept the decision threshold in 0.05 steps on the 6-month baseline (`results_constant_threshold_sweep.txt`).

**Verdict: 0.25 confirmed as the mathematical peak** (+17.72%/0.96); clean concave curve - lower thresholds double drawdowns, higher ones filter out winners.

## Phase 14 - Five suggested enhancements sweep (2016–2026)

Time-decay sample weighting, move to BE at +2RR, 3-seed model ensembling, feature interactions, feature pruning - individually and combined (`results_ideas_comparison_report.txt`).

**Verdict: ALL FAILED; baseline remained the most optimal setup (0.80 vs best variant 0.68 on that run's Sharpe accounting). Notable traps: BE at +2RR cuts off entry retests before the big move (Sharpe halved); time-decay overfits the current regime; removing session flags spikes drawdowns.

## Phase 15 - S&P 500 futures (ES) tests

Tested the same model used on NQ on 11 years of ES data (`results_es_grid_sweep.txt`, `results_ES.txt`). ES data retained in `code/data/ES_*`.

**Verdict: FAILED - strategy is NQ-specific.** The NQ baseline config on ES: −1.39%/yr, −0.23 Sharpe. Low-threshold ES configs produced account liquidations (>−100% in 2021/22). Only viable ES config: 1-year window + 0.30 threshold → a weak +3.47%/0.26. Conclusion: trade NQ/MNQ only.

## Phase 16 - GBDT ensemble (XGBoost + LightGBM + CatBoost) - INCOMPLETE

Planned grid of 6 ensemble weightings vs the LightGBM baseline (implementation plan exists in the Antigravity brain folder; task checklist unchecked, no results file found).

**Verdict: never run to completion.** Note Phase 14 already tested a simpler seed-ensemble and it degraded performance - temper expectations if you revisit this.

## Phase 17 - Publicly taught top-down strategy 

Full implementation of a reportedly successful publicly-taught ICT methodology: 1h/4h trends determine bias (4h trumps 1h), wait for high-timeframe liquidity sweeps, then scale down to either 5m or 15m timeframe for execution based on whether the 1h/4h trends match or not. On lower timeframe, look for break of structure, then retrace into OB/FVG/breaker/EQ, and scale down into an even lower timeframe to enter off of another break of structure. Four stop/target interpretations tested, costs included, no parameter optimization (`src/htf_bias_strategy.py`, `HTF_STRATEGY_RESULTS.md`).

**Verdict: FAILED.** All four variants negative (best: −19.5R / −3.6% CAGR net over 11 years). Setup funnel healthy (~4.8k sweeps → 1.2k BOS → 1.1k POI → ~560–640 entries), but the full stack performed indistinguishably from random entries with identical exits, and confirmation entries had a lower win rate than random ones. Confirms Phase 4's conclusion was not an implementation artifact: HTF-bias gating does not produce edge on NQ intraday.

---

## Quick reference: what worked vs what didn't

**Adopted:** BOS-only confirmation + swing-pivot dealing range (Ph. 3), Breaker Block confluence (Ph. 5), NY-session-only + no trend filter (Ph. 4/6), fixed 16 contracts (Ph. 6), 6-month chronological training window (Ph. 11), constant 0.25 threshold (Ph. 12/13), 60-pt risk cap, split TP.

**Rejected/failed:** rules-only trading (Ph. 1), FVG feature splitting (Ph. 3), daily-bias features & filters (Ph. 4), Order Blocks (Ph. 5), risk-based sizing (Ph. 6 - borderline), VIX features (Ph. 7), delayed/state-space entries (Ph. 9), break-even win-rate gating (Ph. 10), dynamic thresholds (Ph. 12), time-decay weighting / BE at +2RR / ensembling / feature interactions / pruning (Ph. 14), porting to ES (Ph. 15).

**Recurring lessons:** the model needs fresh but sufficient data (~6 months / ~220 setups). Anything that delays entry destroys the R:R of discount entries; anything that blocks counter-trend trades kills the NQ mean-reversion edge. Simpler input features beat over-engineered ones on this sample size.
