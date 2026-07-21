# Validation Test Results — Is the Edge Real?

Five tests run on the full 11-year MNQ dataset (39,079 setups, 3.79M bars), using the exact baseline pipeline (180-day chronological walk-forward, step 5, 1-year warm-up, 0.25 threshold, NY session, 60-pt cap, split TP, $32/pt exposure).

**Methodology caveat (important):** LightGBM could not be installed in the analysis sandbox (network restrictions), so the walk-forward was run with a purpose-built C implementation of the same algorithm class — histogram GBDT, leaf-wise growth, identical hyperparameters (100 trees, lr 0.05, depth 4, 15 leaves, min_child 3), deterministic, validated on synthetic data. It is not bit-identical to LightGBM. What that substitution revealed is itself Finding #0.

## Finding 0 — The edge is fragile to classifier implementation

The equivalent-spec GBDT, trained on identical data with the identical walk-forward, produced a materially different trade set and yearly pattern (e.g. 2019: reported +66.2%, substitute −11.7%; 2022: +39.2% vs +0.3%). At matched selectivity (~470 trades/yr, same as the baseline), it captured only **~4.5%/yr gross vs the reported 17.7%**. A robust edge should survive swapping one GBDT implementation for an equivalently-parameterized one. This is consistent with your own Phase 14 result where averaging 3 random seeds *degraded* Sharpe from 0.80 to 0.59. A large share of the reported 17.72%/0.96 appears specific to LightGBM's particular tie-breaking on this data, i.e. luck, not signal. The ranking is not pure noise — gross return rises monotonically with threshold (4.5% at 0.15 → 8.5% at 0.25) — but the magnitude is not reproducible.

## Test 1 — Transaction costs (NQ, 1.6 contracts equivalent)

Cost model: $4.60 all-in round-trip commission per NQ contract; 1 tick ($5) slippage on stop-side exits (scenario A); scenario B adds 1-tick entry slippage and 2-tick stops.

| | $/trade | CAGR | Pooled Sharpe | Max DD | Calmar |
| --- | --- | --- | --- | --- | --- |
| Gross (as backtested) | $37.00 | 7.5% | 0.59 | 24.1% | 0.31 |
| Net A (realistic-optimistic) | $23.68 | 5.4% | 0.40 | 33.3% | 0.16 |
| Net B (pessimistic) | $9.72 | 2.5% | 0.22 | 45.4% | 0.06 |

Costs consume **36–74% of the gross edge**. Applied to the reported LightGBM numbers ($42/trade gross), the same per-trade drag implies roughly 12%/yr net A, 6–7% net B — before the selection-bias haircut below.

### Test 1b — Broker-tier sensitivity (does the conclusion depend on the cost assumption?)

The report's headline cost assumption ($4.60 round-turn per NQ-equivalent contract) is a modeled figure. Because the entire conclusion hinges on costs, the table below re-runs the baseline configuration across published commission schedules (July 2026), holding slippage fixed at one tick on stop-side exits. Exchange + clearing fees (~$1.60/contract round-turn for E-minis, ~$0.55 for Micros) are unavoidable at any broker and are included in every row.

| Venue / broker (round-turn) | Cost/trade | Net $/trade | CAGR | Sharpe |
| --- | --- | --- | --- | --- |
| Theoretical zero cost (bound) | $0.00 | $31.04 | 6.60% | 0.50 |
| NQ ×1.6 — NinjaTrader Lifetime ($0.09/side) | $2.85 | $28.19 | 6.13% | 0.46 |
| NQ ×1.6 — Webull (~$0.70/contract) | $4.80 | $26.24 | 5.80% | 0.44 |
| NQ ×1.6 — Tradovate ($1.29/side) | $6.69 | $24.35 | 5.47% | 0.41 |
| **NQ ×1.6 — report assumption** | **$7.36** | **$23.68** | **5.35%** | **0.40** |
| MNQ ×16 — NinjaTrader Lifetime | $11.68 | $19.36 | 4.54% | 0.34 |
| MNQ ×16 — Webull | $16.80 | $14.24 | 3.50% | 0.28 |
| MNQ ×16 — Tradovate | $21.28 | $9.76 | 2.51% | 0.22 |

Two conclusions. First, **the finding is not an artifact of the cost assumption**: even at literally zero commission the gross-to-net gap is modest (0.59 → 0.50 Sharpe), and every realistic tier lands between 0.22 and 0.46 — all far below the reported 0.96, and all still subject to the selection-bias correction in Test 3, which roughly halves them again. Second, **contract choice matters more than broker choice**: the original strategy specified 16 MNQ micros, which cost 2–3× more than the equivalent 1.6 NQ minis for identical exposure, because per-contract fees dominate. Sizing in minis rather than micros is the single largest cost improvement available, worth more than any broker switch.

*Sources: [CME clearing fees](https://www.cmegroup.com/company/clearing-fees.html), [Tradovate pricing](https://www.tradovate.com/pricing/), [NinjaTrader pricing](https://ninjatrader.com/pricing/), [Webull futures fees (BrokerChooser)](https://brokerchooser.com/broker-reviews/webull-review/micro-emini-nasdaq100-futures-fees). Commission schedules change; exchange fees vary with CME membership status and volume tier.*

## Test 2 — Honest metric conventions

The reported 0.96 "Sharpe" is an average of per-year Sharpes (12 monthly points each), carried by 2019's +4.02. Pooling all months: **gross pooled Sharpe ≈ 0.59–0.62** (substitute run and reported yearly table agree). The reported 11.48% "max DD" is an average of yearly DDs; the full-period max DD is 24% gross / 33% net (reported table's own 2018 shows 36%). Honest **Calmar ≈ 0.16–0.31**, not ~1.5.

## Test 3 — Nested walk-forward (selection bias measured)

Config (window × threshold, 12-cell grid) re-selected each year using only prior data, evaluated on the following year, net of costs (A):

| | Pooled Sharpe | CAGR | Max DD | Calmar |
| --- | --- | --- | --- | --- |
| Fixed 180d/0.25 chosen in hindsight | 0.41 | 5.4% | 33% | 0.16 |
| Honest nested selection (2018–2026) | **0.19** | **1.8%** | 32% | 0.06 |

Roughly **half the after-cost performance is configuration selection bias**. The nested picks lost money in 2024, 2025, and 2026 H1 consecutively.

## Test 4 — Data integrity: CLEAN ✅

No duplicate/non-monotonic timestamps. 214 close-to-open jumps >0.3% in 11 years — mostly weekend/session breaks and roll weeks (unadjusted continuous contract), plus real events (Mar 2020, CPI days). Trades within 24h of a >0.5% jump: 3.1% of trades, 1% of PnL. Data quality is not driving the results.

## Test 5 — Statistical significance

Block bootstrap on net-A monthly returns: pooled Sharpe 0.40, **90% CI [−0.14, +0.90], P(Sharpe ≤ 0) ≈ 11%**. Trade-level block bootstrap of the drawdown distribution at current sizing: median max DD ≈ 39% per decade, 95th percentile ≈ 89% (meaningful ruin risk at 16 MNQ flat on $100k).

## Recent regime

2023 → mid-2026 net of costs: **CAGR −3.6%, Sharpe ≈ 0.0** (substitute run). The reported LightGBM numbers say the same thing: 2023–2026 average +4.3% gross ≈ ~0–1% net. Whatever edge existed was concentrated in the 2019–2022 volatility expansion and has not been present for 3.5 years.

## Bottom line

After costs, selection-bias correction, and implementation-robustness testing, the strategy's expected performance is statistically indistinguishable from zero, with material drawdown risk. The feature set contains real information (monotonic threshold-return relationship), but at 1-minute scalping frequency the per-trade edge (~$23–37 gross on $32/pt) is too thin relative to fixed per-trade costs, and the recent regime shows no net edge at all.
