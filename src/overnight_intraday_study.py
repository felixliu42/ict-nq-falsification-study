"""
Overnight vs. Intraday Returns in Equity Index Futures (NQ & ES, 2015-2026)

Replication of the documented overnight-return effect (Cliff/Cooper/Gulen 2008;
Lou/Polk/Skouras 2019 JFE) on tradeable futures prices.

PRE-REGISTERED DESIGN — fixed before computing any results, no parameters swept:
- Overnight leg: 16:00 ET -> 09:30 ET next trading day (the literature's
  close-to-open convention, directly tradeable in futures).
- Intraday leg:  09:30 ET -> 16:00 ET.
- Boundary fills: close of the FIRST 1-minute bar at/after each boundary time.
  A day enters the sample only if both boundary fills exist within 30 minutes
  of the boundary (drops holidays/half-days cleanly).
- Tradability test: "overnight-only" strategy holds 1 contract 16:00->09:30,
  flat intraday. Costs per day: 1 round-trip commission ($4.60 NQ / $4.62 ES
  all-in) + 1 tick slippage per side (NQ tick 0.25pt=$5; ES tick 0.25pt=$12.50).
- Statistics: mean daily leg returns with circular block bootstrap CIs
  (block=21 days, 10k reps), pooled monthly Sharpe, yearly table, and an
  a-priori subperiod split at 2020-12-31 (pre-registered halfway point).

Usage: python src/overnight_intraday_study.py [NQ|ES]
Outputs: printed tables + <symbol>_legs.csv in data/cache/.
"""
import os, sys, glob
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(os.environ.get("ICT_DATA_DIR", Path(__file__).resolve().parent.parent / "data"))
CACHE = DATA_DIR / "cache"
CACHE.mkdir(parents=True, exist_ok=True)

SPEC = {
    "NQ": dict(prefix="MNQ", tick_pts=0.25, dollars_per_pt=20.0, comm_rt=4.60),
    "ES": dict(prefix="ES",  tick_pts=0.25, dollars_per_pt=50.0, comm_rt=4.62),
}
BOUNDARY_TOL_MIN = 30
BLOCK = 21
NBOOT = 10000
SPLIT_DATE = "2020-12-31"   # a-priori halfway split

def load_1m(prefix):
    files = sorted(glob.glob(str(DATA_DIR / f"{prefix}_*" / "raw_data.csv")))
    if not files:
        raise FileNotFoundError(f"no {prefix}_*/raw_data.csv under {DATA_DIR}")
    dfs = [pd.read_csv(f, usecols=lambda c: c.strip().lower() in ("time", "close")) for f in files]
    raw = pd.concat(dfs, ignore_index=True)
    raw.columns = [c.strip().lower() for c in raw.columns]
    raw["time"] = pd.to_numeric(raw["time"], errors="coerce")
    raw = raw.dropna().sort_values("time").drop_duplicates("time").reset_index(drop=True)
    return raw["time"].to_numpy(np.int64), raw["close"].to_numpy(np.float64)

def boundary_fills(t_ms, close):
    """For each calendar day: fill price at first bar >= 09:30 and >= 16:00 ET."""
    dt = pd.to_datetime(t_ms, unit="ms", utc=True).tz_convert("America/New_York")
    days = pd.Series(dt.date)
    uniq_days = days.unique()
    tz = "America/New_York"
    fills = {}
    for which, hh, mm in (("open", 9, 30), ("close", 16, 0)):
        bounds = pd.DatetimeIndex([pd.Timestamp(d).tz_localize(tz) + pd.Timedelta(hours=hh, minutes=mm)
                                   for d in uniq_days]).tz_convert("UTC").asi8 // 10**6
        idx = np.searchsorted(t_ms, bounds, side="left")
        ok = idx < len(t_ms)
        lag_min = np.full(len(bounds), np.inf)
        lag_min[ok] = (t_ms[idx[ok]] - bounds[ok]) / 60000.0
        valid = ok & (lag_min <= BOUNDARY_TOL_MIN)
        fills[which] = dict(days=uniq_days, price=np.where(valid, close[np.minimum(idx, len(t_ms)-1)], np.nan))
    return uniq_days, fills["open"]["price"], fills["close"]["price"]

def build_legs(symbol):
    cachef = CACHE / f"{symbol}_legs.csv"
    if cachef.exists():
        df = pd.read_csv(cachef, parse_dates=["day"])
        return df
    spec = SPEC[symbol]
    t_ms, close = load_1m(spec["prefix"])
    days, p_open, p_close = boundary_fills(t_ms, close)
    df = pd.DataFrame({"day": pd.to_datetime(days), "p_open": p_open, "p_close": p_close}).dropna()
    df = df.sort_values("day").reset_index(drop=True)
    # overnight: prev day 16:00 close -> today 09:30 open; intraday: today open->close
    df["overnight"] = df["p_open"] / df["p_close"].shift(1) - 1.0
    df["intraday"] = df["p_close"] / df["p_open"] - 1.0
    df["fullday"] = df["p_close"] / df["p_close"].shift(1) - 1.0
    df = df.dropna().reset_index(drop=True)
    # drop weekends artifacts: overnight legs spanning >4 calendar days (long gaps)
    gap_days = df["day"].diff().dt.days.fillna(1)
    df = df[gap_days <= 4].reset_index(drop=True)
    df.to_csv(cachef, index=False)
    return df

def block_bootstrap_ci(x, nboot=NBOOT, block=BLOCK, seed=42):
    rng = np.random.default_rng(seed)
    L = len(x)
    nb = int(np.ceil(L / block))
    means = np.empty(nboot)
    for k in range(nboot):
        starts = rng.integers(0, L, nb)
        sample = np.concatenate([x[(s + np.arange(block)) % L] for s in starts])[:L]
        means[k] = sample.mean()
    return np.percentile(means, 5), np.percentile(means, 95), (means <= 0).mean()

def ann(x, periods=252):
    return (1 + x.mean()) ** periods - 1

def sharpe_monthly(returns, days):
    ser = pd.Series(returns.values, index=days.values)
    m = ser.groupby([d.year * 100 + d.month for d in ser.index]).sum()
    if m.std(ddof=1) == 0: return 0.0
    return m.mean() / m.std(ddof=1) * np.sqrt(12)

def cost_per_day_ret(df, spec):
    """round-trip commission + 2 ticks slippage, as a fraction of notional."""
    notional = df["p_close"] * spec["dollars_per_pt"]
    dollars = spec["comm_rt"] + 2 * spec["tick_pts"] * spec["dollars_per_pt"]
    return dollars / notional

def report(symbol):
    spec = SPEC[symbol]
    df = build_legs(symbol)
    print("=" * 96)
    print(f"{symbol}: {len(df)} trading days, {df['day'].iloc[0].date()} -> {df['day'].iloc[-1].date()}")
    print("=" * 96)
    rows = []
    for leg in ("overnight", "intraday", "fullday"):
        x = df[leg].to_numpy()
        lo, hi, p0 = block_bootstrap_ci(x)
        rows.append((leg, x.mean() * 1e4, ann(df[leg]) * 100, lo * 1e4, hi * 1e4, p0 * 100,
                     sharpe_monthly(df[leg], df["day"])))
    print(f"{'leg':<11}{'mean bp/day':>12}{'annualized':>12}{'90% CI (bp)':>18}{'P(<=0)':>8}{'Sharpe':>8}")
    for r in rows:
        print(f"{r[0]:<11}{r[1]:>12.2f}{r[2]:>11.2f}%{f'[{r[3]:+.2f},{r[4]:+.2f}]':>18}{r[5]:>7.1f}%{r[6]:>8.2f}")

    # tradability: overnight-only net of costs
    cost = cost_per_day_ret(df, spec)
    net = df["overnight"] - cost
    lo, hi, p0 = block_bootstrap_ci(net.to_numpy())
    print(f"\nOVERNIGHT-ONLY STRATEGY, NET of costs (1 RT/day: ${spec['comm_rt']:.2f} comm + 2 ticks):")
    print(f"  avg cost: {cost.mean()*1e4:.2f} bp/day | net: {net.mean()*1e4:.2f} bp/day "
          f"= {ann(net)*100:.2f}%/yr | 90% CI [{lo*1e4:+.2f},{hi*1e4:+.2f}] bp | P(<=0) {p0*100:.1f}% "
          f"| Sharpe {sharpe_monthly(net, df['day']):.2f}")

    # a-priori subperiods
    print(f"\nSubperiods (pre-registered split at {SPLIT_DATE}):")
    for name, sub in (("first half", df[df['day'] <= SPLIT_DATE]), ("second half", df[df['day'] > SPLIT_DATE])):
        nsub = sub["overnight"] - cost_per_day_ret(sub, spec)
        print(f"  {name:<12} O/N {sub['overnight'].mean()*1e4:+6.2f} bp/d ({ann(sub['overnight'])*100:+6.2f}%/yr) | "
              f"intraday {sub['intraday'].mean()*1e4:+6.2f} bp/d ({ann(sub['intraday'])*100:+6.2f}%/yr) | "
              f"O/N net {nsub.mean()*1e4:+6.2f} bp/d")

    # yearly table
    print("\nYearly (bp/day):")
    print(f"{'year':<6}{'days':>6}{'overnight':>11}{'intraday':>11}{'O/N net':>10}")
    for yr, g in df.groupby(df["day"].dt.year):
        gnet = g["overnight"] - cost_per_day_ret(g, spec)
        print(f"{yr:<6}{len(g):>6}{g['overnight'].mean()*1e4:>+11.2f}{g['intraday'].mean()*1e4:>+11.2f}{gnet.mean()*1e4:>+10.2f}")
    return df

if __name__ == "__main__":
    symbols = [sys.argv[1]] if len(sys.argv) > 1 else ["NQ", "ES"]
    for s in symbols:
        report(s)
        print()
