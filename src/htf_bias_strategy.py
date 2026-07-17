"""
HTF Bias Top-Down Strategy (NQ/MNQ, 1m data resampled to 5m/15m/1h/4h)

Rules implemented (as specified):
1. Daily bias from 1h & 4h BOS-trend. 4h trumps an opposite 1h.
   - agree            -> execute on 5m  (LTF confirm on 1m)
   - oppose / 1h flat -> execute on 15m (LTF confirm on 5m)
   - 4h flat          -> use 1h trend, execute on 15m
   - both flat        -> no trading
2. Wait for a bias-matching HTF liquidity sweep:
   long bias -> sweep of a 1h/4h confirmed pivot LOW, previous-day low,
   or a completed session low (Asia/London/NY). Mirror for shorts.
3. Then BOS on the execution TF (close beyond the most recent confirmed
   exec-TF pivot in trade direction).
4. Then wait for retracement into a POI: order block, FVG (of the
   displacement leg), breaker block, or dealing-range equilibrium.
5. Then LTF confirmation: scale down (5m->1m, 15m->5m) and enter on an
   LTF BOS instead of waiting for the HTF candle to close.

Exits: split TP (50% at +2R then stop to BE, 50% at +4R), stop at the
local retracement low/high, 3-day timeout. Fills simulated on 1m bars,
dual-touch counts as a stop (conservative).

Costs are included from the start: $4.60 all-in round-turn per NQ
contract + 1-tick ($5/NQ) slippage on market/stop legs. Sizing: 1% of
$100k risked per trade (fractional contracts, flat — no compounding).

ALL PARAMETERS ARE FIXED A PRIORI (below). No optimization was run on
them. Changing them and re-running to find a better number reintroduces
the selection bias documented in docs/VALIDATION_RESULTS.md.
"""
import os, sys, time, pickle, glob
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(os.environ.get("ICT_DATA_DIR", Path(__file__).resolve().parent.parent / "data"))
CACHE = DATA_DIR / "cache"
CACHE.mkdir(parents=True, exist_ok=True)
TMP = str(CACHE)

# ------------------------- fixed parameters -------------------------
PIVOT_K_HTF = 3        # fractal half-width, 1h/4h pivots
PIVOT_K_EXEC = 3       # fractal half-width, exec-TF pivots
PIVOT_K_LTF = 2        # fractal half-width, LTF confirmation pivots
SWEEP_TO_BOS_BARS = 24     # exec bars allowed from sweep to BOS
RETRACE_BARS_EXEC = 36     # exec bars allowed for POI touch + confirm
LTF_CONFIRM_BARS = 60      # LTF bars allowed after POI touch for LTF BOS
MIN_R_PTS, MAX_R_PTS = 5.0, 150.0
TP1_R, TP2_R = 2.0, 4.0
TIMEOUT_1M_BARS = 3 * 1380     # ~3 trading days
COOLDOWN_EXEC_BARS = 6
RISK_DOLLARS = 1000.0          # 1% of 100k
NQ_DPP = 20.0                  # $/pt per NQ contract
COMM_RT = 4.60                 # $ round trip per NQ contract
TICK_VAL = 5.0                 # $ per tick per NQ
SESSIONS_NY = [(18, 3), (3, 8), (8, 17)]   # Asia / London / NY (ET, end-exclusive)

# ------------------------- data / resampling -------------------------
def resample(t, o, h, l, c, tf_ms):
    key = (t // tf_ms).astype(np.int64)
    changes = np.nonzero(np.diff(key))[0] + 1
    starts = np.concatenate([[0], changes])
    ends = np.concatenate([changes, [len(t)]])
    n = len(starts)
    T = t[starts]
    O = o[starts]; C = c[ends - 1]
    H = np.array([h[s:e].max() for s, e in zip(starts, ends)])
    L = np.array([l[s:e].min() for s, e in zip(starts, ends)])
    Tend = t[ends - 1] + 60000            # bar completion time (close of last 1m)
    return dict(t=T, o=O, h=H, l=L, c=C, tend=Tend, start_i=starts, end_i=ends - 1)

def causal_pivots(H, L, k):
    """returns arrays: for each bar j, lists of pivots CONFIRMED at j:
    (kind, price, formed_idx). Vectorized detection, +k confirmation lag."""
    n = len(H)
    ph = np.ones(n, dtype=bool); pl = np.ones(n, dtype=bool)
    for d in range(1, k + 1):
        ph[k:n-k] &= (H[k:n-k] >= np.roll(H, -d)[k:n-k]) & (H[k:n-k] > np.roll(H, d)[k:n-k])
        pl[k:n-k] &= (L[k:n-k] <= np.roll(L, -d)[k:n-k]) & (L[k:n-k] < np.roll(L, d)[k:n-k])
    ph[:k] = ph[n-k:] = False; pl[:k] = pl[n-k:] = False
    return ph, pl   # pivot at i is knowable at bar i+k

def trend_series(C, ph, pl, H, L, k):
    """BOS trend per bar (uses only pivots confirmed <= bar). 1=up,-1=down,0=none."""
    n = len(C)
    trend = np.zeros(n, dtype=np.int8)
    last_ph, last_pl = np.nan, np.nan
    tr = 0
    for i in range(n):
        j = i - k                      # pivot formed at j is confirmed now
        if j >= 0:
            if ph[j]: last_ph = H[j]
            if pl[j]: last_pl = L[j]
        if not np.isnan(last_ph) and C[i] > last_ph:
            tr = 1; last_ph = np.nan   # consumed; wait for next pivot
        elif not np.isnan(last_pl) and C[i] < last_pl:
            tr = -1; last_pl = np.nan
        trend[i] = tr
    return trend

def build_raw_cache():
    """Combine per-year 1m CSVs into a single sorted npz (built once)."""
    rp = f"{TMP}/raw_cache.npz"
    if os.path.exists(rp):
        return np.load(rp)
    files = sorted(glob.glob(str(DATA_DIR / "MNQ_*" / "raw_data.csv")))
    if not files:
        raise FileNotFoundError(f"No data/MNQ_*/raw_data.csv found under {DATA_DIR}. "
                                "Run download_databento.py first (see data/README.md).")
    dfs = [pd.read_csv(f, usecols=lambda c: c.strip().lower() in ('time','open','high','low','close')) for f in files]
    raw = pd.concat(dfs, ignore_index=True)
    raw.columns = [c.strip().lower() for c in raw.columns]
    raw['time'] = pd.to_numeric(raw['time'], errors='coerce')
    raw = raw.sort_values('time').drop_duplicates(subset=['time']).reset_index(drop=True)
    np.savez(rp, time=raw['time'].to_numpy(np.float64), open=raw['open'].to_numpy(np.float64),
             high=raw['high'].to_numpy(np.float64), low=raw['low'].to_numpy(np.float64),
             close=raw['close'].to_numpy(np.float64))
    return np.load(rp)

def build_context():
    ck = f"{TMP}/htf_ctx.pkl"
    if os.path.exists(ck):
        with open(ck, 'rb') as f: return pickle.load(f)
    rc = build_raw_cache()
    t, o, h, l, c = (rc[k] for k in ('time', 'open', 'high', 'low', 'close'))
    ctx = {'m1': dict(t=t, o=o, h=h, l=l, c=c)}
    for name, tf in (('m5', 5), ('m15', 15), ('h1', 60), ('h4', 240)):
        ctx[name] = resample(t, o, h, l, c, tf * 60000)
    for name, k in (('h1', PIVOT_K_HTF), ('h4', PIVOT_K_HTF),
                    ('m5', PIVOT_K_EXEC), ('m15', PIVOT_K_EXEC)):
        b = ctx[name]
        b['ph'], b['pl'] = causal_pivots(b['h'], b['l'], k)
    for name in ('h1', 'h4'):
        b = ctx[name]
        b['trend'] = trend_series(b['c'], b['ph'], b['pl'], b['h'], b['l'], PIVOT_K_HTF)
    # 1m LTF pivots
    m1 = ctx['m1']
    m1_ph, m1_pl = causal_pivots(m1['h'], m1['l'], PIVOT_K_LTF)
    m1['ph'], m1['pl'] = m1_ph, m1_pl
    ctx['m5']['ph_ltf'], ctx['m5']['pl_ltf'] = causal_pivots(ctx['m5']['h'], ctx['m5']['l'], PIVOT_K_LTF)
    # sessions / prev-day levels on 1m timeline (ET)
    dt = pd.to_datetime(t, unit='ms', utc=True).tz_convert('America/New_York')
    hours = dt.hour.to_numpy()
    tday = (dt - pd.Timedelta(hours=18)).date  # trading day starts 18:00 ET
    ctx['hours'] = hours
    ctx['tday'] = np.array([d.toordinal() for d in tday])
    with open(ck, 'wb') as f: pickle.dump(ctx, f)
    return ctx

# ------------------------- level tracking -------------------------
class Levels:
    """active liquidity levels below/above price: HTF pivot lows/highs,
    prev-day and completed-session extremes."""
    def __init__(self):
        self.lows = {}    # id -> price
        self.highs = {}
        self._next = 0
    def add(self, price, side):
        (self.lows if side == 'low' else self.highs)[self._next] = price
        self._next += 1
    def mitigate(self, close):
        self.lows = {k: v for k, v in self.lows.items() if close > v}
        self.highs = {k: v for k, v in self.highs.items() if close < v}
    def swept_low(self, bar_low):
        hit = [k for k, v in self.lows.items() if bar_low <= v]
        for k in hit: del self.lows[k]
        return len(hit) > 0
    def swept_high(self, bar_high):
        hit = [k for k, v in self.highs.items() if bar_high >= v]
        for k in hit: del self.highs[k]
        return len(hit) > 0
    def trim(self, px):
        # keep 40 nearest per side
        if len(self.lows) > 40:
            keep = sorted(self.lows.items(), key=lambda kv: px - kv[1])[:40]
            self.lows = dict(keep)
        if len(self.highs) > 40:
            keep = sorted(self.highs.items(), key=lambda kv: kv[1] - px)[:40]
            self.highs = dict(keep)

# ------------------------- exit simulation on 1m -------------------------
def simulate_exit(m1, ei, direction, entry, stop, funnel):
    h, l, c = m1['h'], m1['l'], m1['c']
    R = abs(entry - stop)
    tp1 = entry + direction * TP1_R * R
    tp2 = entry + direction * TP2_R * R
    hit1 = False
    st = stop
    end = min(ei + TIMEOUT_1M_BARS, len(h))
    for i in range(ei + 1, end):
        if direction == 1:
            hs, ht1, ht2 = l[i] <= st, h[i] >= tp1, h[i] >= tp2
        else:
            hs, ht1, ht2 = h[i] >= st, l[i] <= tp1, l[i] <= tp2
        if not hit1:
            if hs:  return -1.0, 'stop', i
            if ht1:
                hit1 = True; st = entry
                if (direction == 1 and l[i] <= st) or (direction == -1 and h[i] >= st):
                    return 0.5 * TP1_R, 'trail_be', i
        else:
            if hs:  return 0.5 * TP1_R, 'trail_be', i
            if ht2: return 0.5 * TP1_R + 0.5 * TP2_R, 'tp2', i
    i = end - 1
    fr = (c[i] - entry) / R * direction
    if hit1: return 0.5 * TP1_R + max(0.0, min(TP2_R, fr)) * 0.5, 'timeout', i
    return max(-1.0, min(TP1_R, fr)) * 1.0 if not hit1 else 0.0, 'timeout', i

# ------------------------- POI + LTF confirmation -------------------------
def find_entry(ctx, stream, direction, sweep_ext, leg_lo_i, leg_hi_i, bos_i, funnel, breaker_i=None):
    """After BOS at exec bar bos_i: build zones, scan 1m for POI touch,
    then LTF BOS entry. Returns (entry_1m_idx, entry_px, stop_px) or None.
    Long: zones below; mirrored for shorts via sign flips."""
    b = ctx[stream]; m1 = ctx['m1']
    s, e = b['start_i'][leg_lo_i], b['end_i'][bos_i]
    H, L, O, C = b['h'], b['l'], b['o'], b['c']
    if direction == 1:
        leg_hi = H[leg_lo_i:bos_i + 1].max()
        rng = leg_hi - sweep_ext
        if rng <= 0: return None
        zones = [(sweep_ext + 0.5 * rng)]                       # EQ line
        for i in range(leg_lo_i + 2, bos_i + 1):                # bullish FVGs
            if L[i] > H[i - 2]: zones.append(L[i])
        for i in range(leg_lo_i, max(leg_lo_i - 8, 0) - 1, -1): # OB: last down candle
            if C[i] < O[i]: zones.append(H[i]); break
        if breaker_i is not None:                                # breaker: broken pivot-high candle
            zones.append(H[breaker_i])
        ztop = max(zones)                                        # nearest zone top
        zfloor = sweep_ext
    else:
        leg_lo = L[leg_lo_i:bos_i + 1].min()
        rng = sweep_ext - leg_lo
        if rng <= 0: return None
        zones = [(sweep_ext - 0.5 * rng)]
        for i in range(leg_lo_i + 2, bos_i + 1):
            if H[i] < L[i - 2]: zones.append(H[i])
        for i in range(leg_lo_i, max(leg_lo_i - 8, 0) - 1, -1):
            if C[i] > O[i]: zones.append(L[i]); break
        if breaker_i is not None:                                # breaker: broken pivot-low candle
            zones.append(L[breaker_i])
        ztop = min(zones)
        zfloor = sweep_ext

    # scan 1m from BOS close for POI touch
    tf_1m = 5 if stream == 'm5' else 15
    start = b['end_i'][bos_i] + 1
    horizon = start + RETRACE_BARS_EXEC * tf_1m
    h1m, l1m, c1m = m1['h'], m1['l'], m1['c']
    ph1m, pl1m = m1['ph'], m1['pl']
    touch = None
    for i in range(start, min(horizon, len(h1m))):
        if direction == 1:
            if c1m[i] < zfloor: return None          # invalidated
            if l1m[i] <= ztop: touch = i; break
        else:
            if c1m[i] > zfloor: return None
            if h1m[i] >= ztop: touch = i; break
    if touch is None: return None
    funnel['poi_touch'] += 1

    # LTF BOS confirmation (1m pivots for m5 stream; 5m pivots for m15 stream)
    k = PIVOT_K_LTF
    if stream == 'm5':
        ref = np.nan; lo_track = l1m[touch] if direction == 1 else h1m[touch]
        for i in range(touch + 1, min(touch + LTF_CONFIRM_BARS, len(h1m))):
            j = i - k
            if j > touch:
                if direction == 1 and ph1m[j]: ref = h1m[j]
                if direction == -1 and pl1m[j]: ref = l1m[j]
            if direction == 1:
                lo_track = min(lo_track, l1m[i])
                if c1m[i] < zfloor: return None
                if not np.isnan(ref) and c1m[i] > ref:
                    return i, c1m[i], lo_track
            else:
                lo_track = max(lo_track, h1m[i])
                if c1m[i] > zfloor: return None
                if not np.isnan(ref) and c1m[i] < ref:
                    return i, c1m[i], lo_track
        return None
    else:
        m5 = ctx['m5']
        j0 = int(np.searchsorted(m5['end_i'], touch, side='left'))
        ref = np.nan
        lo_track = l1m[touch] if direction == 1 else h1m[touch]
        ph5, pl5 = m5['ph_ltf'], m5['pl_ltf']
        for jj in range(j0 + 1, min(j0 + LTF_CONFIRM_BARS // 3, len(m5['c']))):
            jp = jj - k
            if jp > j0:
                if direction == 1 and ph5[jp]: ref = m5['h'][jp]
                if direction == -1 and pl5[jp]: ref = m5['l'][jp]
            if direction == 1:
                lo_track = min(lo_track, m5['l'][jj])
                if m5['c'][jj] < zfloor: return None
                if not np.isnan(ref) and m5['c'][jj] > ref:
                    return m5['end_i'][jj], m5['c'][jj], lo_track
            else:
                lo_track = max(lo_track, m5['h'][jj])
                if m5['c'][jj] > zfloor: return None
                if not np.isnan(ref) and m5['c'][jj] < ref:
                    return m5['end_i'][jj], m5['c'][jj], lo_track
        return None

# ------------------------- draw-target exit (spec-faithful variant) ----------
def simulate_exit_draw(m1, ei, direction, entry, stop, target):
    """Full position: stop at `stop`, single TP at the opposing liquidity draw."""
    h, l, c = m1['h'], m1['l'], m1['c']
    R = abs(entry - stop)
    tgtR = abs(target - entry) / R
    end = min(ei + TIMEOUT_1M_BARS, len(h))
    for i in range(ei + 1, end):
        if direction == 1:
            hs, ht = l[i] <= stop, h[i] >= target
        else:
            hs, ht = h[i] >= stop, l[i] <= target
        if hs: return -1.0, 'stop', i
        if ht: return tgtR, 'tp_draw', i
    i = end - 1
    fr = (c[i] - entry) / R * direction
    return max(-1.0, min(tgtR, fr)), 'timeout', i

# ------------------------- main engine -------------------------
def run(stop_mode='local', tp_mode='split'):
    """stop_mode: 'local' (retracement low) | 'sweep' (below sweep extreme)
       tp_mode:   'split' (2R/BE/4R)       | 'draw' (opposing HTF level)"""
    ctx = build_context()
    m1, m5, m15, h1, h4 = ctx['m1'], ctx['m5'], ctx['m15'], ctx['h1'], ctx['h4']
    n5 = len(m5['c'])
    funnel = dict(sweeps=0, bos=0, poi_touch=0, entries=0)
    levels = Levels()
    trades = []

    # per-5m-bar last completed 1h/4h indices
    h1_idx = np.searchsorted(h1['tend'], m5['tend'], side='right') - 1
    h4_idx = np.searchsorted(h4['tend'], m5['tend'], side='right') - 1
    m15_of_m5 = np.searchsorted(m15['tend'], m5['tend'], side='right') - 1

    # session/day level feed: completed session & prev-day extremes, appended when session ends
    hours = ctx['hours']; tday = ctx['tday']
    sess_id = np.zeros(len(hours), dtype=np.int64)
    sess_code = np.full(len(hours), -1, dtype=np.int8)
    for si, (a, bnd) in enumerate(SESSIONS_NY):
        if a < bnd: mask = (hours >= a) & (hours < bnd)
        else:       mask = (hours >= a) | (hours < bnd)
        sess_code[mask] = si
    # session key: (trading day, session code)
    skey = tday * 4 + sess_code
    ch = np.nonzero(np.diff(skey))[0]
    sess_ends = ch                       # 1m idx where a session block ends
    sess_starts = np.concatenate([[0], ch + 1])
    sess_bounds = list(zip(sess_starts, np.concatenate([ch, [len(hours) - 1]])))
    sess_end_times = m1['t'][[b for _, b in sess_bounds]] + 60000
    # prev trading day H/L
    dch = np.nonzero(np.diff(tday))[0]
    day_bounds = list(zip(np.concatenate([[0], dch + 1]), np.concatenate([dch, [len(hours) - 1]])))
    day_end_times = m1['t'][[b for _, b in day_bounds]] + 60000

    sp = dp = 0          # feed pointers
    hp1 = hp4 = 0        # HTF pivot confirmation pointers (bar index)
    state = 'IDLE'
    stream = 'm5'; sdir = 0
    sweep_ext = 0.0; sweep_i = 0; leg_lo_i = 0; bos_ref = np.nan
    cooldown_until = -1
    resume_1m = -1       # while in trade / resolving, skip exec bars before this
    exec_i_prev = -1

    for i in range(n5):
        t_end = m5['tend'][i]
        # ---- feed levels from completed sessions/days ----
        while sp < len(sess_bounds) and sess_end_times[sp] <= t_end:
            a, bnd = sess_bounds[sp]
            if bnd > a + 3:
                levels.add(m1['l'][a:bnd + 1].min(), 'low')
                levels.add(m1['h'][a:bnd + 1].max(), 'high')
            sp += 1
        while dp < len(day_bounds) and day_end_times[dp] <= t_end:
            a, bnd = day_bounds[dp]
            if bnd > a + 30:
                levels.add(m1['l'][a:bnd + 1].min(), 'low')
                levels.add(m1['h'][a:bnd + 1].max(), 'high')
            dp += 1
        # ---- feed levels from newly confirmed 1h/4h pivots; mitigation on 1h close ----
        while hp1 <= h1_idx[i]:
            j = hp1 - PIVOT_K_HTF
            if j >= 0:
                if h1['pl'][j]: levels.add(h1['l'][j], 'low')
                if h1['ph'][j]: levels.add(h1['h'][j], 'high')
            hp1 += 1
        while hp4 <= h4_idx[i]:
            j = hp4 - PIVOT_K_HTF
            if j >= 0:
                if h4['pl'][j]: levels.add(h4['l'][j], 'low')
                if h4['ph'][j]: levels.add(h4['h'][j], 'high')
            hp4 += 1
        if h1_idx[i] >= 0 and (i == 0 or h1_idx[i] != h1_idx[i - 1]):
            levels.mitigate(h1['c'][h1_idx[i]])
            levels.trim(m5['c'][i])

        if m5['end_i'][i] < resume_1m:
            continue
        # ---- bias ----
        t1 = h1['trend'][h1_idx[i]] if h1_idx[i] >= 0 else 0
        t4 = h4['trend'][h4_idx[i]] if h4_idx[i] >= 0 else 0
        if t4 != 0:
            bias = t4; exec_tf = 'm5' if t1 == t4 else 'm15'
        elif t1 != 0:
            bias = t1; exec_tf = 'm15'
        else:
            bias = 0
        if bias == 0:
            state = 'IDLE'; continue

        # active exec bar? (m15 stream acts only when a 15m bar completes at this 5m close)
        if exec_tf == 'm15':
            k15 = m15_of_m5[i]
            if k15 < 0 or (i > 0 and k15 == m15_of_m5[i - 1]):
                continue
            b = m15; ei = k15
        else:
            b = m5; ei = i

        if state != 'IDLE' and (stream != exec_tf or sdir != bias):
            state = 'IDLE'   # bias/stream flip invalidates pending setup

        if state == 'IDLE':
            if i < cooldown_until: continue
            swept = levels.swept_low(b['l'][ei]) if bias == 1 else levels.swept_high(b['h'][ei])
            if swept:
                funnel['sweeps'] += 1
                state = 'SWEPT'; stream = exec_tf; sdir = bias
                sweep_i = ei
                sweep_ext = b['l'][ei] if bias == 1 else b['h'][ei]
                leg_lo_i = ei
                # BOS reference: most recent confirmed exec pivot in trade direction
                bos_ref = np.nan; bos_ref_i = None
                kk = PIVOT_K_EXEC
                for j in range(ei - kk, max(ei - 60, 0), -1):
                    if sdir == 1 and b['ph'][j]: bos_ref = b['h'][j]; bos_ref_i = j; break
                    if sdir == -1 and b['pl'][j]: bos_ref = b['l'][j]; bos_ref_i = j; break
                if np.isnan(bos_ref): state = 'IDLE'
        elif state == 'SWEPT':
            if ei - sweep_i > SWEEP_TO_BOS_BARS:
                state = 'IDLE'; cooldown_until = i + COOLDOWN_EXEC_BARS; continue
            bos_hit = (b['c'][ei] > bos_ref) if sdir == 1 else (b['c'][ei] < bos_ref)
            if sdir == 1 and b['l'][ei] < sweep_ext: sweep_ext = b['l'][ei]; leg_lo_i = ei
            if sdir == -1 and b['h'][ei] > sweep_ext: sweep_ext = b['h'][ei]; leg_lo_i = ei
            if bos_hit:
                funnel['bos'] += 1
                r = find_entry(ctx, stream, sdir, sweep_ext, leg_lo_i, ei, ei, funnel, breaker_i=bos_ref_i)
                state = 'IDLE'; cooldown_until = i + COOLDOWN_EXEC_BARS
                if r is not None:
                    e1m, entry, stop_local = r
                    stop = sweep_ext if stop_mode == 'sweep' else stop_local
                    if (sdir == 1 and stop >= entry) or (sdir == -1 and stop <= entry):
                        continue
                    R = abs(entry - stop)
                    if MIN_R_PTS <= R <= MAX_R_PTS:
                        if tp_mode == 'draw':
                            # nearest active opposing draw beyond entry; fallback +4R
                            if sdir == 1:
                                cands = [v for v in levels.highs.values() if v > entry + R]
                                target = min(cands) if cands else entry + 4.0 * R
                            else:
                                cands = [v for v in levels.lows.values() if v < entry - R]
                                target = max(cands) if cands else entry - 4.0 * R
                            retR, etype, xi = simulate_exit_draw(m1, e1m, sdir, entry, stop, target)
                        else:
                            retR, etype, xi = simulate_exit(m1, e1m, sdir, entry, stop, funnel)
                        funnel['entries'] += 1
                        trades.append((m1['t'][e1m], sdir, R, retR, etype, stream, m1['t'][xi]))
                        resume_1m = xi

    return trades, funnel

if __name__ == '__main__':
    import itertools
    combos = [(sm, tm) for sm in ('local', 'sweep') for tm in ('split', 'draw')]
    if len(sys.argv) > 2:
        combos = [(sys.argv[1], sys.argv[2])]
    for sm, tm in combos:
        t0 = time.time()
        trades, funnel = run(stop_mode=sm, tp_mode=tm)
        with open(f"{TMP}/htf_trades_{sm}_{tm}.pkl", 'wb') as f:
            pickle.dump(trades, f)
        print(f"stop={sm} tp={tm}: {len(trades)} trades in {time.time()-t0:.0f}s | funnel {funnel}")
