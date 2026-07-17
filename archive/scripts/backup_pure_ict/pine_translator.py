import pandas as pd
import numpy as np
import os
import pytz

# =====================================================================
# 1. HELPER FUNCTIONS FOR COMPATIBILITY & TESTING
# =====================================================================
def add_liquidity_pool(arr, price, atr_val, is_high):
    """
    Proximity / expansion check: A new level is valid if it is a structural expansion
    (price > all active highs, or price < all active lows), OR if it is not in a stronger
    cluster (no active pool within 1.5 * ATR with strength_score > 1.0).
    Stacking logic: if price is close to any existing active level, increment both strength scores by 0.5.
    """
    if pd.isna(price) or pd.isna(atr_val) or atr_val <= 0.0:
        return
    
    is_expansion = True
    for p in arr:
        if p['active']:
            if is_high:
                if price <= p['price']:
                    is_expansion = False
                    break
            else:
                if price >= p['price']:
                    is_expansion = False
                    break
                    
    has_near_cluster = False
    max_near_strength = 0.0
    for p in arr:
        if p['active']:
            dist = abs(price - p['price'])
            if dist < atr_val * 1.5:
                has_near_cluster = True
                if p['strength_score'] > max_near_strength:
                    max_near_strength = p['strength_score']
                    
    is_valid = is_expansion or (not has_near_cluster or max_near_strength <= 1.0)
    
    if is_valid:
        new_pool = {
            'price': float(price),
            'strength_score': 1.0,
            'active': True
        }
        for p in arr:
            if p['active']:
                dist = abs(price - p['price'])
                if dist < atr_val * 1.5:
                    p['strength_score'] += 0.5
                    new_pool['strength_score'] += 0.5
        arr.append(new_pool)

def update_mitigations(arr, is_high, close_val):
    if pd.isna(close_val):
        return
    for p in arr:
        if p['active']:
            if is_high:
                if close_val >= p['price']:
                    p['active'] = False
            else:
                if close_val <= p['price']:
                    p['active'] = False

def check_sweep(arr, is_high, bar_high, bar_low):
    swept = False
    max_strength = 0.0
    for p in arr:
        if p['active']:
            if is_high:
                if bar_high >= p['price']:
                    swept = True
                    if p['strength_score'] > max_strength:
                        max_strength = p['strength_score']
            else:
                if bar_low <= p['price']:
                    swept = True
                    if p['strength_score'] > max_strength:
                        max_strength = p['strength_score']
    return swept, max_strength

def find_closest_high(arr1, arr2, pdh_val, pdh_active_val, entry_price):
    closest_high = np.nan
    for p in arr1:
        if p['active'] and p['price'] > entry_price:
            if pd.isna(closest_high) or p['price'] < closest_high:
                closest_high = p['price']
    for p in arr2:
        if p['active'] and p['price'] > entry_price:
            if pd.isna(closest_high) or p['price'] < closest_high:
                closest_high = p['price']
    if pdh_active_val and not pd.isna(pdh_val) and pdh_val > entry_price:
        if pd.isna(closest_high) or pdh_val < closest_high:
            closest_high = pdh_val
    return closest_high

def find_closest_low(arr1, arr2, pdl_val, pdl_active_val, entry_price):
    closest_low = np.nan
    for p in arr1:
        if p['active'] and p['price'] < entry_price:
            if pd.isna(closest_low) or p['price'] > closest_low:
                closest_low = p['price']
    for p in arr2:
        if p['active'] and p['price'] < entry_price:
            if pd.isna(closest_low) or p['price'] > closest_low:
                closest_low = p['price']
    if pdl_active_val and not pd.isna(pdl_val) and pdl_val < entry_price:
        if pd.isna(closest_low) or pdl_val > closest_low:
            closest_low = pdl_val
    return closest_low

def compute_structure_signals(df):
    """
    Compute structure signals on a dataframe (used for testing).
    """
    n = len(df)
    high = df['High'].values
    low = df['Low'].values
    close = df['Close'].values
    
    bos_up_arr = np.zeros(n)
    bos_down_arr = np.zeros(n)
    fvg_bull_arr = np.zeros(n)
    fvg_bull_top_arr = np.full(n, np.nan)
    fvg_bull_bottom_arr = np.full(n, np.nan)
    fvg_bear_arr = np.zeros(n)
    fvg_bear_top_arr = np.full(n, np.nan)
    fvg_bear_bottom_arr = np.full(n, np.nan)
    fvg_bull_inv_arr = np.zeros(n)
    fvg_bear_inv_arr = np.zeros(n)
    
    last_ph = None
    last_pl = None
    fvg_bull_top = None
    fvg_bull_bottom = None
    fvg_bull_revisited = False
    fvg_bear_top = None
    fvg_bear_bottom = None
    fvg_bear_revisited = False
    
    last_fvg_bull_top = None
    last_fvg_bull_bottom = None
    last_fvg_bear_top = None
    last_fvg_bear_bottom = None
    
    for i in range(n):
        ph, pl = None, None
        if i >= 4:
            if high[i-2] > max(high[i-4], high[i-3], high[i-1], high[i]):
                ph = high[i-2]
            if low[i-2] < min(low[i-4], low[i-3], low[i-1], low[i]):
                pl = low[i-2]
                
        if ph is not None: last_ph = ph
        if pl is not None: last_pl = pl
            
        bos_up = False
        bos_down = False
        if last_ph is not None and close[i] > last_ph:
            bos_up = True
            last_ph = None
        if last_pl is not None and close[i] < last_pl:
            bos_down = True
            last_pl = None
            
        if i >= 2:
            if low[i] > high[i-2]:
                fvg_bull_top = low[i]
                fvg_bull_bottom = high[i-2]
                fvg_bull_revisited = False
            if high[i] < low[i-2]:
                fvg_bear_top = low[i-2]
                fvg_bear_bottom = high[i]
                fvg_bear_revisited = False
                
        fvg_bull_rej = False
        fvg_bear_rej = False
        fvg_bull_inv = False
        fvg_bear_inv = False
        
        if fvg_bull_top is not None:
            if low[i] < fvg_bull_top:
                fvg_bull_revisited = True
            if fvg_bull_revisited and close[i] > fvg_bull_top and low[i] > fvg_bull_bottom:
                fvg_bull_rej = True
                last_fvg_bull_top = fvg_bull_top
                last_fvg_bull_bottom = fvg_bull_bottom
                fvg_bull_top = None
                fvg_bull_bottom = None
                fvg_bull_revisited = False
            elif low[i] <= fvg_bull_bottom:
                fvg_bull_inv = True
                fvg_bull_top = None
                fvg_bull_bottom = None
                fvg_bull_revisited = False
                
        if fvg_bear_bottom is not None:
            if high[i] > fvg_bear_bottom:
                fvg_bear_revisited = True
            if fvg_bear_revisited and close[i] < fvg_bear_bottom and high[i] < fvg_bear_top:
                fvg_bear_rej = True
                last_fvg_bear_top = fvg_bear_top
                last_fvg_bear_bottom = fvg_bear_bottom
                fvg_bear_top = None
                fvg_bear_bottom = None
                fvg_bear_revisited = False
            elif high[i] >= fvg_bear_top:
                fvg_bear_inv = True
                fvg_bear_top = None
                fvg_bear_bottom = None
                fvg_bear_revisited = False
                
        bos_up_arr[i] = 1.0 if bos_up else 0.0
        bos_down_arr[i] = 1.0 if bos_down else 0.0
        fvg_bull_arr[i] = 1.0 if fvg_bull_rej else 0.0
        fvg_bear_arr[i] = 1.0 if fvg_bear_rej else 0.0
        fvg_bull_inv_arr[i] = 1.0 if fvg_bull_inv else 0.0
        fvg_bear_inv_arr[i] = 1.0 if fvg_bear_inv else 0.0
        
        if last_fvg_bull_top is not None:
            fvg_bull_top_arr[i] = last_fvg_bull_top
            fvg_bull_bottom_arr[i] = last_fvg_bull_bottom
        if last_fvg_bear_top is not None:
            fvg_bear_top_arr[i] = last_fvg_bear_top
            fvg_bear_bottom_arr[i] = last_fvg_bear_bottom
            
    res_df = pd.DataFrame(index=df.index)
    res_df['bos_up'] = bos_up_arr
    res_df['bos_down'] = bos_down_arr
    res_df['fvg_bull'] = fvg_bull_arr
    res_df['fvg_bear'] = fvg_bear_arr
    res_df['fvg_bull_inv'] = fvg_bull_inv_arr
    res_df['fvg_bear_inv'] = fvg_bear_inv_arr
    res_df['fvg_bull_top'] = fvg_bull_top_arr
    res_df['fvg_bull_bottom'] = fvg_bull_bottom_arr
    res_df['fvg_bear_top'] = fvg_bear_top_arr
    res_df['fvg_bear_bottom'] = fvg_bear_bottom_arr
    return res_df

def prepare_htf_series(df_1m, rule, offset):
    res = df_1m.set_index('datetime').resample(rule, closed='left', label='left').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    })
    res['Close'] = res['Close'].ffill().bfill()
    res['Open'] = res['Open'].fillna(res['Close'])
    res['High'] = res['High'].fillna(res['Close'])
    res['Low'] = res['Low'].fillna(res['Close'])
    res['Volume'] = res['Volume'].fillna(0.0)
    
    high_val = res['High']
    low_val = res['Low']
    close_val = res['Close']
    tr = np.maximum(high_val - low_val, 
                    np.maximum(np.abs(high_val - close_val.shift(1)), 
                               np.abs(low_val - close_val.shift(1))))
    tr.iloc[0] = high_val.iloc[0] - low_val.iloc[0]
    res['atr'] = tr.ewm(alpha=1.0/14.0, adjust=False).mean()
    
    close_prev = res['Close'].shift(1)
    open_prev = res['Open'].shift(1)
    is_green_red = (close_prev > open_prev) & (res['Close'] < res['Open'])
    is_red_green = (close_prev < open_prev) & (res['Close'] > res['Open'])
    
    res['pivot_high'] = np.where(is_green_red, np.maximum(res['High'].shift(1), res['High']), np.nan)
    res['pivot_low'] = np.where(is_red_green, np.minimum(res['Low'].shift(1), res['Low']), np.nan)
    
    res.index = res.index + offset
    return res

# =====================================================================
# 2. STREAMING ENGINE STRUCTURES
# =====================================================================
class StreamingBar:
    def __init__(self, period_minutes):
        self.period_minutes = period_minutes
        self.current_bar_start_min = None
        self.open = None
        self.high = None
        self.low = None
        self.close = None
        self.volume = 0.0
        self.completed_history = []
        self.atr = None

    def update(self, t_ms, o, h, l, c, v):
        t_min = t_ms // 60000
        bin_start_min = (t_min // self.period_minutes) * self.period_minutes
        
        completed_bar = None
        
        if self.current_bar_start_min is None:
            self.current_bar_start_min = bin_start_min
            self.open = o
            self.high = h
            self.low = l
            self.close = c
            self.volume = v
        elif bin_start_min != self.current_bar_start_min:
            completed_bar = {
                'time_min': self.current_bar_start_min,
                'open': self.open,
                'high': self.high,
                'low': self.low,
                'close': self.close,
                'volume': self.volume,
                'comp_time_min': self.current_bar_start_min + self.period_minutes
            }
            # Calculate ATR for 1H & 4H
            if self.period_minutes in [60, 240]:
                tr = self.high - self.low
                if self.completed_history:
                    prev_c = self.completed_history[-1]['close']
                    tr = max(tr, abs(self.high - prev_c), abs(self.low - prev_c))
                if self.atr is None:
                    self.atr = tr
                else:
                    self.atr = (tr + 13.0 * self.atr) / 14.0
                completed_bar['atr'] = self.atr
                
                # Pivot detector
                completed_bar['pivot_high'] = np.nan
                completed_bar['pivot_low'] = np.nan
                if self.completed_history:
                    prev = self.completed_history[-1]
                    is_green_red = (prev['close'] > prev['open']) and (self.close < self.open)
                    is_red_green = (prev['close'] < prev['open']) and (self.close > self.open)
                    if is_green_red:
                        completed_bar['pivot_high'] = max(prev['high'], self.high)
                    if is_red_green:
                        completed_bar['pivot_low'] = min(prev['low'], self.low)
            
            self.completed_history.append(completed_bar)
            if len(self.completed_history) > 10:
                self.completed_history.pop(0)
                
            self.current_bar_start_min = bin_start_min
            self.open = o
            self.high = h
            self.low = l
            self.close = c
            self.volume = v
        else:
            self.high = max(self.high, h)
            self.low = min(self.low, l)
            self.close = c
            self.volume += v
            
        return completed_bar

class StructureEngine:
    def __init__(self):
        self.history = []
        self.last_ph = None
        self.last_pl = None
        self.fvg_bull_top = None
        self.fvg_bull_bottom = None
        self.fvg_bull_revisited = False
        self.fvg_bear_top = None
        self.fvg_bear_bottom = None
        self.fvg_bear_revisited = False
        
        self.bos_up = 0.0
        self.bos_down = 0.0
        self.fvg_bull = 0.0
        self.fvg_bear = 0.0
        self.fvg_bull_inv = 0.0
        self.fvg_bear_inv = 0.0
        self.fvg_bull_top_val = np.nan
        self.fvg_bull_bottom_val = np.nan
        self.fvg_bear_top_val = np.nan
        self.fvg_bear_bottom_val = np.nan

    def update(self, t_ms, o, h, l, c):
        self.history.append({'open': o, 'high': h, 'low': l, 'close': c})
        if len(self.history) > 10:
            self.history.pop(0)
            
        n = len(self.history)
        self.bos_up = 0.0
        self.bos_down = 0.0
        self.fvg_bull = 0.0
        self.fvg_bear = 0.0
        self.fvg_bull_inv = 0.0
        self.fvg_bear_inv = 0.0
        
        ph = None
        pl = None
        if n >= 5:
            h3 = self.history[-3]['high']
            l3 = self.history[-3]['low']
            if h3 > max(self.history[-5]['high'], self.history[-4]['high'], self.history[-2]['high'], self.history[-1]['high']):
                ph = h3
            if l3 < min(self.history[-5]['low'], self.history[-4]['low'], self.history[-2]['low'], self.history[-1]['low']):
                pl = l3
                
        if ph is not None: self.last_ph = ph
        if pl is not None: self.last_pl = pl
            
        if self.last_ph is not None and c > self.last_ph:
            self.bos_up = 1.0
            self.last_ph = None
        if self.last_pl is not None and c < self.last_pl:
            self.bos_down = 1.0
            self.last_pl = None
            
        if n >= 3:
            curr_low = self.history[-1]['low']
            curr_high = self.history[-1]['high']
            h2 = self.history[-3]['high']
            l2 = self.history[-3]['low']
            if curr_low > h2:
                self.fvg_bull_top = curr_low
                self.fvg_bull_bottom = h2
                self.fvg_bull_revisited = False
            if curr_high < l2:
                self.fvg_bear_top = l2
                self.fvg_bear_bottom = curr_high
                self.fvg_bear_revisited = False
                
        if self.fvg_bull_top is not None:
            curr_low = self.history[-1]['low']
            curr_close = self.history[-1]['close']
            if curr_low < self.fvg_bull_top:
                self.fvg_bull_revisited = True
            if self.fvg_bull_revisited and curr_close > self.fvg_bull_top and curr_low > self.fvg_bull_bottom:
                self.fvg_bull = 1.0
                self.fvg_bull_top_val = self.fvg_bull_top
                self.fvg_bull_bottom_val = self.fvg_bull_bottom
                self.fvg_bull_top = None
                self.fvg_bull_bottom = None
                self.fvg_bull_revisited = False
            elif curr_low <= self.fvg_bull_bottom:
                self.fvg_bull_inv = 1.0
                self.fvg_bull_top = None
                self.fvg_bull_bottom = None
                self.fvg_bull_revisited = False
                
        if self.fvg_bear_bottom is not None:
            curr_high = self.history[-1]['high']
            curr_close = self.history[-1]['close']
            if curr_high > self.fvg_bear_bottom:
                self.fvg_bear_revisited = True
            if self.fvg_bear_revisited and curr_close < self.fvg_bear_bottom and curr_high < self.fvg_bear_top:
                self.fvg_bear = 1.0
                self.fvg_bear_top_val = self.fvg_bear_top
                self.fvg_bear_bottom_val = self.fvg_bear_bottom
                self.fvg_bear_top = None
                self.fvg_bear_bottom = None
                self.fvg_bear_revisited = False
            elif curr_high >= self.fvg_bear_top:
                self.fvg_bear_inv = 1.0
                self.fvg_bear_top = None
                self.fvg_bear_bottom = None
                self.fvg_bear_revisited = False

# =====================================================================
# 3. MAIN STREAMING TRANSLATOR
# =====================================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Pine Translator Streaming Engine")
    parser.add_argument('--input', type=str, default="mnq_raw_data.csv", help="Path to raw 1-minute OHLCV data")
    parser.add_argument('--output', type=str, default="translated_tv_export.csv", help="Path to output translated TV export")
    args = parser.parse_args()
    
    raw_path = args.input
    if not os.path.isabs(raw_path):
        raw_path = os.path.join("C:\\Users\\felix\\.gemini\\antigravity\\scratch\\mnq_liquidity_feature_engine", raw_path)
        
    print(f"Loading raw 1-minute OHLCV data from {raw_path}...")
    
    # Read CSV with optimized types
    df_raw = pd.read_csv(raw_path, dtype={
        'Time': np.int64,
        'Open': np.float32,
        'High': np.float32,
        'Low': np.float32,
        'Close': np.float32,
        'Volume': np.float32
    })
    df_raw = df_raw.sort_values('Time').reset_index(drop=True)
    n = len(df_raw)
    
    times = df_raw['Time'].values
    opens = df_raw['Open'].values
    highs = df_raw['High'].values
    lows = df_raw['Low'].values
    closes = df_raw['Close'].values
    volumes = df_raw['Volume'].values
    
    # Pre-allocate output arrays as float32 to conserve memory
    out_sweep_direction = np.full(n, np.nan, dtype=np.float32)
    out_liquidity_type = np.full(n, np.nan, dtype=np.float32)
    out_liquidity_strength = np.full(n, np.nan, dtype=np.float32)
    out_setup_origin = np.full(n, np.nan, dtype=np.float32)
    out_bos_up_strength = np.full(n, np.nan, dtype=np.float32)
    out_bos_down_strength = np.full(n, np.nan, dtype=np.float32)
    out_bullish_fvg_rejected = np.full(n, np.nan, dtype=np.float32)
    out_bearish_fvg_rejected = np.full(n, np.nan, dtype=np.float32)
    out_retracement_depth = np.full(n, np.nan, dtype=np.float32)
    out_distance_to_equilibrium = np.full(n, np.nan, dtype=np.float32)
    out_time_since_sweep = np.full(n, np.nan, dtype=np.float32)
    out_ny_session = np.full(n, np.nan, dtype=np.float32)
    out_london_session = np.full(n, np.nan, dtype=np.float32)
    out_asian_session = np.full(n, np.nan, dtype=np.float32)
    out_suggested_tp = np.full(n, np.nan, dtype=np.float32)
    out_suggested_sl = np.full(n, np.nan, dtype=np.float32)
    
    # Initialize streaming bars & engines
    bar_5m = StreamingBar(5)
    bar_1h = StreamingBar(60)
    bar_4h = StreamingBar(240)
    bar_daily = StreamingBar(1440)
    
    struct_1m = StructureEngine()
    struct_5m = StructureEngine()
    
    # Session time variables
    tz_ny = pytz.timezone('America/New_York')
    
    # Active S/R lists
    active_highs_1h = []
    active_lows_1h = []
    active_highs_4h = []
    active_lows_4h = []
    
    pdh_val = np.nan
    pdl_val = np.nan
    pdh_breached = False
    pdl_breached = False
    prev_date = None
    
    # State machine variables
    state = 0
    sweep_dir = 0
    valid_setup = 0
    sweep_extreme = np.nan
    current_extreme = np.nan
    sweep_time = np.nan
    sweep_bar_index = np.nan
    liquidity_type = 0
    liquidity_strength = 0.0
    
    setup_origin_tf = ""
    setup_origin_bar_index = np.nan
    setup_reason = ""
    setup_reason_output = ""
    
    trade_active = False
    trade_dir = 0
    trade_sl = np.nan
    trade_tp = np.nan
    setups_in_current_sweep = 0
    
    # Confirmations tracking
    active_setup_1m_bull = False
    active_setup_1m_bear = False
    active_setup_5m_bull = False
    active_setup_5m_bear = False
    
    setup_origin_bar_index_1m_bull = np.nan
    setup_origin_bar_index_1m_bear = np.nan
    setup_origin_bar_index_5m_bull = np.nan
    setup_origin_bar_index_5m_bear = np.nan
    
    setup_reason_1m_bull = ""
    setup_reason_1m_bear = ""
    setup_reason_5m_bull = ""
    setup_reason_5m_bear = ""
    
    # New Dual-Gate Confirmation Variables
    reversal_confirmed = False
    continuation_confirmed = False
    bos_occurred = False
    fvg_occurred = False
    reversal_tf = ""
    continuation_tf = ""
    
    # Wilder's 1m ATR
    atr_1m = 0.0
    
    # Log tables for retrospective sweep search and confirmations reconstruction
    pdh_sweep_log = np.zeros(n, dtype=bool)
    pdl_sweep_log = np.zeros(n, dtype=bool)
    sw_4h_high_log = np.zeros(n, dtype=bool)
    sw_4h_low_log = np.zeros(n, dtype=bool)
    sw_1h_high_log = np.zeros(n, dtype=bool)
    sw_1h_low_log = np.zeros(n, dtype=bool)
    str_4h_high_log = np.zeros(n, dtype=np.float32)
    str_4h_low_log = np.zeros(n, dtype=np.float32)
    str_1h_high_log = np.zeros(n, dtype=np.float32)
    str_1h_low_log = np.zeros(n, dtype=np.float32)
    
    bos_up_1m_log = np.zeros(n, dtype=bool)
    bos_down_1m_log = np.zeros(n, dtype=bool)
    fvg_bull_1m_log = np.zeros(n, dtype=bool)
    fvg_bear_1m_log = np.zeros(n, dtype=bool)
    fvg_bull_inv_1m_log = np.zeros(n, dtype=bool)
    fvg_bear_inv_1m_log = np.zeros(n, dtype=bool)
    
    bos_up_5m_log = np.zeros(n, dtype=bool)
    bos_down_5m_log = np.zeros(n, dtype=bool)
    fvg_bull_5m_log = np.zeros(n, dtype=bool)
    fvg_bear_5m_log = np.zeros(n, dtype=bool)
    fvg_bull_inv_5m_log = np.zeros(n, dtype=bool)
    fvg_bear_inv_5m_log = np.zeros(n, dtype=bool)
    
    print("Running chronological streaming loop...")
    for i in range(n):
        bar_time = times[i]
        o = opens[i]
        h = highs[i]
        l = lows[i]
        c = closes[i]
        v = volumes[i]
        
        # Wilder's ATR(14)
        tr = h - l
        if i > 0:
            tr = max(tr, abs(h - closes[i-1]), abs(l - closes[i-1]))
            atr_1m = (tr + 13.0 * atr_1m) / 14.0
        else:
            atr_1m = tr
            
        # Update streaming bars
        comp_5m = bar_5m.update(bar_time, o, h, l, c, v)
        comp_1h = bar_1h.update(bar_time, o, h, l, c, v)
        comp_4h = bar_4h.update(bar_time, o, h, l, c, v)
        comp_daily = bar_daily.update(bar_time, o, h, l, c, v)
        
        # Update 1m structure
        struct_1m.update(bar_time, o, h, l, c)
        
        # If 5m completed, update 5m structure
        if comp_5m is not None:
            struct_5m.update(comp_5m['time_min'] * 60000, comp_5m['open'], comp_5m['high'], comp_5m['low'], comp_5m['close'])
            
        # If 1H completed, update 1H levels
        if comp_1h is not None:
            update_mitigations(active_highs_1h, True, comp_1h['close'])
            update_mitigations(active_lows_1h, False, comp_1h['close'])
            if not pd.isna(comp_1h['pivot_high']):
                add_liquidity_pool(active_highs_1h, comp_1h['pivot_high'], comp_1h['atr'], True)
            if not pd.isna(comp_1h['pivot_low']):
                add_liquidity_pool(active_lows_1h, comp_1h['pivot_low'], comp_1h['atr'], False)
                
        # If 4H completed, update 4H levels
        if comp_4h is not None:
            update_mitigations(active_highs_4h, True, comp_4h['close'])
            update_mitigations(active_lows_4h, False, comp_4h['close'])
            if not pd.isna(comp_4h['pivot_high']):
                add_liquidity_pool(active_highs_4h, comp_4h['pivot_high'], comp_4h['atr'], True)
            if not pd.isna(comp_4h['pivot_low']):
                add_liquidity_pool(active_lows_4h, comp_4h['pivot_low'], comp_4h['atr'], False)
                
        # If Daily completed, update Daily PDH/PDL and reset breaches
        if comp_daily is not None:
            pdh_val = comp_daily['high']
            pdl_val = comp_daily['low']
            pdh_breached = False
            pdl_breached = False
            
        # Timezone conversions for session filtering
        dt_ny = pd.Timestamp(bar_time, unit='ms', tz='UTC').tz_convert(tz_ny)
        t_sec = dt_ny.hour * 3600 + dt_ny.minute * 60 + dt_ny.second
        ny_sess = 1.0 if (8 * 3600 <= t_sec < 17 * 3600) else 0.0
        london_sess = 1.0 if (3 * 3600 <= t_sec < 12 * 3600) else 0.0
        asian_sess = 1.0 if (t_sec >= 19 * 3600 or t_sec < 3 * 3600) else 0.0
        
        # Daily sweeps
        pdh_active = not pdh_breached
        pdl_active = not pdl_breached
        
        pdh_sweep = False
        if pdh_active and not pd.isna(pdh_val):
            if h >= pdh_val:
                pdh_breached = True
                pdh_sweep = True
                
        pdl_sweep = False
        if pdl_active and not pd.isna(pdl_val):
            if l <= pdl_val:
                pdl_breached = True
                pdl_sweep = True
                
        pdh_sweep_log[i] = pdh_sweep
        pdl_sweep_log[i] = pdl_sweep
        
        # Sweep checks on active levels
        sw_4h_high, str_4h_high = check_sweep(active_highs_4h, True, h, l)
        sw_4h_low, str_4h_low = check_sweep(active_lows_4h, False, h, l)
        sw_1h_high, str_1h_high = check_sweep(active_highs_1h, True, h, l)
        sw_1h_low, str_1h_low = check_sweep(active_lows_1h, False, h, l)
        
        sw_4h_high_log[i] = sw_4h_high
        sw_4h_low_log[i] = sw_4h_low
        sw_1h_high_log[i] = sw_1h_high
        sw_1h_low_log[i] = sw_1h_low
        str_4h_high_log[i] = str_4h_high
        str_4h_low_log[i] = str_4h_low
        str_1h_high_log[i] = str_1h_high
        str_1h_low_log[i] = str_1h_low
        
        # Decay active levels
        prev_time = times[i-1] if i > 0 else np.nan
        elapsed = bar_time - prev_time if not pd.isna(prev_time) else 0.0
        decay_amt = elapsed * 0.9 / 259200000.0 if i > 0 else 0.0
        for p in active_highs_4h:
            if p['active']:
                p['strength_score'] -= decay_amt
                if p['strength_score'] < 0.1: p['active'] = False
        for p in active_lows_4h:
            if p['active']:
                p['strength_score'] -= decay_amt
                if p['strength_score'] < 0.1: p['active'] = False
        for p in active_highs_1h:
            if p['active']:
                p['strength_score'] -= decay_amt
                if p['strength_score'] < 0.1: p['active'] = False
        for p in active_lows_1h:
            if p['active']:
                p['strength_score'] -= decay_amt
                if p['strength_score'] < 0.1: p['active'] = False
                
        # Limit capacity to prevent memory bloat
        if len(active_highs_4h) > 30: active_highs_4h = active_highs_4h[-30:]
        if len(active_lows_4h) > 30: active_lows_4h = active_lows_4h[-30:]
        if len(active_highs_1h) > 30: active_highs_1h = active_highs_1h[-30:]
        if len(active_lows_1h) > 30: active_lows_1h = active_lows_1h[-30:]
        
        # Trade resolution check
        if trade_active:
            sl_hit = False
            tp_hit = False
            if trade_dir == 1:
                if l <= trade_sl: sl_hit = True
                elif h >= trade_tp: tp_hit = True
            elif trade_dir == -1:
                if h >= trade_sl: sl_hit = True
                elif l <= trade_tp: tp_hit = True
                
            if sl_hit or tp_hit:
                trade_active = False
                trade_sl = np.nan
                trade_tp = np.nan
                trade_dir = 0
                
                # Reset state machine
                state = 0
                sweep_dir = 0
                valid_setup = 0
                sweep_time = np.nan
                sweep_extreme = np.nan
                current_extreme = np.nan
                sweep_bar_index = np.nan
                liquidity_type = 0
                liquidity_strength = 0.0
                setup_origin_tf = ""
                setup_origin_bar_index = np.nan
                setup_reason = ""
                setup_reason_output = ""
                
                reversal_confirmed = False
                continuation_confirmed = False
                bos_occurred = False
                fvg_occurred = False
                reversal_tf = ""
                continuation_tf = ""
                
                # Retrospective pass (lookback 10 bars)
                lookback = 10
                found_sweep_offset = -1
                found_sweep_dir = 0
                found_sweep_extreme = np.nan
                found_sweep_time = np.nan
                found_sweep_bar_index = np.nan
                found_l_type = 0
                found_l_strength = 0.0
                
                for offset in range(lookback):
                    idx_check = i - offset
                    if idx_check < 0:
                        break
                    is_h_sweep = pdh_sweep_log[idx_check] or sw_4h_high_log[idx_check] or sw_1h_high_log[idx_check]
                    is_l_sweep = pdl_sweep_log[idx_check] or sw_4h_low_log[idx_check] or sw_1h_low_log[idx_check]
                    
                    if is_h_sweep or is_l_sweep:
                        s_dir = -1 if is_h_sweep else 1
                        s_extreme = highs[idx_check] if s_dir == -1 else lows[idx_check]
                        invalidated = False
                        
                        if offset > 0:
                            for j in range(offset):
                                idx_mid = i - j
                                if s_dir == -1:
                                    if highs[idx_mid] > s_extreme:
                                        invalidated = True
                                        break
                                else:
                                    if lows[idx_mid] < s_extreme:
                                        invalidated = True
                                        break
                                        
                        if not invalidated:
                            found_sweep_offset = offset
                            found_sweep_dir = s_dir
                            found_sweep_extreme = s_extreme
                            found_sweep_time = times[idx_check]
                            found_sweep_bar_index = idx_check
                            
                            if is_h_sweep:
                                if sw_4h_high_log[idx_check]:
                                    found_l_type = 3
                                    found_l_strength = str_4h_high_log[idx_check]
                                elif sw_1h_high_log[idx_check]:
                                    found_l_type = 2
                                    found_l_strength = str_1h_high_log[idx_check]
                                else:
                                    found_l_type = 1
                                    found_l_strength = 1.0
                            else:
                                if sw_4h_low_log[idx_check]:
                                    found_l_type = 3
                                    found_l_strength = str_4h_low_log[idx_check]
                                elif sw_1h_low_log[idx_check]:
                                    found_l_type = 2
                                    found_l_strength = str_1h_low_log[idx_check]
                                else:
                                    found_l_type = 1
                                    found_l_strength = 1.0
                            break
                            
                if found_sweep_offset != -1:
                    state = 2
                    sweep_dir = found_sweep_dir
                    sweep_extreme = found_sweep_extreme
                    sweep_time = found_sweep_time
                    sweep_bar_index = found_sweep_bar_index
                    liquidity_type = found_l_type
                    liquidity_strength = found_l_strength
                    
                    reversal_confirmed = False
                    continuation_confirmed = False
                    bos_occurred = False
                    fvg_occurred = False
                    reversal_tf = ""
                    continuation_tf = ""
                    
                    # Reconstruct current_extreme and confluences chronologically
                    extreme_val = highs[found_sweep_bar_index] if found_sweep_dir == 1 else lows[found_sweep_bar_index]
                    for k in range(found_sweep_offset + 1):
                        idx_k = i - (found_sweep_offset - k)
                        if found_sweep_dir == 1:
                            extreme_val = max(extreme_val, highs[idx_k])
                        else:
                            extreme_val = min(extreme_val, lows[idx_k])
                        current_extreme = extreme_val
                        
                        eq_val = (found_sweep_extreme + current_extreme) / 2.0
                        
                        if found_sweep_dir == 1: # Bullish setup
                            bos_ev = bos_up_1m_log[idx_k] or bos_up_5m_log[idx_k]
                            inv_ev = fvg_bear_inv_1m_log[idx_k] or fvg_bear_inv_5m_log[idx_k]
                            fill_ev = fvg_bull_1m_log[idx_k] or fvg_bull_5m_log[idx_k]
                            eq_touch = (lows[idx_k] <= eq_val)
                            
                            if bos_ev or inv_ev:
                                reversal_confirmed = True
                            if bos_ev:
                                bos_occurred = True
                            if inv_ev:
                                fvg_occurred = True
                                
                            if fill_ev or eq_touch:
                                continuation_confirmed = True
                            if fill_ev:
                                fvg_occurred = True
                                
                            if bos_up_1m_log[idx_k] or fvg_bear_inv_1m_log[idx_k]:
                                reversal_tf = "1m"
                            elif bos_up_5m_log[idx_k] or fvg_bear_inv_5m_log[idx_k]:
                                if reversal_tf != "1m":
                                    reversal_tf = "5m"
                            if fvg_bull_1m_log[idx_k] or eq_touch:
                                continuation_tf = "1m"
                            elif fvg_bull_5m_log[idx_k]:
                                if continuation_tf != "1m":
                                    continuation_tf = "5m"
                        else: # Bearish setup
                            bos_ev = bos_down_1m_log[idx_k] or bos_down_5m_log[idx_k]
                            inv_ev = fvg_bull_inv_1m_log[idx_k] or fvg_bull_inv_5m_log[idx_k]
                            fill_ev = fvg_bear_1m_log[idx_k] or fvg_bear_5m_log[idx_k]
                            eq_touch = (highs[idx_k] >= eq_val)
                            
                            if bos_ev or inv_ev:
                                reversal_confirmed = True
                            if bos_ev:
                                bos_occurred = True
                            if inv_ev:
                                fvg_occurred = True
                                
                            if fill_ev or eq_touch:
                                continuation_confirmed = True
                            if fill_ev:
                                fvg_occurred = True
                                
                            if bos_down_1m_log[idx_k] or fvg_bull_inv_1m_log[idx_k]:
                                reversal_tf = "1m"
                            elif bos_down_5m_log[idx_k] or fvg_bull_inv_5m_log[idx_k]:
                                if reversal_tf != "1m":
                                    reversal_tf = "5m"
                            if fvg_bear_1m_log[idx_k] or eq_touch:
                                continuation_tf = "1m"
                            elif fvg_bear_5m_log[idx_k]:
                                if continuation_tf != "1m":
                                    continuation_tf = "5m"

        # Update confirmations and logs
        bos_up_1m_event = (struct_1m.bos_up == 1.0)
        bos_down_1m_event = (struct_1m.bos_down == 1.0)
        fvg_bull_1m_event = (struct_1m.fvg_bull == 1.0)
        fvg_bear_1m_event = (struct_1m.fvg_bear == 1.0)
        fvg_bull_inv_1m_event = (struct_1m.fvg_bull_inv == 1.0)
        fvg_bear_inv_1m_event = (struct_1m.fvg_bear_inv == 1.0)
        
        bos_up_5m_event = (struct_5m.bos_up == 1.0)
        bos_down_5m_event = (struct_5m.bos_down == 1.0)
        fvg_bull_5m_event = (struct_5m.fvg_bull == 1.0)
        fvg_bear_5m_event = (struct_5m.fvg_bear == 1.0)
        fvg_bull_inv_5m_event = (struct_5m.fvg_bull_inv == 1.0)
        fvg_bear_inv_5m_event = (struct_5m.fvg_bear_inv == 1.0)
        
        bos_up_1m_log[i] = bos_up_1m_event
        bos_down_1m_log[i] = bos_down_1m_event
        fvg_bull_1m_log[i] = fvg_bull_1m_event
        fvg_bear_1m_log[i] = fvg_bear_1m_event
        fvg_bull_inv_1m_log[i] = fvg_bull_inv_1m_event
        fvg_bear_inv_1m_log[i] = fvg_bear_inv_1m_event
        
        bos_up_5m_log[i] = bos_up_5m_event
        bos_down_5m_log[i] = bos_down_5m_event
        fvg_bull_5m_log[i] = fvg_bull_5m_event
        fvg_bear_5m_log[i] = fvg_bear_5m_event
        fvg_bull_inv_5m_log[i] = fvg_bull_inv_5m_event
        fvg_bear_inv_5m_log[i] = fvg_bear_inv_5m_event

        # Sweep activation
        sweep_buy_side = 1 if (pdh_sweep or sw_4h_high or sw_1h_high) else 0
        sweep_sell_side = 1 if (pdl_sweep or sw_4h_low or sw_1h_low) else 0
        
        l_type_curr = 0
        l_strength_curr = 0.0
        if sw_4h_high or sw_4h_low:
            l_type_curr = 3
            l_strength_curr = str_4h_high if sw_4h_high else str_4h_low
        elif sw_1h_high or sw_1h_low:
            l_type_curr = 2
            l_strength_curr = str_1h_high if sw_1h_high else str_1h_low
        elif pdh_sweep or pdl_sweep:
            l_type_curr = 1
            l_strength_curr = 1.0
            
        is_new_sweep_event = False
        new_sweep_dir = -1 if sweep_buy_side == 1 else 1
        is_new_sweep = (sweep_buy_side == 1 or sweep_sell_side == 1) and (state == 0 or (state == 2 and new_sweep_dir != sweep_dir))
        
        if is_new_sweep:
            is_new_sweep_event = True
            state = 2
            valid_setup = 0
            sweep_time = bar_time
            sweep_bar_index = i
            liquidity_type = l_type_curr
            liquidity_strength = l_strength_curr
            setup_origin_tf = ""
            setup_origin_bar_index = np.nan
            setup_reason = ""
            setup_reason_output = ""
            sweep_dir = new_sweep_dir
            if sweep_dir == -1:
                sweep_extreme = h
                current_extreme = l
            else:
                sweep_extreme = l
                current_extreme = h
                
            setups_in_current_sweep = 0
            trade_active = False
            trade_sl = np.nan
            trade_tp = np.nan
            trade_dir = 0
            
            reversal_confirmed = False
            continuation_confirmed = False
            bos_occurred = False
            fvg_occurred = False
            reversal_tf = ""
            continuation_tf = ""
            
        # Reversal Logic
        prev_valid_setup = valid_setup
        invalidation_triggered = False
        
        if state == 2:
            if sweep_dir == -1:
                current_extreme = min(current_extreme, l)
            else:
                current_extreme = max(current_extreme, h)
                
            if sweep_dir == -1:
                if h > sweep_extreme:
                    if valid_setup == 0:
                        sweep_extreme = h
                        sweep_bar_index = i
                    else:
                        invalidation_triggered = True
            elif sweep_dir == 1:
                if l < sweep_extreme:
                    if valid_setup == 0:
                        sweep_extreme = l
                        sweep_bar_index = i
                    else:
                        invalidation_triggered = True
                        
            if invalidation_triggered and valid_setup == 1:
                state = 0
                sweep_dir = 0
                valid_setup = 0
                sweep_time = np.nan
                sweep_extreme = np.nan
                current_extreme = np.nan
                sweep_bar_index = np.nan
                liquidity_type = 0
                liquidity_strength = 0.0
                setup_origin_tf = ""
                setup_origin_bar_index = np.nan
                setup_reason = ""
                setup_reason_output = ""
                reversal_confirmed = False
                continuation_confirmed = False
                bos_occurred = False
                fvg_occurred = False
                reversal_tf = ""
                continuation_tf = ""
            else:
                if sweep_dir == 1: # Bullish
                    bos_ev = bos_up_1m_event or bos_up_5m_event
                    inv_ev = fvg_bear_inv_1m_event or fvg_bear_inv_5m_event
                    fill_ev = fvg_bull_1m_event or fvg_bull_5m_event
                    eq_touch = (l <= (sweep_extreme + current_extreme) / 2.0)
                    
                    if bos_ev or inv_ev:
                        reversal_confirmed = True
                    if bos_ev:
                        bos_occurred = True
                    if inv_ev:
                        fvg_occurred = True
                        
                    if fill_ev or eq_touch:
                        continuation_confirmed = True
                    if fill_ev:
                        fvg_occurred = True
                        
                    if bos_up_1m_event or fvg_bear_inv_1m_event:
                        reversal_tf = "1m"
                    elif bos_up_5m_event or fvg_bear_inv_5m_event:
                        if reversal_tf != "1m":
                            reversal_tf = "5m"
                    if fvg_bull_1m_event or eq_touch:
                        continuation_tf = "1m"
                    elif fvg_bull_5m_event:
                        if continuation_tf != "1m":
                            continuation_tf = "5m"
                else: # Bearish
                    bos_ev = bos_down_1m_event or bos_down_5m_event
                    inv_ev = fvg_bull_inv_1m_event or fvg_bull_inv_5m_event
                    fill_ev = fvg_bear_1m_event or fvg_bear_5m_event
                    eq_touch = (h >= (sweep_extreme + current_extreme) / 2.0)
                    
                    if bos_ev or inv_ev:
                        reversal_confirmed = True
                    if bos_ev:
                        bos_occurred = True
                    if inv_ev:
                        fvg_occurred = True
                        
                    if fill_ev or eq_touch:
                        continuation_confirmed = True
                    if fill_ev:
                        fvg_occurred = True
                        
                    if bos_down_1m_event or fvg_bull_inv_1m_event:
                        reversal_tf = "1m"
                    elif bos_down_5m_event or fvg_bull_inv_5m_event:
                        if reversal_tf != "1m":
                            reversal_tf = "5m"
                    if fvg_bear_1m_event or eq_touch:
                        continuation_tf = "1m"
                    elif fvg_bear_5m_event:
                        if continuation_tf != "1m":
                            continuation_tf = "5m"
                            
                valid_setup = 1 if (reversal_confirmed and continuation_confirmed) else 0
                
                if valid_setup == 1 and prev_valid_setup == 0:
                    setup_origin_tf = "1m" if (reversal_tf == "1m" or continuation_tf == "1m") else "5m"
                    setup_reason = "BOS+FVG" if (bos_occurred and fvg_occurred) else ("BOS" if bos_occurred else ("FVG" if fvg_occurred else "EQ"))
                    setup_reason_output = setup_reason

        # Trade Entry
        sweep_direction_val = sweep_dir if state != 0 else 0
        prospective_sl = sweep_extreme
        prospective_tp = np.nan
        
        if sweep_direction_val == 1:
            prospective_tp = find_closest_high(active_highs_1h, active_highs_4h, pdh_val, pdh_active, c)
            if pd.isna(prospective_tp):
                prospective_tp = c + 2.0 * atr_1m
        elif sweep_direction_val == -1:
            prospective_tp = find_closest_low(active_lows_1h, active_lows_4h, pdl_val, pdl_active, c)
            if pd.isna(prospective_tp):
                prospective_tp = c - 2.0 * atr_1m
                
        risk = c - prospective_sl if sweep_direction_val == 1 else (prospective_sl - c if sweep_direction_val == -1 else 0.0)
        reward = prospective_tp - c if sweep_direction_val == 1 else (c - prospective_tp if sweep_direction_val == -1 else 0.0)
        
        rr_ratio = reward / risk if risk > 0.0 else 0.0
        rr_satisfied = rr_ratio >= 1.5
        
        can_emit_setup = (setups_in_current_sweep < 3) and not (trade_active and trade_dir == sweep_direction_val) and rr_satisfied
        setup_emitted = False
        
        if valid_setup == 1 and prev_valid_setup == 0 and can_emit_setup:
            setup_emitted = True
            setups_in_current_sweep += 1
            trade_active = True
            trade_dir = sweep_direction_val
            trade_sl = prospective_sl
            trade_tp = prospective_tp
            
        # Record Output geometry and sessions
        if valid_setup == 1:
            bos_confirmed = (setup_reason_output in ["BOS", "BOS+FVG"])
            fvg_confirmed = (setup_reason_output in ["FVG", "BOS+FVG"])
            
            impulse_size = abs(sweep_extreme - current_extreme)
            ret_depth = 0.0
            if impulse_size > 0.0:
                if sweep_dir == -1:
                    ret_depth = max(0.0, min(1.0, (c - current_extreme) / impulse_size))
                else:
                    ret_depth = max(0.0, min(1.0, (current_extreme - c) / impulse_size))
            
            equilibrium_val = (sweep_extreme + current_extreme) / 2.0
            dist_eq = 0.0
            if c != 0.0:
                dist_eq = sweep_dir * (c - equilibrium_val) / c
                
            time_since_sw = i - sweep_bar_index
            
            out_sweep_direction[i] = sweep_direction_val
            out_liquidity_type[i] = liquidity_type
            out_liquidity_strength[i] = liquidity_strength
            out_setup_origin[i] = 1.0 if setup_origin_tf == "1m" else 5.0
            
            if sweep_direction_val == 1:
                out_bos_up_strength[i] = 1.0 if bos_confirmed else 0.0
                out_bullish_fvg_rejected[i] = 1.0 if fvg_confirmed else 0.0
            elif sweep_direction_val == -1:
                out_bos_down_strength[i] = 1.0 if bos_confirmed else 0.0
                out_bearish_fvg_rejected[i] = 1.0 if fvg_confirmed else 0.0
                
            out_retracement_depth[i] = ret_depth
            out_distance_to_equilibrium[i] = dist_eq
            out_time_since_sweep[i] = time_since_sw
            out_ny_session[i] = ny_sess
            out_london_session[i] = london_sess
            out_asian_session[i] = asian_sess
            out_suggested_tp[i] = trade_tp if not pd.isna(trade_tp) else prospective_tp
            out_suggested_sl[i] = trade_sl if not pd.isna(trade_sl) else prospective_sl

    print("Writing computed features to the dataframe...")
    suffix = " (MNQ Liquidity Feature Engine (ML Export))"
    df_raw['sweep_direction' + suffix] = out_sweep_direction
    df_raw['liquidity_type' + suffix] = out_liquidity_type
    df_raw['liquidity_strength' + suffix] = out_liquidity_strength
    df_raw['setup_origin' + suffix] = out_setup_origin
    
    df_raw['bos_up_strength' + suffix] = out_bos_up_strength
    df_raw['bos_down_strength' + suffix] = out_bos_down_strength
    df_raw['bullish_fvg_rejected' + suffix] = out_bullish_fvg_rejected
    df_raw['bearish_fvg_rejected' + suffix] = out_bearish_fvg_rejected
    
    df_raw['bearish_displacement_size' + suffix] = np.nan
    df_raw['bullish_displacement_size' + suffix] = np.nan
    
    df_raw['retracement_depth' + suffix] = out_retracement_depth
    df_raw['distance_to_equilibrium' + suffix] = out_distance_to_equilibrium
    df_raw['time_since_sweep' + suffix] = out_time_since_sweep
    
    df_raw['ny_session' + suffix] = out_ny_session
    df_raw['london_session' + suffix] = out_london_session
    df_raw['asian_session' + suffix] = out_asian_session
    
    df_raw['suggested_tp' + suffix] = out_suggested_tp
    df_raw['suggested_sl' + suffix] = out_suggested_sl
    
    output_path = args.output
    if not os.path.isabs(output_path):
        output_path = os.path.join("C:\\Users\\felix\\.gemini\\antigravity\\scratch\\mnq_liquidity_feature_engine", output_path)
        
    print(f"Saving translated TV export to {output_path}...")
    
    final_cols = [
        'Time', 'Open', 'High', 'Low', 'Close', 'Volume',
        'sweep_direction' + suffix,
        'liquidity_type' + suffix,
        'liquidity_strength' + suffix,
        'setup_origin' + suffix,
        'bos_up_strength' + suffix,
        'bos_down_strength' + suffix,
        'bullish_fvg_rejected' + suffix,
        'bearish_fvg_rejected' + suffix,
        'bearish_displacement_size' + suffix,
        'bullish_displacement_size' + suffix,
        'retracement_depth' + suffix,
        'distance_to_equilibrium' + suffix,
        'time_since_sweep' + suffix,
        'ny_session' + suffix,
        'london_session' + suffix,
        'asian_session' + suffix,
        'suggested_tp' + suffix,
        'suggested_sl' + suffix
    ]
    
    df_out = df_raw[final_cols]
    df_out.to_csv(output_path, index=False)
    print("Translation completed successfully!")

if __name__ == '__main__':
    main()
