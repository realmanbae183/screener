"""
과매도 지표 계산 엔진.
모든 함수는 OHLCV DataFrame(컬럼: Open/High/Low/Close/Volume)을 받아
Series 또는 스칼라를 반환한다. 미래 데이터를 참조하지 않는다(look-ahead 방지).
"""
import numpy as np
import pandas as pd

import config as C


# ═══════════════════════════════════════════
# 기본 지표
# ═══════════════════════════════════════════

def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def atr(df: pd.DataFrame, n: int = C.ATR_PERIOD) -> pd.Series:
    """Average True Range — 변동성 자(ruler). Wilder 평활 사용."""
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def rsi(close: pd.Series, n: int = C.RSI_PERIOD) -> pd.Series:
    """Relative Strength Index — 자기 자신의 최근 n일 내 상승/하락 강도 비율."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    avg_loss = loss.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(100.0).where(avg_loss.notna())


def bollinger_percent_b(close: pd.Series, n: int = C.BB_PERIOD,
                        k: float = C.BB_STD) -> pd.Series:
    """%B — 0 미만이면 하단 밴드 이탈(통계적 이상치)."""
    mid = close.rolling(n, min_periods=n).mean()
    sd = close.rolling(n, min_periods=n).std(ddof=0)
    upper, lower = mid + k * sd, mid - k * sd
    width = (upper - lower).replace(0, np.nan)
    return (close - lower) / width


def macd(close: pd.Series):
    """MACD line, signal, histogram."""
    ema_f = close.ewm(span=C.MACD_FAST, adjust=False).mean()
    ema_s = close.ewm(span=C.MACD_SLOW, adjust=False).mean()
    line = ema_f - ema_s
    signal = line.ewm(span=C.MACD_SIGNAL, adjust=False).mean()
    return line, signal, line - signal


# ═══════════════════════════════════════════
# ② 추세 필터 — 떨어지는 칼날 차단
# ═══════════════════════════════════════════

def trend_ok(df: pd.DataFrame) -> tuple[bool, dict]:
    """
    장기 추세가 살아있는지 판정.
    통과: 종가가 200일선 근처 이상  OR  200일선이 우상향.
    """
    close = df["Close"]
    ma = sma(close, C.TREND_MA)
    if ma.isna().iloc[-1]:
        return False, {"reason": "히스토리 부족"}

    ma_now = float(ma.iloc[-1])
    ma_prev = float(ma.iloc[-1 - C.TREND_SLOPE_DAYS])
    px = float(close.iloc[-1])

    above = px > ma_now * C.TREND_MA_TOLERANCE
    rising = ma_now > ma_prev
    ok = (above and rising) if C.TREND_REQUIRE_BOTH else (above or rising)

    return ok, {
        "price": px,
        "ma200": ma_now,
        "price_vs_ma200": px / ma_now - 1,
        "ma200_slope": ma_now / ma_prev - 1,
        "above_ma200": above,
        "ma200_rising": rising,
    }


# ═══════════════════════════════════════════
# ②-b 구조 필터 — 하락 채널 차단
# ═══════════════════════════════════════════

def _swing_highs(s: pd.Series, w: int, min_gap: int) -> list[int]:
    """좌우 w봉보다 엄격히 높은 국소 고점."""
    vals = s.to_numpy()
    out: list[int] = []
    for i in range(w, len(vals) - w):
        v = vals[i]
        if np.isnan(v):
            continue
        if v > vals[i - w:i].max() and v > vals[i + 1:i + w + 1].max():
            if out and i - out[-1] < min_gap:
                if v > vals[out[-1]]:
                    out[-1] = i
                continue
            out.append(i)
    return out


def structure_ok(df: pd.DataFrame) -> tuple[bool, dict]:
    """
    하락 채널(descending channel) 판정.

    200일선 기울기는 200일치를 평균하므로 방향 전환이 몇 달 늦다.
    넉 달을 내리 빠져도 그 전 상승분이 남아 기울기가 양수로 나온다.
    그래서 더 짧은 이평선의 배열과 고점 흐름으로 현재 구조를 따로 본다.

    차단 조건 (둘 다 충족):
      1) 이평선 역배열  (20일선 < 60일선 < 120일선)
      2) 120일선이 하락 중
    추가로 고점이 계속 낮아지고 저점 근처에 있으면 함께 차단한다.
    """
    close = df["Close"]
    if len(close) < C.STRUCT_MA_LONG + 20:
        return True, {"reason": "히스토리 부족 — 구조 판정 생략"}

    ma_s = sma(close, C.STRUCT_MA_SHORT)
    ma_m = sma(close, C.STRUCT_MA_MID)
    ma_l = sma(close, C.STRUCT_MA_LONG)
    if ma_l.isna().iloc[-1]:
        return True, {"reason": "히스토리 부족 — 구조 판정 생략"}

    s, m, l = float(ma_s.iloc[-1]), float(ma_m.iloc[-1]), float(ma_l.iloc[-1])
    l_prev = float(ma_l.iloc[-1 - C.STRUCT_SLOPE_DAYS])
    l_slope = l / l_prev - 1

    inverted = s < m < l                       # 역배열
    long_falling = l_slope <= -C.STRUCT_SLOPE_MIN

    win = close.tail(C.STRUCT_LOOKBACK)

    # ★ 급락 직후와 장기 하락 채널을 가르는 것은 '시간'이다.
    #   어떤 종목이든 급락 직후에는 이평선이 역배열이 된다. 그것만으로
    #   차단하면 우리가 사려는 눌림목까지 전부 막힌다.
    #   고점을 찍은 지 얼마나 됐는지로 둘을 구분한다.
    hi_pos = int(np.argmax(win.to_numpy()))
    bars_since_high = len(win) - 1 - hi_pos

    # 고점 흐름 — 고점이 계단식으로 낮아지는가
    highs = _swing_highs(win, C.DIV_PIVOT_WINDOW, C.DIV_MIN_GAP)
    lower_highs = False
    if len(highs) >= 2:
        h1, h2 = float(win.iloc[highs[-2]]), float(win.iloc[highs[-1]])
        lower_highs = h2 < h1 * (1 - C.STRUCT_LOWER_HIGH_MIN)

    # 최근 저점 대비 현재 위치 (저점 바로 위면 지지가 얇다)
    lo = float(win.min())
    px = float(close.iloc[-1])
    above_low = px / lo - 1 if lo > 0 else np.nan

    info = {
        "ma_short": round(s, 2), "ma_mid": round(m, 2), "ma_long": round(l, 2),
        "ma_long_slope": round(l_slope, 4),
        "inverted": bool(inverted), "long_falling": bool(long_falling),
        "lower_highs": bool(lower_highs),
        "bars_since_high": int(bars_since_high),
        "above_recent_low": None if np.isnan(above_low) else round(above_low, 4),
    }

    # 고점을 찍은 지 얼마 안 됐으면 '급락 직후'이지 하락 채널이 아니다
    if bars_since_high < C.STRUCT_MIN_DECLINE_BARS:
        info["reason"] = f"급락 직후 ({bars_since_high}봉 전 고점) — 구조 판정 유보"
        return True, info

    # 오래 흘러내리는 중 + 역배열 + 장기선 하락 = 하락 채널
    if inverted and long_falling:
        info["reason"] = (f"하락 채널 — {bars_since_high}봉째 하락, 이평 역배열, "
                          f"{C.STRUCT_MA_LONG}일선 {l_slope*100:+.1f}%")
        return False, info

    # 고점이 계단식으로 낮아지면서 저점 코앞 = 지지가 얇다
    if lower_highs and long_falling and not np.isnan(above_low) \
            and above_low < C.STRUCT_NEAR_LOW:
        info["reason"] = (f"고점 낮아짐 + 저점 근접 (저점 대비 +{above_low*100:.1f}%)")
        return False, info

    return True, info


# ═══════════════════════════════════════════
# ③ 과매도 지표 7종
# ═══════════════════════════════════════════

def atr_distance(df: pd.DataFrame) -> float:
    """
    20일선에서 몇 ATR 아래였는가 — 종목별 변동성으로 정규화된 낙폭.
    최근 SIGNAL_WINDOW 봉 중 '가장 늘어났던' 값을 쓴다.
    (냉각 기간을 기다리는 동안 20일선이 가격을 따라잡기 때문)
    """
    close = df["Close"]
    ma = sma(close, C.ATR_MA)
    a = atr(df)
    dist = (ma - close) / a.replace(0, np.nan)
    win = dist.tail(C.SIGNAL_WINDOW).dropna()
    if win.empty:
        return np.nan
    return float(win.max())


def drawdown_from_high(df: pd.DataFrame) -> float:
    """52주 고점 대비 낙폭 (양수 = 하락)."""
    window = df["Close"].tail(C.DRAWDOWN_LOOKBACK)
    if len(window) < 60:
        return np.nan
    return float(1 - window.iloc[-1] / window.max())


def fib_retracement(df: pd.DataFrame) -> tuple[float, bool]:
    """
    직전 스윙 고점→저점 대비 현재 위치.
    고점 이후 저점이 형성된 하락 국면에서만 의미가 있다.
    반환: (되돌림 비율, 38.2~61.8% 구간 내 여부)
    """
    window = df["Close"].tail(C.FIB_LOOKBACK)
    if len(window) < 60:
        return np.nan, False

    hi_idx = window.idxmax()
    hi = float(window.loc[hi_idx])
    after = window.loc[hi_idx:]
    if len(after) < 5:
        return np.nan, False

    lo = float(after.min())
    if hi <= lo:
        return np.nan, False

    px = float(window.iloc[-1])
    # 스윙 폭(고점-저점) 대비 현재가가 고점에서 얼마나 내려왔는가
    ratio = (hi - px) / (hi - lo)
    in_zone = bool(C.FIB_LOW <= ratio <= C.FIB_HIGH)
    return float(ratio), in_zone


def volume_climax(df: pd.DataFrame) -> tuple[bool, dict]:
    """
    투매 클라이맥스 판정.
    최근 구간에 평균 대비 N배 거래량이 터진 날이 있고,
    그 이후 거래량이 줄어들고 있으면 True (팔 사람이 다 팔았다).
    """
    vol = df["Volume"]
    if len(vol) < C.VOL_AVG_PERIOD + C.VOL_CLIMAX_LOOKBACK:
        return False, {}

    avg = vol.rolling(C.VOL_AVG_PERIOD, min_periods=C.VOL_AVG_PERIOD).mean()
    recent = vol.tail(C.VOL_CLIMAX_LOOKBACK)
    recent_avg = avg.tail(C.VOL_CLIMAX_LOOKBACK)
    ratio = recent / recent_avg

    spikes = ratio[ratio >= C.VOL_CLIMAX_MULT]
    if spikes.empty:
        return False, {"max_vol_ratio": float(ratio.max())}

    spike_pos = recent.index.get_loc(spikes.index[-1])
    after_ratio = ratio.iloc[spike_pos + 1:]
    if len(after_ratio) < C.VOL_DECLINE_DAYS:
        return False, {"climax_too_recent": True,
                       "max_vol_ratio": float(spikes.iloc[-1])}

    # 클라이맥스 이후 거래량이 확연히 줄었는가 (배수끼리 비교)
    climax_ratio = float(spikes.iloc[-1])
    after_avg = float(after_ratio.tail(C.VOL_DECLINE_DAYS).mean())
    declining = after_avg < climax_ratio * C.VOL_DECLINE_FRAC
    return bool(declining), {
        "climax_ratio": round(climax_ratio, 2),
        "after_ratio": round(after_avg, 2),
        "days_since_climax": int(len(after_ratio)),
        "volume_declining": declining,
    }


def _local_lows(s: pd.Series, w: int, min_gap: int) -> list[int]:
    """
    좌우 w봉보다 '엄격히' 낮은 국소 저점의 위치 인덱스.
    평탄 구간에서 무의미한 피벗이 양산되는 것을 막고,
    피벗 간 최소 간격(min_gap)을 강제해 인접 피벗 비교를 방지한다.
    """
    vals = s.to_numpy()
    out: list[int] = []
    for i in range(w, len(vals) - w):
        v = vals[i]
        if np.isnan(v):
            continue
        left, right = vals[i - w:i], vals[i + 1:i + w + 1]
        if v < left.min() and v < right.min():          # 엄격 부등호
            if out and i - out[-1] < min_gap:
                if v < vals[out[-1]]:                    # 더 깊은 쪽만 남김
                    out[-1] = i
                continue
            out.append(i)
    return out


def macd_bullish_divergence(df: pd.DataFrame) -> tuple[bool, dict]:
    """
    강세 다이버전스: 가격은 더 낮은 저점, MACD는 더 높은 저점.
    → 가격은 떨어지는데 하락 모멘텀은 약해지는 중 = 바닥 근처 선행 신호.
    (골든크로스는 이미 반등 시작 후라 바닥 포착엔 늦다)
    """
    close = df["Close"].tail(C.DIV_LOOKBACK)
    if len(close) < C.DIV_LOOKBACK:
        return False, {}

    line, _, _ = macd(df["Close"])
    line = line.tail(C.DIV_LOOKBACK)

    lows = _local_lows(close, C.DIV_PIVOT_WINDOW, C.DIV_MIN_GAP)
    if len(lows) < 2:
        return False, {"pivots_found": len(lows)}

    i1, i2 = lows[-2], lows[-1]
    p1, p2 = float(close.iloc[i1]), float(close.iloc[i2])
    m1, m2 = float(line.iloc[i1]), float(line.iloc[i2])

    diverging = (p2 < p1) and (m2 > m1)
    return bool(diverging), {
        "prev_low": round(p1, 2), "last_low": round(p2, 2),
        "prev_macd": round(m1, 3), "last_macd": round(m2, 3),
        "bars_between": int(i2 - i1),
    }


# ═══════════════════════════════════════════
# 냉각 기간 — EPS 추정치 반영 시차 대응
# ═══════════════════════════════════════════

def cooling_ok(df: pd.DataFrame) -> tuple[bool, dict]:
    """
    최근 COOLING_DAYS 안에 급락(-10% 이상)이 있었으면 아직 평가하지 않는다.
    애널리스트가 추정치를 고칠 시간을 주고, 투매 진정도 확인한다.
    """
    ret = df["Close"].pct_change().tail(C.COOLING_DAYS)
    crash = ret[ret <= -C.COOLING_DROP_PCT]
    if crash.empty:
        return True, {}
    pos_from_end = len(ret) - ret.index.get_loc(crash.index[-1]) - 1
    return False, {
        "recent_crash_pct": float(crash.iloc[-1]),
        "days_since_crash": int(pos_from_end),
        "wait_more_days": int(C.COOLING_DAYS - pos_from_end),
    }


# ═══════════════════════════════════════════
# 종합 스코어
# ═══════════════════════════════════════════

def _partial(value: float, full: float, zero: float, points: float) -> float:
    """
    부분점수. full일 때 만점, zero일 때 0점, 사이는 비례.
    full < zero 인 경우(낮을수록 좋은 지표)도 자동으로 처리된다.
    """
    if value is None or np.isnan(value) or full == zero:
        return 0.0
    frac = (value - zero) / (full - zero)
    return max(0.0, float(np.clip(frac, 0.0, 1.0)) * points)


def _entry(value, points: float, earned: float, extra=None) -> dict:
    return {"value": value, "score": round(earned, 2), "max": points,
            "hit": earned >= points * C.HIT_FRACTION,
            **({"info": extra} if extra is not None else {})}


def oversold_score(df: pd.DataFrame) -> dict:
    """
    7개 지표를 17점 만점으로 채점한다.
    PARTIAL_CREDIT=True면 비례 배분, False면 예전 O/X 방식.
    """
    close = df["Close"]
    detail, score = {}, 0.0
    P = C.PARTIAL_CREDIT

    # 1. ATR 이격도 (3점) — 종목별 변동성으로 정규화한 낙폭
    d = atr_distance(df)
    pt = (_partial(d, C.ATR_FULL, C.ATR_ZERO, C.ATR_SCORE) if P
          else (C.ATR_SCORE if (not np.isnan(d)) and d >= C.ATR_DISTANCE_MIN else 0))
    score += pt
    detail["atr_distance"] = _entry(None if np.isnan(d) else round(d, 2),
                                    C.ATR_SCORE, pt)

    # 2. RSI (2점) — 최근 창 내 최저치
    r = rsi(close).tail(C.SIGNAL_WINDOW).dropna()
    rv = float(r.min()) if not r.empty else np.nan
    pt = (_partial(rv, C.RSI_FULL, C.RSI_ZERO, C.RSI_SCORE) if P
          else (C.RSI_SCORE if (not np.isnan(rv)) and rv < C.RSI_MAX else 0))
    score += pt
    detail["rsi"] = _entry(None if np.isnan(rv) else round(rv, 1), C.RSI_SCORE, pt)

    # 3. 볼린저 %B (2점) — 최근 창 내 최저치
    b = bollinger_percent_b(close).tail(C.SIGNAL_WINDOW).dropna()
    bv = float(b.min()) if not b.empty else np.nan
    pt = (_partial(bv, C.BB_FULL, C.BB_ZERO, C.BB_SCORE) if P
          else (C.BB_SCORE if (not np.isnan(bv)) and bv < C.BB_PERCENT_B_MAX else 0))
    score += pt
    detail["bollinger_pct_b"] = _entry(None if np.isnan(bv) else round(bv, 3),
                                       C.BB_SCORE, pt)

    # 4. 52주 고점 대비 낙폭 (2점)
    dd = drawdown_from_high(df)
    pt = (_partial(dd, C.DD_FULL, C.DD_ZERO, C.DRAWDOWN_SCORE) if P
          else (C.DRAWDOWN_SCORE if (not np.isnan(dd)) and dd >= C.DRAWDOWN_MIN else 0))
    score += pt
    detail["drawdown_52w"] = _entry(None if np.isnan(dd) else round(dd, 3),
                                    C.DRAWDOWN_SCORE, pt)

    # 5. 피보나치 되돌림 (2점) — 38.2~61.8% 중심에 가까울수록 만점
    ratio, in_zone = fib_retracement(df)
    if P:
        pt = (0.0 if np.isnan(ratio) else
              _partial(abs(ratio - C.FIB_CENTER), 0.0, C.FIB_SPREAD, C.FIB_SCORE))
    else:
        pt = C.FIB_SCORE if in_zone else 0
    score += pt
    detail["fib_retracement"] = _entry(None if np.isnan(ratio) else round(ratio, 3),
                                       C.FIB_SCORE, pt)

    # 6. 거래량 클라이맥스 (3점) — 투매 배수에 비례, 소진 확인 시 만점
    vhit, vinfo = volume_climax(df)
    if P:
        mult = vinfo.get("climax_ratio") or vinfo.get("max_vol_ratio") or 0.0
        pt = _partial(float(mult), C.VOL_FULL, C.VOL_ZERO, C.VOL_SCORE)
        if not vhit:
            pt *= 0.5          # 투매는 있었지만 아직 소진 확인 안 됨 → 절반
    else:
        pt = C.VOL_SCORE if vhit else 0
    score += pt
    detail["volume_climax"] = _entry(vinfo, C.VOL_SCORE, pt)

    # 7. MACD 강세 다이버전스 (3점) — 있냐 없냐라 부분점수 없음
    dhit, dinfo = macd_bullish_divergence(df)
    pt = C.MACD_DIV_SCORE if dhit else 0
    score += pt
    detail["macd_divergence"] = _entry(dinfo, C.MACD_DIV_SCORE, pt)

    if P:
        score = min(score * C.SCORE_SCALE, float(C.MAX_SCORE))
    return {"score": round(score, 1), "max_score": C.MAX_SCORE, "detail": detail}
