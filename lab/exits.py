"""
lab/exits.py — 사는 규칙 × 파는 규칙 전수 시험 (MNQ 대회용, 가장 빡빡한 버전)

사는 쪽
  - 신호 17개(돌파·추세 9, 과매도 7, 거래량 1)를 하나 또는 둘 겹쳐서 153가지
  - 여기에 필터 6개(없음, 200일선, VIX 낮음, VIX 높음, VIX 하락 중, 미국 정규장) → 918가지
  - 매수와 매도(공매도) 양쪽
파는 쪽 (전부 조합: 5 × 4 × 4 × 3 × 7 = 1,680가지)
  - 최대 보유: 6시간 / 1일 / 2일 / 3일 / 5일
  - 손절: 없음 / 0.7% / 1.5% / 3%
  - 익절: 없음 / 1% / 2% / 4%
  - 따라 올라가는 손절: 없음 / 최고점에서 1% / 2%
  - 지표 청산: 없음 / RSI 50 / RSI 70 / 20봉 평균선 회복 / MACD 꺾임 / 10봉 평균선 이탈 / 개장 30분 저점 이탈
안전장치
  1. 앞 2/3 기간 성적으로만 고르고 뒤 1/3로 채점
  2. 통과 기준: 앞·뒤 모두 플러스, 뒤 기간 주 2회 이상, 4주 단위 60% 이상 플러스, 최대 낙폭 -35% 이내,
     수수료 $1과 $7.5 둘 다 플러스, 매매 30번 이상
  3. 이웃 설정(파는 규칙 숫자를 한 칸씩 바꾼 것) 70% 이상도 플러스
  4. 원숭이 우주: 사는 시점만 무작위로 바꾼 똑같은 크기의 시험을 2번 더 돌려, 거기서도 몇 개가 통과하는지 비교
  5. 최종 후보는 매매 횟수·파는 규칙까지 같은 원숭이 2,000마리와 비교. 40개를 동시에 시험하므로
     기준도 5% ÷ 40 = 0.125% (2,000마리 중 2마리 이하만 우리만큼 벌어야 통과)
결과: lab/results/exits_1h.json, exits_1h.md
"""
from __future__ import annotations

import itertools
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
from numba import njit, prange

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mnq  # noqa: E402

PT = 2.0
CAP = 10000.0
COST_LO = 1.5   # 수수료 $1 + 미끄러짐 $0.5
COST_HI = 8.0   # 수수료 $7.5 + 미끄러짐 $0.5
D = 23
HOLDS = np.array([6, 23, 46, 69, 115], np.int64)
STOPS = np.array([0.0, 0.007, 0.015, 0.03])
TGTS = np.array([0.0, 0.01, 0.02, 0.04])
TRAILS = np.array([0.0, 0.01, 0.02])
XNAMES = ["없음", "RSI 50 도달", "RSI 70 도달", "20봉 평균선 회복", "MACD 꺾임", "10봉 평균선 이탈", "개장 30분 저점 이탈"]
NX = len(XNAMES)
N_RANDOM_UNIVERSE = int(os.getenv("LAB_UNIVERSES", "2"))
N_MONKEY = int(os.getenv("LAB_MONKEY", "2000"))
SEED = 11

# 통계 칸 (수수료 낮음 L / 높음 H)
ST = ["fn", "bn", "btpw", "bwtpw_min",
      "f_ret_L", "f_mdd_L", "fw_med_L", "fw_mdd_L", "fw_pos_L",
      "b_ret_L", "b_mdd_L", "bw_med_L", "bw_mdd_L", "bw_pos_L", "bw_worst_L",
      "f_ret_H", "b_ret_H", "b_mdd_H", "bw_pos_H"]
SI = {k: i for i, k in enumerate(ST)}
NST = len(ST)


# ─────────────────────────────────────────────
# 데이터
# ─────────────────────────────────────────────
def load_vix() -> pd.Series:
    path = os.path.join(mnq.DATA, "VIX_1d.csv")
    os.makedirs(mnq.DATA, exist_ok=True)
    if os.path.exists(path) and (datetime.now().timestamp() - os.path.getmtime(path)) < 20 * 3600:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
    else:
        import yfinance as yf
        df = yf.download("^VIX", period="15y", interval="1d", auto_adjust=False, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Close"]].dropna()
        df.to_csv(path)
    s = df["Close"]
    ix = pd.to_datetime(s.index, utc=True)
    s.index = ix.tz_convert(None).normalize()
    return s


def prev_day_map(idx: pd.DatetimeIndex, daily: pd.Series) -> pd.Series:
    """시간봉마다 '전날까지 확정된' 일봉 값 (미래 정보 안 씀)."""
    day = pd.to_datetime(pd.Series(idx.date, index=idx))
    dd = daily.dropna().sort_index()
    left = pd.DataFrame({"date": pd.to_datetime(day.values).astype("datetime64[ns]"), "k": np.arange(len(idx))})
    right = pd.DataFrame({"date": pd.to_datetime(dd.index).astype("datetime64[ns]"), "v": dd.values}).sort_values("date")
    left = left.sort_values("date", kind="mergesort")
    m = pd.merge_asof(left, right, on="date", allow_exact_matches=False).sort_values("k")
    return pd.Series(m["v"].values, index=idx)


# ─────────────────────────────────────────────
# 지표
# ─────────────────────────────────────────────
def psar(h, l, step=0.02, mx=0.2):
    n = len(h)
    up = np.zeros(n, bool)
    sar = np.zeros(n)
    if n < 3:
        return up
    bull, af, ep, sar[0] = True, step, h[0], l[0]
    for i in range(1, n):
        prev = sar[i - 1]
        if bull:
            s = prev + af * (ep - prev)
            s = min(s, l[i - 1], l[i - 2] if i > 1 else l[i - 1])
            if l[i] < s:
                bull, s, ep, af = False, ep, l[i], step
            elif h[i] > ep:
                ep, af = h[i], min(mx, af + step)
        else:
            s = prev + af * (ep - prev)
            s = max(s, h[i - 1], h[i - 2] if i > 1 else h[i - 1])
            if h[i] > s:
                bull, s, ep, af = True, ep, h[i], step
            elif l[i] < ep:
                ep, af = l[i], min(mx, af + step)
        sar[i] = s
        up[i] = bull
    return up


def build(df: pd.DataFrame, daily: pd.DataFrame, vix: pd.Series):
    o, h, l, c, v = (df[k] for k in ("Open", "High", "Low", "Close", "Volume"))
    idx = df.index
    hr = pd.Series(idx.hour, index=idx)
    day = pd.Series(idx.date, index=idx)
    a14 = mnq.atr(df, 14)
    r = mnq.rsi(c, 14)
    hist = mnq.ema(c, 12) - mnq.ema(c, 26)
    hist = hist - mnq.ema(hist, 9)
    sma20 = c.rolling(20).mean()
    sd20 = c.rolling(20).std()
    ema10, ema20, ema50 = mnq.ema(c, 10), mnq.ema(c, 20), mnq.ema(c, 50)
    s460 = c.rolling(20 * D).mean()

    first = hr == 9
    orh = h.where(first).groupby(day.values).transform("max")
    orl = l.where(first).groupby(day.values).transform("min")
    later = (hr >= 10) & (hr <= 15)

    def cross_up(a, b):
        return (a > b) & (a.shift(1) <= b.shift(1))

    S = {}
    S["개장 돌파"] = (later & cross_up(c, orh), later & cross_up(orl, c))
    S["20일 신고가"] = (c > h.rolling(20 * D).max().shift(1), c < l.rolling(20 * D).min().shift(1))
    tenkan = (h.rolling(9).max() + l.rolling(9).min()) / 2
    kijun = (h.rolling(26).max() + l.rolling(26).min()) / 2
    sa = ((tenkan + kijun) / 2).shift(26)
    sb = ((h.rolling(52).max() + l.rolling(52).min()) / 2).shift(26)
    top, bot = np.maximum(sa, sb), np.minimum(sa, sb)
    S["구름대 돌파"] = (cross_up(c, top), cross_up(bot, c))
    same = v.groupby(hr.values).transform(lambda s: s.rolling(20, min_periods=10).mean().shift(1))
    big = v > 2 * same
    S["거래량 급증"] = (big & (c > o), big & (c < o))
    S["MACD 0선 돌파"] = ((hist > 0) & (hist.shift(1) <= 0), (hist < 0) & (hist.shift(1) >= 0))
    up2 = (hist > hist.shift(1)) & (hist.shift(1) > hist.shift(2))
    dn2 = (hist < hist.shift(1)) & (hist.shift(1) < hist.shift(2))
    S["MACD 돌아섬"] = ((hist < 0) & up2, (hist > 0) & dn2)
    S["이평 골든크로스"] = (cross_up(ema20, ema50), cross_up(ema50, ema20))
    # DMI / ADX
    upm = h.diff()
    dnm = -l.diff()
    pdm = np.where((upm > dnm) & (upm > 0), upm, 0.0)
    ndm = np.where((dnm > upm) & (dnm > 0), dnm, 0.0)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atrw = tr.ewm(alpha=1 / 14, adjust=False).mean()
    pdi = 100 * pd.Series(pdm, index=idx).ewm(alpha=1 / 14, adjust=False).mean() / atrw
    ndi = 100 * pd.Series(ndm, index=idx).ewm(alpha=1 / 14, adjust=False).mean() / atrw
    dx = 100 * (pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan)
    adx = dx.ewm(alpha=1 / 14, adjust=False).mean()
    S["DMI 전환(ADX 25↑)"] = ((adx > 25) & cross_up(pdi, ndi), (adx > 25) & cross_up(ndi, pdi))
    sar_up = pd.Series(psar(h.to_numpy(float), l.to_numpy(float)), index=idx)
    S["파라볼릭 SAR 전환"] = (sar_up & ~sar_up.shift(1, fill_value=False), ~sar_up & sar_up.shift(1, fill_value=False))
    S["RSI 30 상향 돌파"] = (cross_up(r, pd.Series(30.0, index=idx)), cross_up(pd.Series(70.0, index=idx), r))
    S["볼린저 하단 복귀"] = (cross_up(c, sma20 - 2 * sd20), cross_up(sma20 + 2 * sd20, c))
    S["20일선 3% 이격"] = (c < s460 * 0.97, c > s460 * 1.03)
    chg = c - c.shift(5)
    S["급락(흔들림 4배)"] = (chg < -4 * a14, chg > 4 * a14)
    ll, hh = l.rolling(14).min(), h.rolling(14).max()
    k = 100 * (c - ll) / (hh - ll).replace(0, np.nan)
    dk = k.rolling(3).mean()
    S["스토캐스틱 20 반등"] = (cross_up(k, dk) & (k.shift(1) < 20), cross_up(dk, k) & (k.shift(1) > 80))
    tp = (h + l + c) / 3
    mad = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    cci = (tp - tp.rolling(20).mean()) / (0.015 * mad.replace(0, np.nan))
    S["CCI -100 반등"] = (cross_up(cci, pd.Series(-100.0, index=idx)), cross_up(pd.Series(100.0, index=idx), cci))
    wr = -100 * (hh - c) / (hh - ll).replace(0, np.nan)
    S["윌리엄스 %R 반등"] = (cross_up(wr, pd.Series(-80.0, index=idx)), cross_up(pd.Series(-20.0, index=idx), wr))
    obv = (np.sign(c.diff()).fillna(0) * v).cumsum()
    S["OBV 20일 신고가"] = (obv > obv.rolling(20 * D).max().shift(1), obv < obv.rolling(20 * D).min().shift(1))

    # 필터
    dclose = daily["Close"].copy()
    dclose.index = (pd.to_datetime(dclose.index) + pd.Timedelta(hours=12)).tz_localize(None).normalize() \
        if dclose.index.tz is None else (dclose.index + pd.Timedelta(hours=12)).tz_convert(None).normalize()
    up200 = (dclose > dclose.rolling(200).mean()).where(dclose.rolling(200).count() >= 200)
    t200 = prev_day_map(idx, up200.astype(float))
    vx = prev_day_map(idx, vix)
    vx5 = prev_day_map(idx, vix.rolling(5).mean())
    known = t200.notna()
    F = {
        "필터 없음": (pd.Series(True, index=idx), pd.Series(True, index=idx)),
        "200일선 방향": (known & (t200 == 1), known & (t200 == 0)),
        "VIX 20 미만": (vx < 20, vx < 20),
        "VIX 20 이상": (vx >= 20, vx >= 20),
        "VIX 하락 중": (vx < vx5, vx < vx5),
        "미국 정규장": (later, later),
    }
    # 파는 신호
    X = np.zeros((NX, 2, len(df)), bool)
    X[1, 0], X[1, 1] = (r >= 50).to_numpy(), (r <= 50).to_numpy()
    X[2, 0], X[2, 1] = (r >= 70).to_numpy(), (r <= 30).to_numpy()
    X[3, 0], X[3, 1] = (c >= sma20).to_numpy(), (c <= sma20).to_numpy()
    X[4, 0], X[4, 1] = ((hist < 0) & (hist.shift(1) >= 0)).to_numpy(), ((hist > 0) & (hist.shift(1) <= 0)).to_numpy()
    X[5, 0], X[5, 1] = (c < ema10).to_numpy(), (c > ema10).to_numpy()
    X[6, 0], X[6, 1] = (later & (c < orl)).to_numpy(), (later & (c > orh)).to_numpy()

    tob = lambda s: s.fillna(False).to_numpy(bool)
    S = {k: (tob(a), tob(b)) for k, (a, b) in S.items()}
    F = {k: (tob(a), tob(b)) for k, (a, b) in F.items()}
    return S, F, X


# ─────────────────────────────────────────────
# 엔진 (numba): 사는 신호 하나에 대해 파는 규칙 1,680가지를 한 번에
# ─────────────────────────────────────────────
@njit(cache=True)
def _seg_stats(p, a, b):
    eq, peak, mdd, tot = CAP, CAP, 0.0, 0.0
    for i in range(a, b):
        tot += p[i]
        eq = CAP + tot
        if eq > peak:
            peak = eq
        dd = eq / peak - 1.0
        if dd < mdd:
            mdd = dd
    return tot / CAP * 100.0, mdd * 100.0


@njit(cache=True)
def _windows(p, ws, we, tre, ntr):
    m = len(ws)
    rets = np.empty(m)
    mdds = np.empty(m)
    tpw = np.empty(m)
    for w in range(m):
        r, d = _seg_stats(p, ws[w], we[w])
        rets[w], mdds[w] = r, d
        k = 0
        for t in range(ntr):
            if tre[t] >= ws[w] and tre[t] < we[w]:
                k += 1
        tpw[w] = k / 4.0
    return rets, mdds, tpw


@njit(cache=True)
def _sim(o, h, l, c, ent, xs, d, hold, stop, tgt, trail, pl, ph, tre):
    n = len(c)
    ntr = 0
    i = 1
    while i < n:
        if not ent[i - 1]:
            i += 1
            continue
        e = i
        px0 = o[e]
        pl[e] -= COST_LO
        ph[e] -= COST_HI
        ref = px0
        peak = px0
        j = e
        done = False
        while j < n and not done:
            exitp = -1.0
            sp = px0 * (1.0 - d * stop) if stop > 0 else -1.0
            tp = px0 * (1.0 + d * tgt) if tgt > 0 else -1.0
            tl = peak * (1.0 - d * trail) if trail > 0 else -1.0
            # 손절·추적손절 먼저 (보수적으로)
            worst = -1.0
            if sp > 0:
                worst = sp
            if tl > 0:
                if worst < 0:
                    worst = tl
                elif d > 0 and tl > worst:
                    worst = tl
                elif d < 0 and tl < worst:
                    worst = tl
            if worst > 0 and ((d > 0 and l[j] <= worst) or (d < 0 and h[j] >= worst)):
                if j > e and ((d > 0 and o[j] < worst) or (d < 0 and o[j] > worst)):
                    exitp = o[j]
                else:
                    exitp = worst
            elif tp > 0 and ((d > 0 and h[j] >= tp) or (d < 0 and l[j] <= tp)):
                if j > e and ((d > 0 and o[j] > tp) or (d < 0 and o[j] < tp)):
                    exitp = o[j]
                else:
                    exitp = tp
            elif j - e + 1 >= hold or j == n - 1:
                exitp = c[j]
            if exitp > 0:
                x = d * (exitp - ref) * PT
                pl[j] += x - COST_LO
                ph[j] += x - COST_HI
                tre[ntr] = e
                ntr += 1
                done = True
                break
            x = d * (c[j] - ref) * PT
            pl[j] += x
            ph[j] += x
            ref = c[j]
            if (d > 0 and h[j] > peak) or (d < 0 and l[j] < peak):
                peak = h[j] if d > 0 else l[j]
            # 지표 청산: 이 봉 마감에 켜지면 다음 봉 시가에 판다
            if xs[j] and j + 1 < n:
                x = d * (o[j + 1] - ref) * PT
                pl[j + 1] += x - COST_LO
                ph[j + 1] += x - COST_HI
                tre[ntr] = e
                ntr += 1
                j = j + 1
                done = True
                break
            j += 1
        i = j + 1
    return ntr


@njit(parallel=True, cache=True)
def run_block(o, h, l, c, ent, xsd, d, ci, weeks_b, fws, fwe, bws, bwe):
    n = len(c)
    nh, ns, nt, nr = len(HOLDS), len(STOPS), len(TGTS), len(TRAILS)
    total = nh * ns * nt * nr * NX
    out = np.full((total, NST), np.nan)
    for q in prange(total):
        xi = q % NX
        tri = (q // NX) % nr
        ti = (q // (NX * nr)) % nt
        si = (q // (NX * nr * nt)) % ns
        hi = q // (NX * nr * nt * ns)
        pl = np.zeros(n)
        ph = np.zeros(n)
        tre = np.zeros(n, np.int64)
        xs = xsd[xi]
        ntr = _sim(o, h, l, c, ent, xs, d, HOLDS[hi], STOPS[si], TGTS[ti], TRAILS[tri], pl, ph, tre)
        fn = 0
        for t in range(ntr):
            if tre[t] < ci:
                fn += 1
        bn = ntr - fn
        fr, fm = _seg_stats(pl, 0, ci)
        br, bm = _seg_stats(pl, ci, n)
        frh, _ = _seg_stats(ph, 0, ci)
        brh, bmh = _seg_stats(ph, ci, n)
        fwr, fwm, _ = _windows(pl, fws, fwe, tre, ntr)
        bwr, bwm, bwt = _windows(pl, bws, bwe, tre, ntr)
        bwrh, _, _ = _windows(ph, bws, bwe, tre, ntr)
        out[q, 0] = fn
        out[q, 1] = bn
        out[q, 2] = bn / weeks_b
        out[q, 3] = bwt.min() if len(bwt) else 0.0
        out[q, 4] = fr
        out[q, 5] = fm
        out[q, 6] = np.median(fwr) if len(fwr) else np.nan
        out[q, 7] = np.median(fwm) if len(fwm) else np.nan
        out[q, 8] = (fwr > 0).mean() if len(fwr) else np.nan
        out[q, 9] = br
        out[q, 10] = bm
        out[q, 11] = np.median(bwr) if len(bwr) else np.nan
        out[q, 12] = np.median(bwm) if len(bwm) else np.nan
        out[q, 13] = (bwr > 0).mean() if len(bwr) else np.nan
        out[q, 14] = bwr.min() if len(bwr) else np.nan
        out[q, 15] = frh
        out[q, 16] = brh
        out[q, 17] = bmh
        out[q, 18] = (bwrh > 0).mean() if len(bwrh) else np.nan
    return out


@njit(cache=True)
def equity_one(o, h, l, c, ent, xs, d, hold, stop, tgt, trail):
    n = len(c)
    pl = np.zeros(n)
    ph = np.zeros(n)
    tre = np.zeros(n, np.int64)
    ntr = _sim(o, h, l, c, ent, xs, d, hold, stop, tgt, trail, pl, ph, tre)
    return pl, ph, tre[:ntr]


# ─────────────────────────────────────────────
# 판정
# ─────────────────────────────────────────────
def front_score(R):
    fs = R[..., SI["fw_med_L"]] / np.maximum(1.0, np.abs(R[..., SI["fw_mdd_L"]])) + R[..., SI["fw_pos_L"]]
    bad = (R[..., SI["fn"]] < 30) | ~np.isfinite(fs)
    return np.where(bad, -np.inf, fs)


def neighbor_share(R):
    """파는 규칙 숫자(보유·손절·익절·추적)를 한 칸씩 바꾼 이웃 중 뒤 기간 플러스 비율. R: (E,2,5,4,4,3,7,NST)"""
    pos = (R[..., SI["b_ret_L"]] > 0).astype(np.float32)
    ok = np.isfinite(R[..., SI["b_ret_L"]]).astype(np.float32)
    cnt = np.zeros(pos.shape, np.float32)
    tot = np.zeros(pos.shape, np.float32)
    for ax in (2, 3, 4, 5):
        for sh in (1, -1):
            p = np.roll(pos, sh, axis=ax)
            k = np.roll(ok, sh, axis=ax)
            valid = np.ones(pos.shape, np.float32)
            sl = [slice(None)] * pos.ndim
            sl[ax] = 0 if sh == 1 else -1
            valid[tuple(sl)] = 0
            cnt += p * valid * k
            tot += valid * k
    return np.where(tot > 0, cnt / np.maximum(tot, 1), 0)


def passes(R, nb):
    g = lambda k: R[..., SI[k]]
    with np.errstate(invalid="ignore"):
        A = (g("fn") >= 30) & (g("f_ret_L") > 0) & (g("fw_pos_L") >= 0.55)
        B = (g("bn") >= 30) & (g("b_ret_L") > 0) & (g("btpw") >= 2) & (g("bw_pos_L") >= 0.6) \
            & (g("b_mdd_L") >= -35) & (g("b_ret_H") > 0)
        C = nb >= 0.7
    return A & B & C, A, B


# ─────────────────────────────────────────────
def main():
    t0 = time.time()
    os.makedirs(mnq.OUT, exist_ok=True)
    df = mnq.load("1h")
    daily = mnq.load("1d")
    vix = load_vix()
    S, F, X = build(df, daily, vix)
    n = len(df)
    ci = int(n * 2 / 3)
    cut = df.index[ci]
    weeks_b = (df.index[-1] - cut).days / 7
    o, h, l, c = (df[k].to_numpy(float) for k in ("Open", "High", "Low", "Close"))

    def wins(a, b):
        s, e = [], []
        t = df.index[a]
        while t + pd.Timedelta(days=28) <= df.index[b - 1]:
            s.append(int(np.searchsorted(df.index, t)))
            e.append(int(np.searchsorted(df.index, t + pd.Timedelta(days=28))))
            t += pd.Timedelta(days=7)
        return np.array(s, np.int64), np.array(e, np.int64)
    fws, fwe = wins(0, ci)
    bws, bwe = wins(ci, n)

    snames = list(S)[: int(os.getenv("LAB_MAXSIG", "99"))]
    fnames = list(F)
    combos = [(a,) for a in snames] + list(itertools.combinations(snames, 2))
    entries = []
    for cb in combos:
        for fn in fnames:
            L = np.ones(n, bool)
            Sh = np.ones(n, bool)
            for nm in cb:
                L &= S[nm][0]
                Sh &= S[nm][1]
            L &= F[fn][0]
            Sh &= F[fn][1]
            entries.append((" + ".join(cb), fn, L, Sh))
    E = len(entries)
    shape = (E, 2, len(HOLDS), len(STOPS), len(TGTS), len(TRAILS), NX, NST)
    print(f"사는 규칙 {E}개 × 2방향 × 파는 규칙 {np.prod(shape[2:7])}개 = {E * 2 * int(np.prod(shape[2:7])):,}가지", flush=True)

    def universe(entry_arrays):
        R = np.full(shape, np.nan, np.float32)
        for ei, (Lr, Sr) in enumerate(entry_arrays):
            for di, (arr, d) in enumerate(((Lr, 1), (Sr, -1))):
                if arr.sum() < 15:
                    continue
                out = run_block(o, h, l, c, arr, X[:, di, :], d, ci, weeks_b, fws, fwe, bws, bwe)
                R[ei, di] = out.reshape(shape[2:])
        return R

    # 1) 진짜
    R = universe([(e[2], e[3]) for e in entries])
    print(f"진짜 완료 {time.time() - t0:.0f}s", flush=True)
    FS = front_score(R)
    NB = neighbor_share(R)
    P, PA, PB = passes(R, NB)

    # 2) 원숭이 우주: 사는 시점만 무작위 (신호 개수는 그대로)
    rng = np.random.default_rng(SEED)
    uni = []
    for u in range(N_RANDOM_UNIVERSE):
        arrs = []
        for e in entries:
            pair = []
            for arr in (e[2], e[3]):
                k = int(arr.sum())
                z = np.zeros(n, bool)
                if k:
                    z[rng.choice(n - 1, size=min(k, n - 1), replace=False)] = True
                pair.append(z)
            arrs.append(tuple(pair))
        Ru = universe(arrs)
        FSu = front_score(Ru)
        NBu = neighbor_share(Ru)
        Pu, _, _ = passes(Ru, NBu)
        flat = np.argsort(FSu.ravel())[::-1][:20]
        uni.append({"pass": int(Pu.sum()),
                    "top20_back_med": float(np.nanmedian(Ru.reshape(-1, NST)[flat, SI["b_ret_L"]])),
                    "top20_back_med_H": float(np.nanmedian(Ru.reshape(-1, NST)[flat, SI["b_ret_H"]]))})
        del Ru, FSu, NBu, Pu
        print(f"원숭이 우주 {u + 1} 완료 {time.time() - t0:.0f}s · 통과 {uni[-1]['pass']}", flush=True)

    flatR = R.reshape(-1, NST)
    order = np.argsort(FS.ravel())[::-1]
    top20 = order[:20]

    def decode(fi):
        ei, di, hi, si, ti, tri, xi = np.unravel_index(fi, shape[:7])
        e = entries[ei]
        return {"사는 신호": e[0], "필터": e[1], "방향": "매수" if di == 0 else "매도",
                "보유": int(HOLDS[hi]), "손절": float(STOPS[si]), "익절": float(TGTS[ti]),
                "추적손절": float(TRAILS[tri]), "지표청산": XNAMES[xi],
                "_idx": [int(ei), int(di), int(hi), int(si), int(ti), int(tri), int(xi)]}

    def row(fi):
        d = decode(fi)
        st = {k: (None if not np.isfinite(flatR[fi, SI[k]]) else round(float(flatR[fi, SI[k]]), 3)) for k in ST}
        d.update(st)
        d["이웃"] = round(float(NB.ravel()[fi]), 2)
        d["앞점수"] = round(float(FS.ravel()[fi]), 3) if np.isfinite(FS.ravel()[fi]) else None
        d["통과"] = bool(P.ravel()[fi])
        return d

    # 3) 통과한 것 중 앞 기간 순위 상위 40개 → 원숭이 1,000마리 정밀 시험
    pidx = [fi for fi in order if P.ravel()[fi]][:40]
    finals = []
    for fi in pidx:
        d = row(fi)
        ei, di, hi, si, ti, tri, xi = d["_idx"]
        dd = 1 if di == 0 else -1
        real = entries[ei][2 if di == 0 else 3]
        pl, ph, tre = equity_one(o, h, l, c, real, X[xi, di], dd, HOLDS[hi], STOPS[si], TGTS[ti], TRAILS[tri])
        real_b = pl[ci:].sum() / CAP * 100
        k = len(tre)
        better = 0
        for _ in range(N_MONKEY):
            z = np.zeros(n, bool)
            z[rng.choice(n - 1, size=k, replace=False)] = True
            mpl, _, _ = equity_one(o, h, l, c, z, X[xi, di], dd, HOLDS[hi], STOPS[si], TGTS[ti], TRAILS[tri])
            better += mpl[ci:].sum() / CAP * 100 >= real_b
        d["원숭이확률"] = round(float(better) / N_MONKEY, 4)
        d["원숭이5퍼"] = bool(d["원숭이확률"] <= 0.05)
        d["최종"] = bool(d["원숭이확률"] <= 0.05 / max(1, len(pidx)))     # 여러 개를 동시에 시험한 만큼 더 엄하게
        finals.append(d)
    print(f"정밀 시험 완료 {time.time() - t0:.0f}s", flush=True)

    # 4) 요약: 사는 신호별 / 파는 규칙별
    by_signal = {}
    for si_, nm in enumerate(snames):
        ei = combos.index((nm,)) * len(fnames)          # 단독 + 필터 없음
        for di in (0, 1):
            sub = FS[ei, di]
            if not np.isfinite(sub).any():
                continue
            j = np.unravel_index(np.argmax(np.where(np.isfinite(sub), sub, -np.inf)), sub.shape)
            fi = np.ravel_multi_index((ei, di) + tuple(j), shape[:7])
            by_signal.setdefault(nm, {})["매수" if di == 0 else "매도"] = row(fi)
    valid = (R[..., SI["fn"]] >= 30)
    def med_by(axis_vals, ax, names):
        res = []
        for k, nm in enumerate(names):
            sl = [slice(None)] * 7
            sl[ax] = k
            vb = R[tuple(sl) + (SI["b_ret_L"],)]
            vf = valid[tuple(sl)]
            pa = P[tuple(sl)]
            vals = vb[vf & np.isfinite(vb)]
            res.append({"값": nm, "뒤수익_중간": round(float(np.median(vals)), 2) if vals.size else None,
                        "뒤수익_상위10": round(float(np.percentile(vals, 90)), 2) if vals.size else None,
                        "통과수": int(pa.sum())})
        return res
    exit_summary = {
        "최대 보유": med_by(None, 2, [f"{x}시간" for x in HOLDS]),
        "손절": med_by(None, 3, ["없음"] + [f"{x*100:g}%" for x in STOPS[1:]]),
        "익절": med_by(None, 4, ["없음"] + [f"{x*100:g}%" for x in TGTS[1:]]),
        "추적손절": med_by(None, 5, ["없음"] + [f"{x*100:g}%" for x in TRAILS[1:]]),
        "지표 청산": med_by(None, 6, XNAMES),
        "방향": med_by(None, 1, ["매수", "매도"]),
    }
    filt_summary = []
    for fk, fn in enumerate(fnames):
        mask = np.array([i % len(fnames) == fk for i in range(E)])
        vb = R[mask][..., SI["b_ret_L"]]
        vf = valid[mask]
        vals = vb[vf & np.isfinite(vb)]
        filt_summary.append({"값": fn, "뒤수익_중간": round(float(np.median(vals)), 2) if vals.size else None,
                             "통과수": int(P[mask].sum())})

    # 5) 그래프용 자산 곡선 (최종 또는 통과 상위 3개 + 그냥 들고 있기)
    curves = {}
    pick = [d for d in finals if d["최종"]][:3] or finals[:3] or [row(fi) for fi in top20[:3]]
    days = pd.Series(np.arange(n), index=df.index).groupby(df.index.date).last()
    for d in pick:
        ei, di, hi, si, ti, tri, xi = d["_idx"]
        dd = 1 if di == 0 else -1
        pl, _, _ = equity_one(o, h, l, c, entries[ei][2 if di == 0 else 3], X[xi, di], dd,
                              HOLDS[hi], STOPS[si], TGTS[ti], TRAILS[tri])
        eq = CAP + np.cumsum(pl)
        curves[f"{d['사는 신호']} · {d['필터']} · {d['방향']}"] = [round(float(eq[i]), 1) for i in days.values]
    bh = np.zeros(n)
    bh[1:] = np.diff(c) * PT
    curves["그냥 들고 있기(1계약)"] = [round(float(CAP + np.cumsum(bh)[i]), 1) for i in days.values]
    curve_dates = [str(x) for x in days.index]
    bh_back = (c[-1] - o[ci]) * PT / CAP * 100
    eqb = CAP + np.cumsum(bh[ci:])
    bh_mdd = float((eqb / np.maximum.accumulate(eqb) - 1).min() * 100)

    summary = {
        "생성": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "데이터": f"{df.index[0]:%Y-%m-%d} ~ {df.index[-1]:%Y-%m-%d}",
        "나눈날": f"{cut:%Y-%m-%d}",
        "뒤기간_주": round(weeks_b, 1),
        "봉수": n,
        "사는규칙수": E * 2,
        "파는규칙수": int(np.prod(shape[2:7])),
        "전체조합": int(E * 2 * np.prod(shape[2:7])),
        "순위가능": int(np.isfinite(FS).sum()),
        "앞통과": int(PA.sum()), "뒤통과": int(PB.sum()),
        "통과": int(P.sum()),
        "원숭이우주": uni,
        "진짜_top20_뒤중간": float(np.nanmedian(flatR[top20, SI["b_ret_L"]])),
        "진짜_top20_뒤중간_H": float(np.nanmedian(flatR[top20, SI["b_ret_H"]])),
        "그냥들고있기_뒤": round(float(bh_back), 1),
        "그냥들고있기_뒤MDD": round(bh_mdd, 1),
        "최종수": sum(1 for d in finals if d["최종"]),
        "원숭이5퍼수": sum(1 for d in finals if d["원숭이5퍼"]),
        "정밀시험수": len(finals),
        "최종기준": round(0.05 / max(1, len(pidx)), 5),
        "걸린시간_초": int(time.time() - t0),
        "사는신호": snames, "필터": fnames, "지표청산": XNAMES,
    }
    res = {"summary": summary, "top20": [row(fi) for fi in top20], "finals": finals,
           "by_signal": by_signal, "exit_summary": exit_summary, "filter_summary": filt_summary,
           "curves": curves, "curve_dates": curve_dates}
    with open(os.path.join(mnq.OUT, "exits_1h.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, default=str)
    lines = [f"# 사고파는 규칙 전수 시험 · {summary['생성']}", "", json.dumps(summary, ensure_ascii=False, indent=1), "",
             "## 최종/정밀 시험 후보", ""]
    for d in finals:
        lines.append(f"- {d['사는 신호']} · {d['필터']} · {d['방향']} · 보유 {d['보유']}h · 손절 {d['손절']} · 익절 {d['익절']} · 추적 {d['추적손절']} · {d['지표청산']}"
                     f" → 뒤 {d['b_ret_L']}% (수수료7.5: {d['b_ret_H']}%) MDD {d['b_mdd_L']} · 주 {d['btpw']}회 · 4주플러스 {d['bw_pos_L']} · 이웃 {d['이웃']} · 원숭이 {d['원숭이확률']} {'★최종' if d['최종'] else ''}")
    with open(os.path.join(mnq.OUT, "exits_1h.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines)[:6000])


if __name__ == "__main__":
    main()
