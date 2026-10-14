"""
lab/mnq.py — 마이크로 나스닥(MNQ) 대회용 전략 연습장

대회 조건 (제2회 전국대학생 AI시스템트레이딩 경진대회)
  - 상품: CME Micro E-mini Nasdaq-100 (MNQ), 지수 1포인트 = $2, 최소 호가 0.25포인트 = $0.50
  - 수수료: 1계약 편도 $7.5  → 왕복 $15. 여기에 미끄러짐(슬리피지) 한 호가씩 더해 편도 $8로 잡는다
  - 기간: 4주(10/19 ~ 11/13), 순위 = 수익률 · MDD(최대 낙폭) 종합, 주 2회 이상 매매 필수

하는 일
  1. 나스닥100 선물(NQ=F) 시세를 받는다 (1시간봉 약 2년, 일봉 약 15년)
  2. 전략 후보마다 여러 설정을 돌려본다 (1계약 기준)
  3. 앞쪽 2/3 기간에서 고른 설정이 뒤쪽 1/3에서도 통하는지 본다 (과최적화 막기)
  4. 대회처럼 '아무 4주'를 잘라서 수익·낙폭·주간 매매 횟수를 본다
  결과는 lab/results/ 에 표로 남는다.
"""
from __future__ import annotations

import itertools
import json
import math
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "results")
PT = 2.0            # MNQ 1포인트 = $2
SIDE_COST = float(os.getenv("LAB_COST", "8.0"))   # 편도 비용 = 수수료 + 미끄러짐 $0.5 (수수료 $7.5면 8, $1이면 1.5)
CAPITAL = float(os.getenv("LAB_CAPITAL", "10000"))   # 모의계좌 금액 (모르면 $1만으로 가정, 비율은 계약 수로 조절)


# ─────────────────────────────────────────────
# 시세
# ─────────────────────────────────────────────
def load(interval: str) -> pd.DataFrame:
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, f"NQ_{interval}.csv")
    if os.path.exists(path) and (datetime.now().timestamp() - os.path.getmtime(path)) < 20 * 3600:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
    else:
        import yfinance as yf
        period = {"1h": "730d", "15m": "60d", "1d": "15y"}[interval]
        df = yf.download("NQ=F", period=period, interval=interval, auto_adjust=False, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        df.to_csv(path)
    df.index = pd.to_datetime(df.index, utc=True).tz_convert("America/New_York")
    return df[~df.index.duplicated()].sort_index()


# ─────────────────────────────────────────────
# 지표
# ─────────────────────────────────────────────
def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def rsi(s, n):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def atr(df, n):
    pc = df["Close"].shift()
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - pc).abs(), (df["Low"] - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


# ─────────────────────────────────────────────
# 엔진: 봉 마감에 신호 → 다음 봉 시가에 체결. 손절은 봉 안에서 닿으면 그 가격(갭이면 시가)으로.
# ─────────────────────────────────────────────
def simulate(df: pd.DataFrame, want: pd.Series, stop_pts: pd.Series | None = None) -> dict:
    """want: 각 봉 마감 시점에 원하는 포지션(+1 매수 / -1 매도 / 0 없음). 1계약 기준 손익($)을 봉마다 돌려준다."""
    o, h, l, c = (df[k].to_numpy(float) for k in ("Open", "High", "Low", "Close"))
    w = want.reindex(df.index).fillna(0).to_numpy(int)
    sp = None if stop_pts is None else stop_pts.reindex(df.index).to_numpy(float)
    n = len(df)
    pnl = np.zeros(n)
    pos, stop, rest = 0, None, None
    trades, t_in, tp = [], None, 0.0          # (진입 시각, 청산 시각, 방향, 손익$)
    for i in range(1, n):
        target = w[i - 1]
        if rest is not None:                  # 손절 뒤에는 신호가 바뀔 때까지 쉰다
            if target == rest:
                target = 0
            else:
                rest = None
        x = pos * (o[i] - c[i - 1]) * PT      # 봉 사이(밤사이) 움직임
        pnl[i] += x; tp += x
        if target != pos:
            if pos != 0:
                pnl[i] -= SIDE_COST; tp -= SIDE_COST
                trades.append((t_in, df.index[i], pos, tp))
            if target != 0:
                pnl[i] -= SIDE_COST
                t_in, tp = df.index[i], -SIDE_COST
                stop = None if sp is None or not np.isfinite(sp[i - 1]) else o[i] - target * sp[i - 1]
            pos = target
        if pos != 0 and stop is not None and ((pos > 0 and l[i] <= stop) or (pos < 0 and h[i] >= stop)):
            px = min(o[i], stop) if pos > 0 else max(o[i], stop)
            x = pos * (px - o[i]) * PT - SIDE_COST
            pnl[i] += x; tp += x
            trades.append((t_in, df.index[i], pos, tp))
            rest, pos, stop = w[i - 1], 0, None
            continue
        if pos != 0:
            x = pos * (c[i] - o[i]) * PT
            pnl[i] += x; tp += x
    return {"pnl": pd.Series(pnl, index=df.index), "trades": trades}


# ─────────────────────────────────────────────
# 전략 후보 — 모두 '봉 마감 기준 원하는 포지션'을 돌려준다
# ─────────────────────────────────────────────
def s_donchian(df, n, side, atr_k):
    hi, lo = df["High"].rolling(n).max().shift(), df["Low"].rolling(n).min().shift()
    mid = (hi + lo) / 2
    sig = pd.Series(np.nan, index=df.index)
    sig[df["Close"] > hi] = 1
    sig[df["Close"] < lo] = -1 if side == "both" else 0
    if side == "long":
        sig[df["Close"] < mid] = 0
    want = sig.ffill().fillna(0)
    return want, atr(df, 14) * atr_k


def s_ema(df, f, s_, trend, atr_k):
    a, b = ema(df["Close"], f), ema(df["Close"], s_)
    want = pd.Series(np.where(a > b, 1, -1), index=df.index)
    if trend:
        t = ema(df["Close"], trend)
        want = want.where(~((want == 1) & (df["Close"] < t)) & ~((want == -1) & (df["Close"] > t)), 0)
    return want, atr(df, 14) * atr_k


def s_rsi_dip(df, n, lo, hi, trend, atr_k):
    r = rsi(df["Close"], n)
    t = ema(df["Close"], trend)
    sig = pd.Series(np.nan, index=df.index)
    sig[(r < lo) & (df["Close"] > t)] = 1
    sig[r > hi] = 0
    sig[df["Close"] < t * 0.985] = 0
    return sig.ffill().fillna(0), atr(df, 14) * atr_k


def s_bb_revert(df, n, k, trend, atr_k):
    m, sd = df["Close"].rolling(n).mean(), df["Close"].rolling(n).std()
    t = ema(df["Close"], trend)
    sig = pd.Series(np.nan, index=df.index)
    sig[(df["Close"] < m - k * sd) & (df["Close"] > t)] = 1
    sig[(df["Close"] > m + k * sd) & (df["Close"] < t)] = -1
    sig[((df["Close"] > m) & (sig.ffill() == 1)) | ((df["Close"] < m) & (sig.ffill() == -1))] = 0
    return sig.ffill().fillna(0), atr(df, 14) * atr_k


def s_session(df, enter_h, exit_h, trend):
    """시간대 습관: 미국 동부 enter_h시에 사서 exit_h시에 판다 (예: 장 마감 뒤 사서 다음 날 개장 전 팔기)."""
    hr = df.index.hour
    if enter_h < exit_h:
        inwin = (hr >= enter_h) & (hr < exit_h)
    else:
        inwin = (hr >= enter_h) | (hr < exit_h)
    want = pd.Series(np.where(inwin, 1, 0), index=df.index)
    if trend:
        want = want.where(df["Close"] > ema(df["Close"], trend), 0)
    want = want.where(df.index.dayofweek < 5, 0)
    return want, None


GRIDS = {
    "추세 돌파 (돈치안)": (s_donchian, {"n": [20, 40, 60, 100], "side": ["long", "both"], "atr_k": [2, 3, 5]}),
    "이평 교차": (s_ema, {"f": [10, 20], "s_": [50, 100, 200], "trend": [0, 400], "atr_k": [3, 5]}),
    "RSI 눌림목 (상승장만)": (s_rsi_dip, {"n": [2, 3, 5], "lo": [10, 20, 30], "hi": [60, 70, 80], "trend": [200, 400], "atr_k": [2, 3, 4]}),
    "볼린저 되돌림": (s_bb_revert, {"n": [20, 40], "k": [1.5, 2.0, 2.5], "trend": [200, 400], "atr_k": [2, 3]}),
    "시간대 습관": (s_session, {"enter_h": [16, 18, 20, 9], "exit_h": [9, 10, 16], "trend": [0, 200]}),
}


# ─────────────────────────────────────────────
# 채점
# ─────────────────────────────────────────────
def stats(pnl: pd.Series, trades: list, capital: float = CAPITAL) -> dict:
    eq = capital + pnl.cumsum()
    dd = (eq / eq.cummax() - 1).min() * 100
    weeks = max(1e-9, (pnl.index[-1] - pnl.index[0]).days / 7)
    ret = pnl.sum() / capital * 100
    win = sum(1 for t in trades if t[3] > 0) / len(trades) * 100 if trades else 0
    return {"ret": round(ret, 1), "mdd": round(dd, 1), "calmar": round(ret / abs(dd), 2) if dd < 0 else None,
            "tpw": round(len(trades) / weeks, 1), "n": len(trades), "win": round(win)}


def windows(pnl: pd.Series, trades: list, days: int = 28, step: int = 7, capital: float = CAPITAL) -> dict:
    """대회처럼 4주씩 잘라본 성적 (1주씩 밀면서)."""
    out = []
    t0, t1 = pnl.index[0], pnl.index[-1]
    s = t0
    tt = pd.Series(1, index=pd.DatetimeIndex([t[0] for t in trades])) if trades else pd.Series(dtype=float)
    while s + pd.Timedelta(days=days) <= t1:
        e = s + pd.Timedelta(days=days)
        p = pnl[(pnl.index >= s) & (pnl.index < e)]
        eq = capital + p.cumsum()
        r = p.sum() / capital * 100
        dd = (eq / eq.cummax() - 1).min() * 100
        k = int(((tt.index >= s) & (tt.index < e)).sum()) if len(tt) else 0
        out.append((r, dd, k / 4))
        s += pd.Timedelta(days=step)
    if not out:
        return {}
    a = np.array(out)
    return {"w_n": len(a), "w_ret": round(float(np.median(a[:, 0])), 1), "w_pos": round(float((a[:, 0] > 0).mean() * 100)),
            "w_worst": round(float(a[:, 0].min()), 1), "w_mdd": round(float(np.median(a[:, 1])), 1),
            "w_tpw_min": round(float(a[:, 2].min()), 1)}


def run_grid(df: pd.DataFrame, split: float = 2 / 3) -> list[dict]:
    cut = df.index[int(len(df) * split)]
    rows = []
    for name, (fn, grid) in GRIDS.items():
        keys = list(grid)
        for vals in itertools.product(*(grid[k] for k in keys)):
            p = dict(zip(keys, vals))
            if fn is s_ema and p["f"] >= p["s_"]:
                continue
            if fn is s_session and p["enter_h"] == p["exit_h"]:
                continue
            want, stop = fn(df, **{k: v for k, v in p.items()})
            r = simulate(df, want, stop)
            pnl, tr = r["pnl"], r["trades"]
            tr_a = [t for t in tr if t[0] is not None and t[0] < cut]
            tr_b = [t for t in tr if t[0] is not None and t[0] >= cut]
            A, B = stats(pnl[pnl.index < cut], tr_a), stats(pnl[pnl.index >= cut], tr_b)
            W = windows(pnl[pnl.index >= cut], tr_b)
            rows.append({"전략": name, "설정": p, "앞": A, "뒤": B, **W})
    return rows


def score(r: dict) -> float:
    """대회 점수 흉내: 뒤쪽 기간의 4주 성적 중심. 수익은 크게, 낙폭은 작게, 주 2회 미만이면 탈락."""
    if r.get("w_tpw_min", 0) < 2 or not r.get("w_n"):
        return -1e9
    if (r["앞"]["calmar"] or -1) <= 0:          # 앞 기간에서도 벌었어야 함 (운 제거)
        return -1e9
    return r["w_ret"] / max(1.0, abs(r["w_mdd"])) + r["w_pos"] / 100


def report(rows: list[dict], df: pd.DataFrame, tag: str) -> str:
    rows = sorted(rows, key=score, reverse=True)
    cut = df.index[int(len(df) * 2 / 3)]
    lines = [f"# MNQ 연습 결과 · {tag}", "",
             f"- 데이터: NQ=F {df.index[0]:%Y-%m-%d} ~ {df.index[-1]:%Y-%m-%d} ({len(df):,}봉) · 앞/뒤 나눈 날 {cut:%Y-%m-%d}",
             f"- 1계약 · 계좌 ${CAPITAL:,.0f} 가정 · 편도 비용 ${SIDE_COST} (수수료 + 미끄러짐 0.5)",
             "- 4주 성적 = 뒤 기간을 4주씩 잘라 1주씩 밀며 본 값 (중앙값). 주 2회 미만 매매가 한 번이라도 있으면 탈락 처리", "",
             "| 순위 | 전략 | 설정 | 앞 수익%/MDD% | 뒤 수익%/MDD% | 4주 수익% (중앙) | 4주 플러스 비율 | 4주 최악% | 4주 MDD% (중앙) | 주당 매매 |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(rows[:40], 1):
        if score(r) < -1e8:
            break
        lines.append(f"| {i} | {r['전략']} | {json.dumps(r['설정'], ensure_ascii=False)} | {r['앞']['ret']} / {r['앞']['mdd']} | {r['뒤']['ret']} / {r['뒤']['mdd']} | "
                     f"{r['w_ret']} | {r['w_pos']}% | {r['w_worst']} | {r['w_mdd']} | {r['뒤']['tpw']} |")
    lines += ["", "## 전략별 최고", ""]
    seen = set()
    for r in rows:
        if r["전략"] in seen:
            continue
        seen.add(r["전략"])
        lines.append(f"- {r['전략']}: {json.dumps(r['설정'], ensure_ascii=False)} · 뒤 {r['뒤']['ret']}% / MDD {r['뒤']['mdd']}% · 4주 중앙 {r.get('w_ret')}% · 플러스 {r.get('w_pos')}% · 주 {r['뒤']['tpw']}회"
                     + ("" if score(r) > -1e8 else " (탈락: 주 2회 미만 또는 앞 기간 손실)"))
    return "\n".join(lines)


def main():
    os.makedirs(OUT, exist_ok=True)
    tf = sys.argv[1] if len(sys.argv) > 1 else "1h"
    df = load(tf)
    rows = run_grid(df)
    if SIDE_COST != 8.0:
        tf = f"{tf}_cost{SIDE_COST:g}"
    md = report(rows, df, tf)
    with open(os.path.join(OUT, f"grid_{tf}.md"), "w", encoding="utf-8") as f:
        f.write(md)
    with open(os.path.join(OUT, f"grid_{tf}.json"), "w", encoding="utf-8") as f:
        json.dump(sorted(rows, key=score, reverse=True)[:80], f, ensure_ascii=False, default=str)
    print(md[:6000])


if __name__ == "__main__":
    main()
