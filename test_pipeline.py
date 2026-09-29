"""
파이프라인 전체 통합 테스트.
80종목 합성 유니버스(5가지 유형)를 SQLite에 넣고 스크리너를 돌려
각 유형이 의도한 단계에서 걸러지는지 확인한다.
"""
import os
import tempfile

import numpy as np
import pandas as pd

import data as D
import screener as S
import config as C

rng = np.random.default_rng(42)


def series_uptrend(n=700, s=0):
    r = np.random.default_rng(s)
    return 100 * np.exp(np.linspace(0, 1.0, n) + r.normal(0, .01, n).cumsum() * .3)


def series_downtrend(n=700, s=0):
    r = np.random.default_rng(s)
    return 250 * np.exp(np.linspace(0, -1.0, n) + r.normal(0, .01, n).cumsum() * .3)


def series_dip(n=700, s=0, drop=0.30, recent_days=30):
    """상승추세 중 급락 후 안정화 — 우리가 잡고 싶은 바로 그 유형."""
    base = series_uptrend(n - recent_days, s)
    peak = base[-1]
    path = np.concatenate([
        np.linspace(1.0, 1 - drop, 8),
        1 - drop + np.random.default_rng(s).normal(0, .004, recent_days - 8).cumsum() * .3,
    ])
    return np.concatenate([base, peak * path])


def series_crash_today(n=700, s=0):
    """어제 -15% 급락 — 냉각 기간에 걸려야 함."""
    b = series_uptrend(n - 1, s)
    return np.concatenate([b, [b[-1] * 0.85]])


def series_flat(n=700, s=0):
    r = np.random.default_rng(s)
    return 100 + r.normal(0, .6, n).cumsum() * .2


def to_ohlcv(c, climax=False, s=0):
    r = np.random.default_rng(s)
    n = len(c)
    nz = r.uniform(.003, .014, n)
    vol = np.full(n, 1e6) * r.uniform(.8, 1.2, n)
    if climax:
        vol[n - 30:n - 20] = 4.5e6
        vol[n - 20:] = 0.85e6
    idx = pd.bdate_range("2019-01-01", periods=n)
    return pd.DataFrame({"Open": c * (1 + r.uniform(-.004, .004, n)),
                         "High": c * (1 + nz), "Low": c * (1 - nz),
                         "Close": c, "Volume": vol}, index=idx)


# ── 합성 유니버스 구성
tmp = tempfile.mkdtemp()
conn = D.connect(os.path.join(tmp, "test.db"))

groups = {
    "DIP": (25, lambda i: to_ohlcv(series_dip(s=i), climax=True, s=i)),
    "DOWN": (20, lambda i: to_ohlcv(series_downtrend(s=i), s=i)),
    "UP": (20, lambda i: to_ohlcv(series_uptrend(s=i), s=i)),
    "FLAT": (10, lambda i: to_ohlcv(series_flat(s=i), s=i)),
    "CRASH": (5, lambda i: to_ohlcv(series_crash_today(s=i), climax=True, s=i)),
}

tickers, truth = [], {}
for g, (cnt, fn) in groups.items():
    for i in range(cnt):
        t = f"{g}{i:02d}"
        D.store(conn, t, fn(i))
        tickers.append(t)
        truth[t] = g

print(f"합성 유니버스: {len(tickers)}종목 "
      f"({', '.join(f'{g}×{c}' for g,(c,_) in groups.items())})\n")

uni_df = pd.DataFrame({"ticker": tickers, "indices": "TEST",
                       "sector": "", "industry": ""})
cands, stats = S.run(uni_df, conn, verbose=True)
print()
print(S.format_report(cands, stats).split("\n\n")[0])

# ── 유형별 통과율 검증
passed = set(cands["ticker"]) if not cands.empty else set()
print("\n" + "=" * 74)
print("유형별 통과율 (의도대로 걸러지는가)")
print("=" * 74)
ok = True
expect = {"DIP": "높음", "DOWN": "0", "UP": "0", "FLAT": "0", "CRASH": "0"}
for g, (cnt, _) in groups.items():
    hit = sum(1 for t in tickers if truth[t] == g and t in passed)
    verdict = ""
    if g == "DIP":
        good = hit >= cnt * 0.4
        verdict = f"{'PASS' if good else 'FAIL'} (40%+ 기대)"
        ok &= good
    else:
        good = hit == 0
        verdict = f"{'PASS' if good else 'FAIL'} (0 기대)"
        ok &= good
    print(f"  {g:<6} {hit:>3}/{cnt:<3} 통과   {verdict}")

print("=" * 74)
print(f"통합 테스트: {'PASS' if ok else 'FAIL'}")

if not cands.empty:
    print("\n상위 후보 5종목:")
    print(cands[["ticker", "score", "price", "vs_ma200", "atr_dist",
                 "rsi", "dd_52w", "vol_climax", "macd_div"]].head().to_string(index=False))

conn.close()
raise SystemExit(0 if ok else 1)
