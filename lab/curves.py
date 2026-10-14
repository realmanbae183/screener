"""
lab/curves.py — 후보 규칙 몇 개의 계좌 흐름(일별)과 원숭이 범위를 뽑아 그래프용 JSON으로 저장
결과: lab/results/curves_1h.json
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mnq  # noqa: E402
import exits as ex  # noqa: E402

PICKS = [
    {"이름": "스토캐스틱 반등 → RSI 70에 매도", "신호": ["스토캐스틱 20 반등"], "필터": "필터 없음", "방향": 1,
     "보유": 23, "손절": 0.0, "익절": 0.0, "추적": 0.0, "청산": "RSI 70 도달", "원숭이": True},
    {"이름": "윌리엄스 %R 반등 → 20봉 평균선에 매도", "신호": ["윌리엄스 %R 반등"], "필터": "200일선 방향", "방향": 1,
     "보유": 115, "손절": 0.015, "익절": 0.04, "추적": 0.0, "청산": "20봉 평균선 회복", "원숭이": False},
    {"이름": "개장 돌파 → 하루 뒤 매도", "신호": ["개장 돌파"], "필터": "필터 없음", "방향": 1,
     "보유": 23, "손절": 0.0, "익절": 0.0, "추적": 0.0, "청산": "없음", "원숭이": False},
]
N_MONKEY = 400


def main():
    df = mnq.load("1h")
    daily = mnq.load("1d")
    vix = ex.load_vix()
    S, F, X = ex.build(df, daily, vix)
    n = len(df)
    ci = int(n * 2 / 3)
    o, h, l, c = (df[k].to_numpy(float) for k in ("Open", "High", "Low", "Close"))
    last = pd.Series(np.arange(n), index=df.index).groupby(df.index.date).last()
    di_idx = last.values
    rng = np.random.default_rng(5)
    out = {"dates": [str(x) for x in last.index], "cut": str(df.index[ci].date()), "series": [], "bands": {}}
    for p in PICKS:
        side = 0 if p["방향"] > 0 else 1
        ent = np.ones(n, bool)
        for nm in p["신호"]:
            ent &= S[nm][side]
        ent &= F[p["필터"]][side]
        xi = ex.XNAMES.index(p["청산"])
        pl, ph, tre = ex.equity_one(o, h, l, c, ent, X[xi, side], p["방향"], p["보유"], p["손절"], p["익절"], p["추적"])
        eq = ex.CAP + np.cumsum(pl)
        eqh = ex.CAP + np.cumsum(ph)
        seg = eq[ci:] - eq[ci - 1]
        b_eq = ex.CAP + seg
        out["series"].append({"이름": p["이름"], "eq": [round(float(eq[i]), 1) for i in di_idx],
                              "eq_hi": [round(float(eqh[i]), 1) for i in di_idx],
                              "매매수": int(len(tre)),
                              "뒤수익": round(float(seg[-1] / ex.CAP * 100), 1),
                              "뒤MDD": round(float((b_eq / np.maximum.accumulate(b_eq) - 1).min() * 100), 1)})
        if p["원숭이"]:
            k = len(tre)
            M = np.empty((N_MONKEY, len(di_idx)))
            for m in range(N_MONKEY):
                z = np.zeros(n, bool)
                z[rng.choice(n - 1, size=k, replace=False)] = True
                mpl, _, _ = ex.equity_one(o, h, l, c, z, X[xi, side], p["방향"], p["보유"], p["손절"], p["익절"], p["추적"])
                M[m] = (ex.CAP + np.cumsum(mpl))[di_idx]
            out["bands"][p["이름"]] = {q: [round(float(v), 1) for v in np.percentile(M, q, axis=0)] for q in (5, 50, 95)}
    bh = np.zeros(n)
    bh[1:] = np.diff(c) * ex.PT
    eq = ex.CAP + np.cumsum(bh)
    seg = eq[ci:] - eq[ci - 1]
    b_eq = ex.CAP + seg
    out["series"].append({"이름": "그냥 들고 있기", "eq": [round(float(eq[i]), 1) for i in di_idx], "매매수": 1,
                          "뒤수익": round(float(seg[-1] / ex.CAP * 100), 1),
                          "뒤MDD": round(float((b_eq / np.maximum.accumulate(b_eq) - 1).min() * 100), 1)})
    with open(os.path.join(mnq.OUT, "curves_1h.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    for s in out["series"]:
        print(s["이름"], s["매매수"], s["뒤수익"], s["뒤MDD"], s["eq"][-1])


if __name__ == "__main__":
    main()
