"""
lab/orb2.py — 개장 돌파 안정형 최종 후보 비교 (쉬는 날 조건 + 주 2회 안전장치 비용 포함)

파는 규칙(고정): 하루 보유, 손절 1.5%, 익절 2%, 개장 30분 저점 이탈 시 매도
쉬는 날 조건: 없음 / 200일선 아래면 쉼 / VIX 25 이상이면 쉼 / 둘 중 하나라도 해당하면 쉼
안전장치: 한 주(월~금)에 매매가 2번 미만이면, 모자란 만큼 '사자마자 파는 왕복'을 한 것으로 보고 비용만 뺀다
         (왕복 1번 = 편도 비용 × 2, 가격 변화는 0으로 가정)
결과: lab/results/orb2_1h.json
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

SELLOFFS = [("2024-07-10", "2024-08-07", "2024년 7~8월 급락"), ("2025-02-19", "2025-04-08", "2025년 관세 하락")]


def main():
    df = mnq.load("1h")
    daily = mnq.load("1d")
    vix = ex.load_vix()
    S, F, X = ex.build(df, daily, vix)
    idx = df.index
    n = len(df)
    ci = int(n * 2 / 3)
    o, h, l, c = (df[k].to_numpy(float) for k in ("Open", "High", "Low", "Close"))
    vx = ex.prev_day_map(idx, vix).to_numpy()
    t200 = F["200일선 방향"][0]
    vix_ok = np.nan_to_num(vx, nan=99) < 25
    filt = {
        "쉬는 날 없음": np.ones(n, bool),
        "200일선 아래면 쉼": t200,
        "VIX 25 이상이면 쉼": vix_ok,
        "200일선 아래 또는 VIX 25 이상이면 쉼": t200 & vix_ok,
    }
    base = S["개장 돌파"][0]
    xi = ex.XNAMES.index("개장 30분 저점 이탈")
    week = pd.Series(idx.to_period("W-FRI").astype(str), index=idx)
    weeks_all = week.unique()
    out = {"cut": str(idx[ci].date()), "results": []}
    for fname, fa in filt.items():
        ent = base & fa
        res = {"조건": fname}
        for cost_name, cost in (("수수료1", ex.COST_LO), ("수수료7.5", ex.COST_HI)):
            pl, ph, tre = ex.equity_one(o, h, l, c, ent, X[xi, 0], 1, 23, 0.015, 0.02, 0.0)
            p = (pl if cost == ex.COST_LO else ph).copy()
            # 안전장치 비용
            cnt = pd.Series(1, index=week.values[tre]).groupby(level=0).sum() if len(tre) else pd.Series(dtype=int)
            need_weeks = 0
            extra = 0
            for w in weeks_all:
                k = int(cnt.get(w, 0))
                if k < 2:
                    pos = np.where(week.values == w)[0]
                    j = pos[-1]
                    p[j] -= (2 - k) * cost * 2
                    if j >= ci:
                        need_weeks += 1
                        extra += 2 - k
            b = p[ci:]
            eq = ex.CAP + np.cumsum(b)
            wk_back = len([w for w in weeks_all if np.where(week.values == w)[0][0] >= ci])
            # 대회처럼 4주씩 자른 성적
            rets = []
            bi = np.arange(ci, n)
            t = idx[ci]
            while t + pd.Timedelta(days=28) <= idx[-1]:
                a_ = int(np.searchsorted(idx, t))
                e_ = int(np.searchsorted(idx, t + pd.Timedelta(days=28)))
                rets.append(p[a_:e_].sum() / ex.CAP * 100)
                t += pd.Timedelta(days=7)
            rets = np.array(rets)
            sell = {}
            for a, bb, lab in SELLOFFS:
                ia = int(np.searchsorted(idx, pd.Timestamp(a, tz=idx.tz)))
                ib = int(np.searchsorted(idx, pd.Timestamp(bb, tz=idx.tz)))
                sell[lab] = round(float(p[ia:ib].sum()), 0)
            res[cost_name] = {"뒤수익": round(float(b.sum() / ex.CAP * 100), 1),
                              "뒤MDD": round(float((eq / np.maximum.accumulate(eq) - 1).min() * 100), 1),
                              "진짜매매_주당": round(float(sum(1 for x in tre if x >= ci) / max(1, wk_back)), 2),
                              "안전장치_필요한주": need_weeks, "뒤기간_주수": wk_back, "안전장치_왕복수": extra,
                              "4주_플러스비율": round(float((rets > 0).mean() * 100), 0) if len(rets) else None,
                              "4주_중간": round(float(np.median(rets)), 1) if len(rets) else None,
                              "4주_최악": round(float(rets.min()), 1) if len(rets) else None,
                              "하락구간": sell}
        out["results"].append(res)
        print(json.dumps(res, ensure_ascii=False))
    with open(os.path.join(mnq.OUT, "orb2_1h.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
