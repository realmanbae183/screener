"""
lab/orb.py — 개장 돌파 매수 + '쉬는 날' 조건 시험 (안정형 전략용)

살 때: 미국장 개장 후 첫 봉(동부 9시 봉, 9:30 개장 직후 30분 포함)의 최고가를 10~15시 사이에 처음 넘으면 매수
쉬는 날 조건 8가지:
  없음 / VIX 20 미만 / VIX 25 미만 / VIX 급등 아님(전날 VIX < 5일 평균 × 1.1) /
  전날 나스닥 -1.5% 넘게 안 빠짐 / 개장 30분 출렁임이 평소의 1.5배 미만 / 200일선 위 /
  종합(VIX 25 미만 + 전날 급락 아님 + 개장 출렁임 정상)
팔 때: exits.py와 같은 1,680가지 (보유 × 손절 × 익절 × 추적손절 × 지표청산)
결과: lab/results/orb_1h.json
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
N_MONKEY = 1000


def main():
    df = mnq.load("1h")
    daily = mnq.load("1d")
    vix = ex.load_vix()
    S, F, X = ex.build(df, daily, vix)
    idx = df.index
    n = len(df)
    ci = int(n * 2 / 3)
    o, h, l, c = (df[k].to_numpy(float) for k in ("Open", "High", "Low", "Close"))
    hr = pd.Series(idx.hour, index=idx)
    day = pd.Series(idx.date, index=idx)

    # 쉬는 날 재료
    vx = ex.prev_day_map(idx, vix)
    vx5 = ex.prev_day_map(idx, vix.rolling(5).mean())
    dclose = daily["Close"].copy()
    dclose.index = (dclose.index + pd.Timedelta(hours=12)).tz_convert(None).normalize()
    dchg = ex.prev_day_map(idx, dclose.pct_change() * 100)          # 전날 하루 등락률
    first = hr == 9
    orr = (df["High"] - df["Low"]).where(first)
    day_rng = orr.groupby(day.values).transform("max")              # 오늘 개장 첫 봉 출렁임
    per_day = orr.dropna()
    per_day.index = pd.to_datetime(per_day.index.date)
    avg20 = per_day.groupby(level=0).max().rolling(20, min_periods=10).mean().shift(1)
    avg_map = pd.Series(pd.to_datetime(day.values), index=idx).map(avg20)
    calm_open = (day_rng < 1.5 * avg_map)
    t200 = F["200일선 방향"][0]

    tob = lambda s: pd.Series(s, index=idx).fillna(False).to_numpy(bool)
    filt = {
        "쉬는 날 없음": np.ones(n, bool),
        "VIX 20 미만": tob(vx < 20),
        "VIX 25 미만": tob(vx < 25),
        "VIX 급등 아님": tob(vx < vx5 * 1.1),
        "전날 급락 아님": tob(dchg > -1.5),
        "개장 출렁임 정상": tob(calm_open),
        "200일선 위": t200,
        "종합(VIX25·급락·출렁임)": tob((vx < 25) & (dchg > -1.5) & calm_open),
    }
    base = S["개장 돌파"][0]

    def wins(a, b):
        s, e = [], []
        t = idx[a]
        while t + pd.Timedelta(days=28) <= idx[b - 1]:
            s.append(int(np.searchsorted(idx, t)))
            e.append(int(np.searchsorted(idx, t + pd.Timedelta(days=28))))
            t += pd.Timedelta(days=7)
        return np.array(s, np.int64), np.array(e, np.int64)
    fws, fwe = wins(0, ci)
    bws, bwe = wins(ci, n)
    weeks_b = (idx[-1] - idx[ci]).days / 7
    shape = (len(ex.HOLDS), len(ex.STOPS), len(ex.TGTS), len(ex.TRAILS), ex.NX)
    rng = np.random.default_rng(3)

    def seg_loss(pl, a, b):
        ia, ib = int(np.searchsorted(idx, pd.Timestamp(a, tz=idx.tz))), int(np.searchsorted(idx, pd.Timestamp(b, tz=idx.tz)))
        eq = np.cumsum(pl[ia:ib])
        return round(float(eq[-1]), 0) if len(eq) else 0.0

    out = {"filters": [], "cut": str(idx[ci].date()), "weeks_back": round(weeks_b, 1)}
    for fname, fa in filt.items():
        ent = base & fa
        R = ex.run_block(o, h, l, c, ent, X[:, 0, :], 1, ci, weeks_b, fws, fwe, bws, bwe).reshape(shape + (ex.NST,))
        FS = ex.front_score(R[None, None])[0, 0]
        NB = ex.neighbor_share(R[None, None])[0, 0]
        P, _, _ = ex.passes(R[None, None], NB[None, None])
        P = P[0, 0]
        g = lambda k: R[..., ex.SI[k]]
        valid = g("fn") >= 30
        med_b = float(np.nanmedian(np.where(valid, g("b_ret_L"), np.nan))) if valid.any() else None
        # 앞 기간 1등 설정
        j = np.unravel_index(np.argmax(np.where(np.isfinite(FS), FS, -np.inf)), FS.shape)
        hi, si, ti, tri, xi = j
        pl, ph, tre = ex.equity_one(o, h, l, c, ent, X[xi, 0], 1, ex.HOLDS[hi], ex.STOPS[si], ex.TGTS[ti], ex.TRAILS[tri])
        real_b = pl[ci:].sum() / ex.CAP * 100
        k = len(tre)
        better = 0
        for _ in range(N_MONKEY):
            z = np.zeros(n, bool)
            z[rng.choice(n - 1, size=k, replace=False)] = True
            mpl, _, _ = ex.equity_one(o, h, l, c, z, X[xi, 0], 1, ex.HOLDS[hi], ex.STOPS[si], ex.TGTS[ti], ex.TRAILS[tri])
            better += mpl[ci:].sum() / ex.CAP * 100 >= real_b
        st = {kk: (None if not np.isfinite(R[j + (ex.SI[kk],)]) else round(float(R[j + (ex.SI[kk],)]), 2)) for kk in ex.ST}
        # 같은 파는 규칙을 '쉬는 날 없음'과 비교하려고 설정을 저장
        rec = {"필터": fname, "신호날수": int(ent.sum()), "통과수": int(P.sum()), "뒤수익_중간": None if med_b is None else round(med_b, 2),
               "최고설정": {"보유": int(ex.HOLDS[hi]), "손절": float(ex.STOPS[si]), "익절": float(ex.TGTS[ti]),
                         "추적": float(ex.TRAILS[tri]), "청산": ex.XNAMES[xi]},
               **st, "이웃": round(float(NB[j]), 2), "원숭이확률": round(better / N_MONKEY, 3),
               "하락구간": {lab: seg_loss(pl, a, b) for a, b, lab in SELLOFFS}}
        # 공통 기준 설정(하루 보유, 손절 1.5%, 익절 2%, 개장 저점 이탈)으로도 하락 구간 비교
        cx = ex.XNAMES.index("개장 30분 저점 이탈")
        cpl, _, ctre = ex.equity_one(o, h, l, c, ent, X[cx, 0], 1, 23, 0.015, 0.02, 0.0)
        cb = cpl[ci:]
        ceq = ex.CAP + np.cumsum(cb)
        rec["공통설정"] = {"뒤수익": round(float(cb.sum() / ex.CAP * 100), 1),
                        "뒤MDD": round(float((ceq / np.maximum.accumulate(ceq) - 1).min() * 100), 1),
                        "주당매매": round(float(sum(1 for t in ctre if t >= ci) / weeks_b), 2),
                        "하락구간": {lab: seg_loss(cpl, a, b) for a, b, lab in SELLOFFS}}
        out["filters"].append(rec)
        print(fname, rec["신호날수"], rec["통과수"], rec["뒤수익_중간"], rec["최고설정"], "뒤", st["b_ret_L"], "H", st["b_ret_H"],
              "MDD", st["b_mdd_L"], "주", st["btpw"], "원숭이", rec["원숭이확률"], rec["하락구간"], "| 공통", rec["공통설정"], flush=True)
    bh = np.zeros(n)
    bh[1:] = np.diff(c) * ex.PT
    out["그냥들고있기_하락구간"] = {lab: seg_loss(bh, a, b) for a, b, lab in SELLOFFS}
    print("그냥 들고 있기", out["그냥들고있기_하락구간"])
    with open(os.path.join(mnq.OUT, "orb_1h.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, default=str)


if __name__ == "__main__":
    main()
