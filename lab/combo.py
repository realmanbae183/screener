"""
lab/combo.py — 지표 10개 조합 연습장 (MNQ 대회용)

하는 일
  1. 나스닥100 선물(NQ=F) 1시간봉 약 2년 + 일봉 15년(200일선용)을 받는다
  2. 지표 10개 중 1~3개를 골라 "전부 켜지면 진입"하는 조합 175개를 만든다
     × 방향 2개(매수/매도) × 보유 기간 3개 × 손절 유무 2개 = 2,100가지
  3. 안전장치
     - 앞 2/3 기간 성적으로만 순위를 매기고, 뒤 1/3은 채점에만 쓴다
     - 몬테카를로: 아무 의미 없는 무작위 진입 1,000개를 똑같이 돌려 '운만으로 나오는 성적'을 잰다
     - 이웃 확인: 같은 조합을 보유 기간·손절만 바꿔도 뒤 기간에 버는지 본다
     - 한 번에 섞는 지표는 3개까지
  결과는 lab/results/combo_*.md 에 남는다.
"""
from __future__ import annotations

import itertools
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mnq  # noqa: E402  (같은 폴더의 연습장: 시세 받기·지표·채점 재사용)

PT, OUT = mnq.PT, mnq.OUT
SIDE_COST = mnq.SIDE_COST
D = 23                    # 나스닥 선물은 하루 약 23시간 거래 → 1시간봉 23개 ≈ 하루
HOLDS = [6, 23, 115]      # 보유: 6시간 / 하루 / 5거래일
STOPS = [None, 1.5]       # 손절 없음 / ATR(14) × 1.5 × √보유시간 (오래 들수록 넓게)
N_RANDOM = int(os.getenv("LAB_RANDOM", "1000"))
SEED = 7


# ─────────────────────────────────────────────
# 지표 10개 — 각각 (매수 쪽 조건, 매도 쪽 조건)
# ─────────────────────────────────────────────
def conditions(df: pd.DataFrame, daily: pd.DataFrame) -> dict[str, tuple[pd.Series, pd.Series]]:
    o, h, l, c, v = (df[k] for k in ("Open", "High", "Low", "Close", "Volume"))
    a14 = mnq.atr(df, 14)
    out = {}

    r = mnq.rsi(c, 14)
    out["RSI 30/70"] = (r < 30, r > 70)

    m, sd = c.rolling(20).mean(), c.rolling(20).std()
    out["볼린저 이탈"] = (c < m - 2 * sd, c > m + 2 * sd)

    s20 = c.rolling(20 * D).mean()
    out["20일선 3% 이격"] = (c < s20 * 0.97, c > s20 * 1.03)        # 지수는 개별주보다 덜 흔들려 10% 대신 3%

    chg = c - c.shift(5)
    out["급락·급등(흔들림 4배)"] = (chg < -4 * a14, chg > 4 * a14)

    hist = (mnq.ema(c, 12) - mnq.ema(c, 26))
    hist = hist - mnq.ema(hist, 9)
    up2 = (hist > hist.shift(1)) & (hist.shift(1) > hist.shift(2))
    dn2 = (hist < hist.shift(1)) & (hist.shift(1) < hist.shift(2))
    out["MACD 돌아섬"] = ((hist < 0) & up2, (hist > 0) & dn2)

    hi20, lo20 = h.rolling(20 * D).max().shift(1), l.rolling(20 * D).min().shift(1)
    out["20일 신고가·신저가"] = (c > hi20, c < lo20)

    tenkan = (h.rolling(9).max() + l.rolling(9).min()) / 2
    kijun = (h.rolling(26).max() + l.rolling(26).min()) / 2
    span_a = ((tenkan + kijun) / 2).shift(26)
    span_b = ((h.rolling(52).max() + l.rolling(52).min()) / 2).shift(26)
    top, bot = np.maximum(span_a, span_b), np.minimum(span_a, span_b)
    out["구름대 돌파"] = ((c > top) & (c.shift(1) <= top.shift(1)), (c < bot) & (c.shift(1) >= bot.shift(1)))

    hr = pd.Series(df.index.hour, index=df.index)
    same_hour_avg = v.groupby(hr.values).transform(lambda s: s.rolling(20, min_periods=10).mean().shift(1))
    big = v > 2 * same_hour_avg                                       # 같은 시간대 평소 거래량의 2배
    out["거래량 급증"] = (big & (c > o), big & (c < o))

    day = pd.Series(df.index.date, index=df.index)
    first = hr == 9                                                   # 동부 9시 봉 (9:30 개장 직후 30분 포함)
    orh = h.where(first).groupby(day.values).transform("max")
    orl = l.where(first).groupby(day.values).transform("min")
    later = (hr >= 10) & (hr <= 15)
    out["개장 돌파"] = (later & (c > orh) & (c.shift(1) <= orh), later & (c < orl) & (c.shift(1) >= orl))

    dd = daily.copy()
    dd["date"] = (dd.index + pd.Timedelta(hours=12)).date            # 일봉 날짜 맞추기
    dd["up"] = dd["Close"] > dd["Close"].rolling(200).mean()
    dd["ok"] = dd["Close"].rolling(200).count() >= 200
    dd = dd[dd["ok"]][["date", "up"]]
    dd["date"] = pd.to_datetime(dd["date"])
    hh = pd.DataFrame({"date": pd.to_datetime(day.values)}, index=df.index).reset_index()
    mg = pd.merge_asof(hh.sort_values("date"), dd.sort_values("date"), on="date", allow_exact_matches=False)
    mg = mg.set_index(mg.columns[0]).reindex(df.index)                # 전날까지의 일봉 기준 (미래 정보 안 씀)
    up = mg["up"].astype("boolean").fillna(False).astype(bool)
    known = mg["up"].notna()
    out["200일선 위·아래"] = (up & known, (~up) & known)

    return {k: (a.fillna(False).to_numpy(bool), b.fillna(False).to_numpy(bool)) for k, (a, b) in out.items()}


# ─────────────────────────────────────────────
# 엔진: 봉 마감에 조건이 다 켜지면 → 다음 봉 시가에 진입 → 정해진 시간 보유 (손절 닿으면 먼저 정리)
# ─────────────────────────────────────────────
def run(df_np, entries: np.ndarray, d: int, hold: int, stop_k):
    o, h, l, c, a14, idx = df_np
    n = len(c)
    pnl = np.zeros(n)
    trades = []
    i = 1
    while i < n:
        if not entries[i - 1]:
            i += 1
            continue
        ent, e_i = o[i], i
        stop = None
        if stop_k is not None and np.isfinite(a14[i - 1]):
            stop = ent - d * stop_k * a14[i - 1] * np.sqrt(hold)
        pnl[i] -= SIDE_COST
        tp = -SIDE_COST
        ref = ent
        j = i
        while j < n:
            px = None
            if stop is not None and ((d > 0 and l[j] <= stop) or (d < 0 and h[j] >= stop)):
                px = (min(o[j], stop) if d > 0 else max(o[j], stop)) if j > e_i else stop
            elif j - e_i + 1 >= hold or j == n - 1:
                px = c[j]
            if px is not None:
                x = d * (px - ref) * PT - SIDE_COST
                pnl[j] += x
                tp += x
                trades.append((idx[e_i], idx[j], d, tp))
                break
            x = d * (c[j] - ref) * PT
            pnl[j] += x
            tp += x
            ref = c[j]
            j += 1
        i = j + 1
    return pnl, trades


def evaluate(pnl, trades, index, cut):
    p = pd.Series(pnl, index=index)
    ta = [t for t in trades if t[0] < cut]
    tb = [t for t in trades if t[0] >= cut]
    A, B = mnq.stats(p[p.index < cut], ta), mnq.stats(p[p.index >= cut], tb)
    WA = mnq.windows(p[p.index < cut], ta)
    WB = mnq.windows(p[p.index >= cut], tb)
    return A, B, WA, WB


def front_score(A, WA) -> float:
    """순위는 앞 기간 성적으로만: 4주 수익 중앙값 ÷ 4주 낙폭 + 플러스 비율. 매매 30번 미만이면 제외."""
    if A["n"] < 30 or not WA:
        return -1e9
    return WA["w_ret"] / max(1.0, abs(WA["w_mdd"])) + WA["w_pos"] / 100


def main():
    os.makedirs(OUT, exist_ok=True)
    df = mnq.load("1h")
    daily = mnq.load("1d")
    cut = df.index[int(len(df) * 2 / 3)]
    conds = conditions(df, daily)
    names = list(conds)
    df_np = (df["Open"].to_numpy(float), df["High"].to_numpy(float), df["Low"].to_numpy(float),
             df["Close"].to_numpy(float), mnq.atr(df, 14).to_numpy(float), df.index)

    # 1) 진짜 조합 2,100개
    rows = []
    for k in (1, 2, 3):
        for combo in itertools.combinations(names, k):
            for d, side in ((1, "매수"), (-1, "매도")):
                e = np.ones(len(df), bool)
                for nm in combo:
                    e &= conds[nm][0 if d > 0 else 1]
                if e.sum() < 5:
                    continue
                for hold in HOLDS:
                    for sk in STOPS:
                        pnl, tr = run(df_np, e, d, hold, sk)
                        A, B, WA, WB = evaluate(pnl, tr, df.index, cut)
                        rows.append({"조합": " + ".join(combo), "방향": side, "보유": hold, "손절": sk is not None,
                                     "앞": A, "뒤": B, "WA": WA, "WB": WB, "fs": front_score(A, WA)})

    # 2) 몬테카를로: 무작위 진입 1,000개 (진짜 조합 하나를 골라 방향·보유·손절·매매 횟수만 흉내)
    rng = np.random.default_rng(SEED)
    pool = [r for r in rows if r["앞"]["n"] + r["뒤"]["n"] >= 30]
    rand = []
    for _ in range(N_RANDOM if pool else 0):
        ref = pool[rng.integers(len(pool))]
        d = 1 if ref["방향"] == "매수" else -1
        k = ref["앞"]["n"] + ref["뒤"]["n"]
        e = np.zeros(len(df), bool)
        e[rng.choice(len(df) - 1, size=min(k, len(df) - 1), replace=False)] = True
        pnl, tr = run(df_np, e, d, ref["보유"], 1.5 if ref["손절"] else None)
        A, B, WA, WB = evaluate(pnl, tr, df.index, cut)
        rand.append({"방향": ref["방향"], "앞": A, "뒤": B, "WA": WA, "WB": WB, "fs": front_score(A, WA)})

    # 3) 이웃 확인: 같은 조합·방향의 6가지 청산 중 뒤 기간 플러스 개수
    nb = {}
    for r in rows:
        key = (r["조합"], r["방향"])
        nb.setdefault(key, []).append(r["뒤"]["ret"] > 0)
    for r in rows:
        v = nb[(r["조합"], r["방향"])]
        r["이웃"] = f"{sum(v)}/{len(v)}"

    md = report(rows, rand, df, cut)
    tag = "1h" if SIDE_COST == 8.0 else f"1h_cost{SIDE_COST:g}"
    with open(os.path.join(OUT, f"combo_{tag}.md"), "w", encoding="utf-8") as f:
        f.write(md)
    keep = sorted(rows, key=lambda r: r["fs"], reverse=True)[:150]
    with open(os.path.join(OUT, f"combo_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(keep, f, ensure_ascii=False, default=str)
    print(md[:8000])


def pct(vals, q):
    return round(float(np.percentile(vals, q)), 1) if len(vals) else None


def report(rows, rand, df, cut) -> str:
    ranked = sorted([r for r in rows if r["fs"] > -1e8], key=lambda r: r["fs"], reverse=True)
    rnd = sorted([r for r in rand if r["fs"] > -1e8], key=lambda r: r["fs"], reverse=True)
    top_n = 20

    def back_ret(rs):
        return [r["뒤"]["ret"] for r in rs]

    # 운의 선: 무작위 진입의 뒤 기간 수익 분포 (방향별)
    luck = {s: back_ret([r for r in rand if r["방향"] == s]) for s in ("매수", "매도")}
    line = {s: pct(v, 95) for s, v in luck.items()}

    def verdict(r):
        bad = []
        if r["뒤"]["ret"] <= 0:
            bad.append("뒤 손실")
        if line.get(r["방향"]) is not None and r["뒤"]["ret"] <= line[r["방향"]]:
            bad.append("운의 선 못 넘음")
        if int(r["이웃"].split("/")[0]) < 4:
            bad.append("이웃 약함")
        if r["뒤"]["tpw"] < 2:
            bad.append("주 2회 미달")
        return "통과" if not bad else ", ".join(bad)

    L = [f"# MNQ 조합 연습 결과 · 편도 비용 ${SIDE_COST:g}", "",
         f"- 데이터: NQ=F 1시간봉 {df.index[0]:%Y-%m-%d} ~ {df.index[-1]:%Y-%m-%d} ({len(df):,}봉) · 앞/뒤 나눈 날 {cut:%Y-%m-%d}",
         f"- 진짜 조합 {len(rows):,}가지 (매매 30번 이상으로 순위에 든 것 {len(ranked):,}가지) · 무작위 진입 {len(rand):,}가지",
         f"- 1계약 · 계좌 ${mnq.CAPITAL:,.0f} 가정. 순위는 앞 기간 성적으로만 매기고, 뒤 기간은 채점용",
         "",
         "## 운만으로는 어디까지 나오나 (무작위 진입, 뒤 기간 수익%)", ""]
    for s in ("매수", "매도"):
        v = luck[s]
        if v:
            L.append(f"- {s}: 중간 {pct(v, 50)}% · 상위 5% 선(운의 선) {pct(v, 95)}% · 최고 {round(max(v), 1)}%  ({len(v)}개)")
    if ranked and rnd:
        a = back_ret(ranked[:top_n])
        b = back_ret(rnd[:top_n])
        L += ["", f"- 앞 기간 1~{top_n}등의 뒤 기간 수익 중간값: 진짜 조합 {pct(a, 50)}% / 무작위 {pct(b, 50)}%",
              "  (진짜가 무작위보다 확실히 높아야 '조합 고르기'에 실력이 있다는 뜻. 이게 안 되면 아래 개별 통과도 운일 가능성이 큼)"]

    L += ["", f"## 앞 기간 상위 40개 (판정: 뒤 플러스 + 운의 선 넘음 + 이웃 4/6 이상 + 주 2회)", "",
          "| 순위 | 조합 | 방향 | 보유 | 손절 | 앞 수익%/MDD% | 뒤 수익%/MDD% | 뒤 4주 중앙% | 뒤 4주 플러스 | 주당 매매 | 이웃 | 판정 |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(ranked[:40], 1):
        wb = r["WB"] or {}
        L.append(f"| {i} | {r['조합']} | {r['방향']} | {r['보유']}h | {'O' if r['손절'] else '-'} | "
                 f"{r['앞']['ret']} / {r['앞']['mdd']} | {r['뒤']['ret']} / {r['뒤']['mdd']} | "
                 f"{wb.get('w_ret')} | {wb.get('w_pos')}% | {r['뒤']['tpw']} | {r['이웃']} | {verdict(r)} |")

    passed = [r for r in ranked if verdict(r) == "통과"]
    L += ["", f"## 판정 통과 ({len(passed)}개, 앞 기간 순위 순)", ""]
    for r in passed[:20]:
        wb = r["WB"] or {}
        L.append(f"- {r['조합']} · {r['방향']} · {r['보유']}시간 보유 · 손절 {'있음' if r['손절'] else '없음'} — "
                 f"뒤 {r['뒤']['ret']}% / MDD {r['뒤']['mdd']}% · 4주 중앙 {wb.get('w_ret')}% · 주 {r['뒤']['tpw']}회 · 매매 {r['뒤']['n']}번")
    if not passed:
        L.append("- 없음")

    nm_stats = {}
    for r in ranked:
        for nm in r["조합"].split(" + "):
            nm_stats.setdefault(nm, []).append(r["뒤"]["ret"])
    L += ["", "## 지표별: 그 지표가 들어간 조합들의 뒤 기간 수익 중간값", ""]
    for nm, v in sorted(nm_stats.items(), key=lambda x: -np.median(x[1])):
        L.append(f"- {nm}: {pct(v, 50)}% ({len(v)}개)")
    return "\n".join(L)


if __name__ == "__main__":
    main()
