"""
가격대별 진입 사다리.

"살까 말까"의 이분법 대신 "지금도 괜찮고, 얼마까지 내려가면 더 좋다"를 답한다.
손절선이 없는 전략에서 분할 매수는 사실상 유일한 리스크 관리 수단이다.
진입 가격이 유일한 방어선이므로, 그것을 한 점이 아니라 구간으로 만든다.

각 칸은 실제 차트 위의 의미 있는 지점에서 나온다.
  · 피보나치 되돌림 (50% / 61.8% / 78.6%)
  · 최근 스윙 저점
  · 200일 이동평균
  · ATR 배수 (그 종목의 평소 변동폭)
중복되는 후보는 묶고, 현재가 아래 것만 남긴다.
"""
import numpy as np
import pandas as pd

import config as C
import indicators as I


def _levels(df: pd.DataFrame) -> list[dict]:
    """차트에서 의미 있는 가격대 후보를 모은다."""
    close = df["Close"]
    px = float(close.iloc[-1])
    out = []

    win = close.tail(C.FIB_LOOKBACK)
    hi = float(win.max())
    hi_idx = win.idxmax()
    after = win.loc[hi_idx:]
    lo = float(after.min()) if len(after) > 5 else float(win.min())

    # 피보나치 되돌림 — 고점에서 얼마나 내려온 지점인가
    if hi > lo:
        span = hi - lo
        for pct, label in ((0.500, "50% 되돌림"),
                           (0.618, "61.8% 되돌림"),
                           (0.786, "78.6% 되돌림")):
            out.append({"price": hi - span * pct, "why": label})

    # 최근 스윙 저점
    recent_low = float(close.tail(C.STRUCT_LOOKBACK).min())
    out.append({"price": recent_low, "why": "최근 저점"})

    # 200일선 — 장기 지지선 역할
    ma = I.sma(close, C.TREND_MA)
    if not pd.isna(ma.iloc[-1]):
        out.append({"price": float(ma.iloc[-1]), "why": "200일선"})

    # ATR 배수 — 그 종목의 평소 변동폭 기준
    a = I.atr(df)
    if not pd.isna(a.iloc[-1]) and a.iloc[-1] > 0:
        av = float(a.iloc[-1])
        for k in (2, 4):
            out.append({"price": px - av * k, "why": f"현재가 −{k} ATR"})

    return [o for o in out if o["price"] > 0]


def _recovery_target(df: pd.DataFrame, hi: float) -> float:
    """
    회복 목표가 — '정상으로 돌아왔다면 이쯤'이라고 볼 기준점.
    200일선과 52주 고점의 일정 비율 중 높은 쪽을 쓴다.
    목표가가 아니라 매력도를 재기 위한 자(尺)다.
    """
    ma = I.sma(df["Close"], C.TREND_MA)
    ma_v = float(ma.iloc[-1]) if not pd.isna(ma.iloc[-1]) else 0.0
    return max(ma_v, hi * C.LADDER_TARGET_OF_HIGH)


def _attractiveness(price: float, target: float,
                    chart_score: float, fin_score: float | None) -> tuple[float, float]:
    """
    그 가격에 샀을 때의 매력도(0~100)와 기대수익률.
      · 회복 목표가까지의 기대수익 (45점) — 낮게 살수록 커진다
      · 차트 점수 = 타이밍 (25점)
      · 재무 점수 = 품질 (30점)
    """
    exp_ret = (target / price - 1) if price > 0 else 0.0
    ret_pt = float(np.clip(exp_ret / C.LADDER_RETURN_FULL, 0, 1)) * 45
    chart_pt = float(np.clip(chart_score / C.MAX_SCORE, 0, 1)) * 25
    fin_pt = (30 * 0.55 if fin_score is None
              else float(np.clip(fin_score / 100, 0, 1)) * 30)
    return round(ret_pt + chart_pt + fin_pt, 0), round(exp_ret, 4)


def build_ladder(df: pd.DataFrame, chart_score: float,
                 fin_score: float | None = None,
                 in_channel: bool = False) -> dict:
    """
    진입 사다리를 만든다.
    in_channel(하락 채널 진행 중)이면 1차 비중을 줄이고 아래 칸을 두껍게 한다.
    """
    close = df["Close"]
    px = float(close.iloc[-1])
    hi = float(close.tail(C.FIB_LOOKBACK).max())

    cands = [x for x in _levels(df) if x["price"] < px * 0.985]
    cands.sort(key=lambda x: -x["price"])

    # 서로 가까운 가격대는 하나로 묶는다 (같은 지지선을 두 번 세지 않게)
    merged: list[dict] = []
    for c in cands:
        if merged and abs(merged[-1]["price"] - c["price"]) / merged[-1]["price"] \
                < C.LADDER_MERGE_PCT:
            merged[-1]["why"] += f" · {c['why']}"
        else:
            merged.append(dict(c))
    merged = merged[:C.LADDER_STEPS - 1]

    weights = (C.LADDER_WEIGHTS_CHANNEL if in_channel
               else C.LADDER_WEIGHTS_NORMAL)[:len(merged) + 1]
    total_w = sum(weights)
    weights = [w / total_w for w in weights]

    target = _recovery_target(df, hi)

    att, ret = _attractiveness(px, target, chart_score, fin_score)
    rungs = [{
        "step": 1, "price": round(px, 2), "why": "현재가",
        "weight": round(weights[0], 3),
        "attractiveness": att, "expected_return": ret,
        "drop_from_now": 0.0,
    }]
    for i, m in enumerate(merged, start=1):
        p = m["price"]
        att, ret = _attractiveness(p, target, chart_score, fin_score)
        rungs.append({
            "step": i + 1, "price": round(p, 2), "why": m["why"],
            "weight": round(weights[i], 3) if i < len(weights) else 0.0,
            "attractiveness": att, "expected_return": ret,
            "drop_from_now": round(p / px - 1, 4),
        })

    # 가설 폐기선 — 여기까지 오면 판단이 틀린 것으로 본다
    invalid = min(r["price"] for r in rungs) * (1 - C.LADDER_INVALID_BELOW)

    return {
        "current_price": round(px, 2),
        "high_52w": round(hi, 2),
        "recovery_target": round(target, 2),
        "in_channel": bool(in_channel),
        "rungs": rungs,
        "invalidation": round(invalid, 2),
        "invalidation_drop": round(invalid / px - 1, 4),
        "first_weight": rungs[0]["weight"],
    }
