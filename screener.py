"""
스크리너 파이프라인.

  ① 유니버스 필터  → 시총/섹터/히스토리
  ② 추세 필터      → 떨어지는 칼날 차단  ★핵심
  ③ 과매도 스코어  → 7개 지표, 17점 만점
  냉각 기간        → 급락 직후 제외 (EPS 추정치 반영 대기)

④⑤(악재 스크리닝·13F)는 여기 통과한 소수 종목에만 적용한다.
"""
import pandas as pd

import config as C
import data as D
import indicators as I


def screen_one(df: pd.DataFrame) -> dict:
    """단일 종목 평가. 탈락해도 사유를 남긴다."""
    out = {"passed": False, "stage": None, "score": 0, "reason": ""}

    if len(df) < C.MIN_HISTORY_DAYS:
        out.update(stage="history", reason=f"봉 {len(df)}개 (최소 {C.MIN_HISTORY_DAYS})")
        return out

    # ② 추세 필터
    t_ok, t_info = I.trend_ok(df)
    out["trend"] = t_info
    if not t_ok:
        out.update(stage="trend",
                   reason=f"추세 이탈 (200일선 대비 {t_info.get('price_vs_ma200', 0):+.1%}, "
                          f"기울기 {t_info.get('ma200_slope', 0):+.2%})")
        return out

    # ②-b 구조 판정 — 하락 채널이어도 차단하지 않는다.
    #   재무가 좋으면 채널 안에서도 결국 돌아온다는 판단에 따라,
    #   차단 대신 1차 진입 비중을 줄이고 아래 칸을 두껍게 만든다.
    #   (STRUCT_AS_VETO=True로 바꾸면 예전처럼 차단)
    st_ok, st_info = I.structure_ok(df)
    out["structure"] = st_info
    out["in_channel"] = not st_ok
    if not st_ok and C.STRUCT_AS_VETO:
        out.update(stage="structure", reason=st_info.get("reason", "하락 채널"))
        return out

    # 냉각 기간
    c_ok, c_info = I.cooling_ok(df)
    out["cooling"] = c_info
    if not c_ok:
        out.update(stage="cooling",
                   reason=f"급락 직후 ({c_info['recent_crash_pct']:+.1%}), "
                          f"{c_info['wait_more_days']}일 더 대기")
        return out

    # ③ 과매도 스코어
    s = I.oversold_score(df)
    out["score"] = s["score"]
    out["detail"] = s["detail"]
    if s["score"] < C.SCORE_THRESHOLD:
        out.update(stage="score",
                   reason=f"스코어 {s['score']}/{C.MAX_SCORE} (기준 {C.SCORE_THRESHOLD})")
        return out

    out.update(passed=True, stage="passed",
               reason=f"통과 — 스코어 {s['score']}/{C.MAX_SCORE}")
    return out


def apply_universe_filter(uni: pd.DataFrame, meta: pd.DataFrame) -> tuple[list[str], dict]:
    """
    ① 섹터·시총 필터.
    섹터/산업은 동봉 유니버스 CSV에서 바로 읽고(빠름),
    시총은 yfinance 메타데이터가 있을 때만 적용한다(없으면 생략).
    """
    m = meta.set_index("ticker") if not meta.empty else None
    kept, dropped = [], {"industry": 0, "sector": 0, "market_cap": 0}

    for _, row in uni.iterrows():
        t = row["ticker"]
        sector = str(row.get("sector") or "")
        industry = str(row.get("industry") or "")

        if sector and sector in C.EXCLUDE_SECTORS:
            dropped["sector"] += 1
            continue
        if industry and any(x in industry for x in C.EXCLUDE_INDUSTRIES):
            dropped["industry"] += 1
            continue
        if m is not None and t in m.index:
            cap = m.loc[t].get("market_cap")
            if pd.notna(cap) and cap < C.MIN_MARKET_CAP:
                dropped["market_cap"] += 1
                continue
        kept.append(t)

    if m is None:
        dropped["note"] = "시총 필터 미적용 (메타데이터 없음)"
    return kept, dropped


def run(uni: pd.DataFrame, conn, verbose: bool = True) -> tuple[pd.DataFrame, dict]:
    """전체 유니버스 스크리닝. (후보 DataFrame, 단계별 통계) 반환."""
    meta = D.load_meta(conn)
    tickers, uni_drop = apply_universe_filter(uni, meta)
    if verbose:
        print(f"① 유니버스 필터 후: {len(tickers)}종목  {uni_drop}")

    # 티커 → 회사명/섹터 (리포트에서 사람이 읽을 수 있게)
    info = {r["ticker"]: (str(r.get("name") or r["ticker"]), str(r.get("sector") or ""))
            for _, r in uni.iterrows()}

    rows, funnel = [], {"nodata": 0, "history": 0, "trend": 0, "structure": 0,
                        "cooling": 0, "score": 0, "passed": 0}
    near_miss = []

    for t in tickers:
        df = D.load(conn, t)
        if df.empty:
            funnel["nodata"] += 1
            continue
        r = screen_one(df)
        funnel[r["stage"]] = funnel.get(r["stage"], 0) + 1

        if r["passed"]:
            d = r["detail"]
            nm, sec = info.get(t, (t, ""))
            rows.append({
                "ticker": t,
                "name": nm,
                "sector": sec,
                "score": r["score"],
                "price": round(r["trend"]["price"], 2),
                "vs_ma200": round(r["trend"]["price_vs_ma200"], 4),
                "ma200_slope": round(r["trend"]["ma200_slope"], 4),
                "atr_dist": d["atr_distance"]["value"],
                "rsi": d["rsi"]["value"],
                "pct_b": d["bollinger_pct_b"]["value"],
                "dd_52w": d["drawdown_52w"]["value"],
                "fib": d["fib_retracement"]["value"],
                "vol_climax": d["volume_climax"]["hit"],
                "macd_div": d["macd_divergence"]["hit"],
                "hits": ",".join(k for k, v in d.items() if v["hit"]),
                # 지표별 획득 점수 (부분점수제라 정수가 아니다)
                "scores": ",".join(f"{k}:{v['score']:g}" for k, v in d.items()),
                "in_channel": bool(r.get("in_channel", False)),
            })
        elif r["stage"] == "score" and r["score"] >= C.SCORE_THRESHOLD - 3:
            near_miss.append({"ticker": t, "score": r["score"], "reason": r["reason"]})

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("score", ascending=False).reset_index(drop=True)

    funnel["near_miss"] = len(near_miss)
    return out, {"funnel": funnel, "near_miss": near_miss}


def format_report(cands: pd.DataFrame, stats: dict) -> str:
    f = stats["funnel"]
    total = sum(f.get(k, 0) for k in
                ["nodata", "history", "trend", "structure", "cooling",
                 "score", "passed"])
    L = []
    L.append("=" * 74)
    L.append("스크리닝 결과")
    L.append("=" * 74)
    nodata = f.get("nodata", 0) + f.get("history", 0)
    if nodata:
        pct = nodata / total * 100 if total else 0
        flag = "  ⚠ 데이터 누락!" if pct > 5 else ""
        L.append(f"  데이터 없음      : {f.get('nodata', 0):>4}{flag}")
        L.append(f"  히스토리 부족    : {f.get('history', 0):>4}")
    L.append(f"  ② 추세 이탈 차단 : {f.get('trend', 0):>4}   ← 떨어지는 칼날")
    L.append(f"  ②-b 하락 채널    : {f.get('structure', 0):>4}   ← 역배열·고점 낮아짐")
    L.append(f"  냉각 기간 대기   : {f.get('cooling', 0):>4}   ← 급락 직후, EPS 미반영")
    L.append(f"  ③ 스코어 미달    : {f.get('score', 0):>4}   (근접 {f.get('near_miss',0)}종목)")
    L.append(f"  ✅ 최종 후보     : {f.get('passed', 0):>4}")
    evaluated = total - nodata
    if evaluated:
        L.append(f"  (실제 평가 {evaluated}종목 중 통과율 "
                 f"{f.get('passed',0)/evaluated*100:.1f}%)")
    L.append("=" * 74)
    if nodata and total and nodata / total > 0.05:
        L.append("")
        L.append(f"⚠ {nodata}종목의 데이터가 비어 있습니다. 결과가 시장 전체를")
        L.append("  대표하지 못합니다. `py run.py --repair` 로 먼저 채우세요.")
    L.append("=" * 74)

    if cands.empty:
        L.append("\n오늘 조건을 만족하는 종목이 없습니다.")
        if stats["near_miss"]:
            L.append("\n[근접 종목 — 스코어 기준에 3점 이내]")
            for n in sorted(stats["near_miss"], key=lambda x: -x["score"])[:10]:
                L.append(f"  {n['ticker']:<6} {n['reason']}")
        return "\n".join(L)

    L.append("")
    for _, r in cands.iterrows():
        L.append(f"▸ {r['ticker']}   스코어 {r['score']}/{C.MAX_SCORE}   ${r['price']}")
        L.append(f"    추세   : 200일선 대비 {r['vs_ma200']:+.1%}, "
                 f"기울기 {r['ma200_slope']:+.2%}")
        L.append(f"    과매도 : ATR이격 {r['atr_dist']}배 | RSI {r['rsi']} | "
                 f"%B {r['pct_b']} | 52주낙폭 {r['dd_52w']:.1%}")
        L.append(f"    확인   : 거래량소진 {'O' if r['vol_climax'] else 'X'} | "
                 f"MACD다이버전스 {'O' if r['macd_div'] else 'X'}")
        L.append(f"    → 다음 단계: ④악재 스크리닝(EPS 개정·GPM) → ⑤13F 확인")
        L.append("")
    return "\n".join(L)
