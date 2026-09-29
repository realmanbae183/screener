"""
④ 악재 스크리닝 — 차트로 걸러진 소수 후보에만 적용한다.

"왜 빠졌나"를 다섯 갈래로 본다.
  1. EPS 개정 모멘텀   추정치가 어느 방향으로 얼마나 빨리 움직였나
  2. 애널리스트 상하향  최근 30일 상향/하향 인원 수
  3. GPM 추세          매출총이익률 — 점유율 상실은 EPS보다 여기 먼저 나타난다
  4. 실적 블랙아웃      발표 직전·직후는 컨센서스가 아직 안 고쳐졌다
  5. 기관 보유          13F의 간이 버전 (정식 13F는 ⑤단계)

핵심 판단: 주가는 빠졌는데 추정치는 그대로인가?
  - 그대로  → 밸류에이션만 깎인 것. 기회.
  - 급락    → 실적 전망이 실제로 망가진 것. 차단.
  - 완만한 하향 → 구조적 침식 의심. 감점. (인텔 패턴)
"""
import datetime as dt

import numpy as np
import pandas as pd
import yfinance as yf

import config as C


def _safe(fn, default=None):
    try:
        v = fn()
        return default if v is None else v
    except Exception:
        return default


# ═══════════════════════════════════════════
# 1·2. EPS 추정치
# ═══════════════════════════════════════════

def eps_revision(tk: yf.Ticker) -> dict:
    """
    eps_trend: 기간별(0q/+1q/0y/+1y) 현재·7일전·30일전·60일전·90일전 컨센서스.
    '+1y'(내년 EPS)를 기준으로 삼는다 — 지금 빠진 이유가 미래에 반영됐는지를 보는 것.
    """
    out = {"available": False}
    trend = _safe(lambda: tk.eps_trend)
    if trend is None or not isinstance(trend, pd.DataFrame) or trend.empty:
        return out

    period = None
    for p in ("+1y", "0y", "+1q"):
        if p in trend.index:
            period = p
            break
    if period is None:
        return out

    row = trend.loc[period]

    def g(col):
        v = row.get(col)
        try:
            v = float(v)
            return v if np.isfinite(v) else None
        except (TypeError, ValueError):
            return None

    cur = g("current")
    if cur is None or cur == 0:
        return out

    out.update(available=True, period=period, current=cur)
    for label, col in (("d7", "7daysAgo"), ("d30", "30daysAgo"),
                       ("d60", "60daysAgo"), ("d90", "90daysAgo")):
        past = g(col)
        out[label] = None if past in (None, 0) else (cur / past - 1)
        out[f"{label}_value"] = past

    # 판정 기준은 90일. 없으면 60일 → 30일 순으로 대체
    ref = next((out[k] for k in ("d90", "d60", "d30") if out.get(k) is not None), None)
    out["revision"] = ref

    if ref is None:
        out["verdict"] = "unknown"
    elif ref <= C.EPS_REVISION_BLOCK:
        out["verdict"] = "block"        # 전망이 실제로 망가짐
    elif ref <= C.EPS_REVISION_PENALTY:
        out["verdict"] = "penalty"      # 완만한 하향 = 구조적 침식 의심
    elif ref >= 0.03:
        out["verdict"] = "bonus"        # 상향 중
    else:
        out["verdict"] = "hold"         # 유지 = 밸류에이션만 깎임
    return out


def eps_analyst_moves(tk: yf.Ticker) -> dict:
    """최근 30일 상향/하향 애널리스트 수."""
    out = {"available": False}
    rev = _safe(lambda: tk.eps_revisions)
    if rev is None or not isinstance(rev, pd.DataFrame) or rev.empty:
        return out
    period = next((p for p in ("+1y", "0y", "+1q") if p in rev.index), None)
    if period is None:
        return out
    row = rev.loc[period]

    def g(*names):
        for n in names:
            if n in row.index:
                try:
                    v = float(row[n])
                    if np.isfinite(v):
                        return int(v)
                except (TypeError, ValueError):
                    pass
        return None

    up = g("upLast30days", "upLast30Days")
    dn = g("downLast30days", "downLast30Days")
    if up is None and dn is None:
        return out
    up, dn = up or 0, dn or 0
    out.update(available=True, up=up, down=dn, net=up - dn)
    return out


# ═══════════════════════════════════════════
# 3. 매출총이익률(GPM) 추세
# ═══════════════════════════════════════════

def gpm_trend(tk: yf.Ticker) -> dict:
    """
    점유율을 잃으면 EPS보다 마진이 먼저 깎인다.
    EPS가 멀쩡해도 GPM이 연속 하락하면 차단한다.
    """
    out = {"available": False}
    fin = _safe(lambda: tk.quarterly_income_stmt)
    if fin is None or not isinstance(fin, pd.DataFrame) or fin.empty:
        return out

    def pick(*names):
        for n in names:
            if n in fin.index:
                return fin.loc[n]
        return None

    rev = pick("Total Revenue", "Operating Revenue")
    gp = pick("Gross Profit")
    if rev is None:
        return out

    if gp is None:
        cost = pick("Cost Of Revenue", "Cost of Revenue",
                    "Reconciled Cost Of Revenue")
        if cost is None:
            return out
        gp = rev - cost

    # 컬럼은 최신순 — 오래된 것부터 정렬
    s = (gp / rev.replace(0, np.nan)).dropna().sort_index()
    if len(s) < 2:
        return out

    vals = [float(v) for v in s.tail(C.GPM_DECLINE_QUARTERS + 1)]
    quarters = [str(d)[:10] for d in s.tail(C.GPM_DECLINE_QUARTERS + 1).index]

    declines = sum(1 for a, b in zip(vals, vals[1:]) if b < a)
    consecutive = 0
    for a, b in zip(vals, vals[1:]):
        consecutive = consecutive + 1 if b < a else 0

    out.update(available=True, values=[round(v, 4) for v in vals],
               quarters=quarters, latest=round(vals[-1], 4),
               change=round(vals[-1] - vals[0], 4),
               consecutive_declines=consecutive, declines=declines)
    out["verdict"] = "block" if consecutive >= C.GPM_DECLINE_QUARTERS else (
        "penalty" if vals[-1] < vals[0] else "ok")
    return out


# ═══════════════════════════════════════════
# 4. 실적 블랙아웃
# ═══════════════════════════════════════════

def earnings_window(tk: yf.Ticker) -> dict:
    """발표 직전·직후는 컨센서스가 아직 안 고쳐져 있어 평가를 미룬다."""
    out = {"available": False}
    ed = _safe(lambda: tk.earnings_dates)
    if ed is None or not isinstance(ed, pd.DataFrame) or ed.empty:
        return out

    try:
        idx = pd.to_datetime(ed.index).tz_localize(None)
    except (TypeError, AttributeError):
        try:
            idx = pd.to_datetime(ed.index.tz_convert(None))
        except Exception:
            return out

    today = pd.Timestamp(dt.date.today())
    past = idx[idx <= today]
    future = idx[idx > today]

    last = past.max() if len(past) else None
    nxt = future.min() if len(future) else None
    d_since = int((today - last).days) if last is not None else None
    d_until = int((nxt - today).days) if nxt is not None else None

    blackout = ((d_since is not None and d_since <= C.EARNINGS_BLACKOUT_DAYS) or
                (d_until is not None and d_until <= C.EARNINGS_BLACKOUT_DAYS))
    out.update(available=True,
               last=str(last.date()) if last is not None else None,
               next=str(nxt.date()) if nxt is not None else None,
               days_since=d_since, days_until=d_until, blackout=bool(blackout))
    out["verdict"] = "block" if blackout else "ok"
    return out


# ═══════════════════════════════════════════
# 5. 기관 보유 (13F 간이 버전)
# ═══════════════════════════════════════════

def institutions(tk: yf.Ticker) -> dict:
    """정식 13F는 ⑤단계. 여기선 보유 비중과 상위 기관만 본다."""
    out = {"available": False}
    h = _safe(lambda: tk.institutional_holders)
    if h is None or not isinstance(h, pd.DataFrame) or h.empty:
        return out

    top = []
    for _, r in h.head(5).iterrows():
        pct = r.get("pctHeld", r.get("% Out"))
        try:
            pct = float(pct)
        except (TypeError, ValueError):
            pct = None
        top.append({"holder": str(r.get("Holder", "?")),
                    "pct": round(pct * 100, 2) if pct is not None and pct < 1 else pct,
                    "date": str(r.get("Date Reported", ""))[:10]})

    pct_inst = None
    mh = _safe(lambda: tk.major_holders)
    if isinstance(mh, pd.DataFrame) and not mh.empty:
        try:
            if "institutionsPercentHeld" in mh.index:
                pct_inst = round(float(mh.loc["institutionsPercentHeld"].iloc[0]) * 100, 1)
        except Exception:
            pass

    out.update(available=True, top=top, count=len(h),
               institutions_pct=pct_inst)
    return out


# ═══════════════════════════════════════════
# ④-b 재무 점수 (100점) — 후보 순위를 정하는 주 기준
# ═══════════════════════════════════════════

FIN_LABEL = {
    "roe": "ROE",
    "operating_mgn": "영업이익률",
    "fcf_positive": "현금흐름",
    "debt_to_equity": "부채비율",
    "current_ratio": "유동비율",
    "revenue_growth": "매출성장률",
    "forward_pe": "선행 PER",
    "price_to_book": "PBR",
}

_INFO_KEYS = {
    "roe": ("returnOnEquity",),
    "operating_mgn": ("operatingMargins",),
    "debt_to_equity": ("debtToEquity",),
    "current_ratio": ("currentRatio",),
    "revenue_growth": ("revenueGrowth",),
    "forward_pe": ("forwardPE", "trailingPE"),
    "price_to_book": ("priceToBook",),
}


def _scale(v: float, best: float, worst: float) -> float:
    """best일 때 1.0, worst일 때 0.0. 그 사이는 비례. 범위 밖은 잘라낸다."""
    if best == worst:
        return 0.0
    return float(np.clip((v - worst) / (best - worst), 0.0, 1.0))


def financial_score(tk: yf.Ticker) -> dict:
    """
    수익성·건전성·성장·밸류에이션을 100점으로 환산한다.
    조회 못 한 항목은 배점에서 빼고, 받은 항목만으로 백분율을 낸다
    (빠진 항목을 0점 처리하면 데이터 부실한 종목이 부당하게 깎인다).
    """
    info = _safe(lambda: tk.info, {}) or {}
    if not isinstance(info, dict) or not info:
        return {"available": False}

    def num(*keys):
        for k in keys:
            v = info.get(k)
            try:
                v = float(v)
                if np.isfinite(v):
                    return v
            except (TypeError, ValueError):
                continue
        return None

    got, earned, possible = {}, 0.0, 0.0

    for key, (pts, best, worst) in C.FIN_WEIGHTS.items():
        if key == "fcf_positive":
            fcf = num("freeCashflow")
            if fcf is None:
                continue
            raw, frac = fcf, (1.0 if fcf > 0 else 0.0)
        else:
            raw = num(*_INFO_KEYS[key])
            if raw is None:
                continue
            frac = _scale(raw, best, worst)

        pt = pts * frac
        earned += pt
        possible += pts
        got[key] = {"label": FIN_LABEL[key], "raw": raw,
                    "points": round(pt, 1), "max": pts}

    if possible < 40:            # 절반도 못 받았으면 점수를 신뢰할 수 없다
        return {"available": False, "coverage": round(possible, 1),
                "items": got}

    score = earned / possible * 100
    grade = next(g for cut, g in C.FIN_GRADE_CUTS if score >= cut)

    # 항목별 강점·약점 (배점 대비 획득률)
    ranked = sorted(got.items(), key=lambda kv: kv[1]["points"] / kv[1]["max"])
    weak = [FIN_LABEL[k] for k, v in ranked[:2] if v["points"] / v["max"] < 0.4]
    strong = [FIN_LABEL[k] for k, v in ranked[::-1][:2]
              if v["points"] / v["max"] > 0.75]

    return {"available": True, "score": round(score, 1), "grade": grade,
            "coverage": round(possible, 1), "items": got,
            "strong": strong, "weak": weak,
            "sector": info.get("sector"), "name": info.get("shortName")}


# ═══════════════════════════════════════════
# 종합
# ═══════════════════════════════════════════

def analyze(ticker: str) -> dict:
    """한 종목의 ④단계 전체. 어떤 조회가 실패해도 나머지는 살린다."""
    tk = yf.Ticker(ticker)
    r = {
        "ticker": ticker,
        "eps": eps_revision(tk),
        "moves": eps_analyst_moves(tk),
        "gpm": gpm_trend(tk),
        "earnings": earnings_window(tk),
        "inst": institutions(tk),
        "fin": financial_score(tk),
    }

    blocks, penalties, reasons = [], [], []

    if r["earnings"].get("verdict") == "block":
        e = r["earnings"]
        when = (f"{e['days_until']}일 후 발표" if e.get("days_until") is not None
                and e["days_until"] <= C.EARNINGS_BLACKOUT_DAYS
                else f"{e['days_since']}일 전 발표")
        blocks.append("실적 블랙아웃")
        reasons.append(f"실적발표 {when} — 컨센서스 미반영 구간")

    ev = r["eps"].get("verdict")
    if ev == "block":
        blocks.append("EPS 급락")
        reasons.append(f"EPS 추정치 {r['eps']['revision']*100:+.1f}% — 전망이 실제로 악화")
    elif ev == "penalty":
        penalties.append("EPS 완만한 하향")
        reasons.append(f"EPS 추정치 {r['eps']['revision']*100:+.1f}% — 구조적 침식 의심")
    elif ev == "hold":
        reasons.append(f"EPS 추정치 {r['eps']['revision']*100:+.1f}% 유지 — 밸류에이션만 조정")
    elif ev == "bonus":
        reasons.append(f"EPS 추정치 {r['eps']['revision']*100:+.1f}% 상향 중")

    gv = r["gpm"].get("verdict")
    if gv == "block":
        blocks.append("마진 연속 악화")
        reasons.append(f"매출총이익률 {r['gpm']['consecutive_declines']}분기 연속 하락 "
                       f"— 점유율 상실 신호")
    elif gv == "penalty":
        penalties.append("마진 하락")
        reasons.append(f"매출총이익률 {r['gpm']['change']*100:+.1f}%p")
    elif gv == "ok":
        reasons.append(f"매출총이익률 {r['gpm']['latest']*100:.1f}% (유지·개선)")

    if r["moves"].get("available"):
        m = r["moves"]
        if m["net"] < -2:
            penalties.append("애널리스트 하향 우세")
        reasons.append(f"최근 30일 상향 {m['up']} / 하향 {m['down']}")

    # 데이터가 없는 것은 '문제 없음'이 아니다. 판정 불가로 따로 표시한다.
    have_eps = r["eps"].get("available", False)
    have_gpm = r["gpm"].get("available", False)

    if blocks:
        verdict, grade = "차단", "C"
    elif not have_eps and not have_gpm:
        verdict, grade = "확인불가", "?"
        reasons.append("추정치·재무 데이터를 가져오지 못했습니다 — 직접 확인 필요")
    elif len(penalties) >= 2:
        verdict, grade = "주의", "B"
    elif penalties:
        verdict, grade = "조건부", "B"
    elif not (have_eps and have_gpm):
        verdict, grade = "일부확인", "B"
        reasons.append("일부 데이터 누락 — " +
                       ("EPS 추정치 없음" if not have_eps else "재무 데이터 없음"))
    else:
        verdict, grade = "통과", "A"

    fin = r["fin"]
    if fin.get("available"):
        reasons.insert(0, f"재무 점수 {fin['score']:.0f}점 ({fin['grade']}등급)"
                       + (f" · 강점 {', '.join(fin['strong'])}" if fin["strong"] else "")
                       + (f" · 약점 {', '.join(fin['weak'])}" if fin["weak"] else ""))

    r.update(verdict=verdict, grade=grade, blocks=blocks,
             penalties=penalties, reasons=reasons,
             fin_score=fin.get("score"), fin_grade=fin.get("grade"))
    return r


def analyze_many(tickers: list[str], verbose: bool = True) -> dict:
    out = {}
    for i, t in enumerate(tickers, 1):
        if verbose:
            print(f"  [{i}/{len(tickers)}] {t} 조회 중...")
        try:
            out[t] = analyze(t)
        except Exception as e:
            out[t] = {"ticker": t, "verdict": "오류", "grade": "?",
                      "blocks": [], "penalties": [],
                      "reasons": [f"조회 실패: {type(e).__name__}"],
                      "fin_score": None, "fin_grade": None,
                      "eps": {}, "moves": {}, "gpm": {}, "earnings": {},
                      "inst": {}, "fin": {}}
    return out
