"""
④단계 판정 로직 검증.
야후에 실제로 붙지 않고, 가짜 응답을 주입해 판정만 확인한다.
"""
import datetime as dt
import sys

import numpy as np
import pandas as pd
import yfinance as yf

import fundamental as F

results = []


def check(name, got, want):
    ok = got == want
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL':4} | {name:44} got={got!r:<12} want={want!r}")


def mk_trend(cur, d90):
    """eps_trend 흉내 — 기간 인덱스 + 시점별 컨센서스."""
    return pd.DataFrame(
        {"current": [cur], "7daysAgo": [cur], "30daysAgo": [(cur + d90) / 2],
         "60daysAgo": [d90], "90daysAgo": [d90]},
        index=["+1y"])


def mk_revisions(up, down):
    return pd.DataFrame({"upLast30days": [up], "downLast30days": [down]},
                        index=["+1y"])


def mk_income(gpms, revenue=1000.0):
    """quarterly_income_stmt 흉내 — 컬럼이 최신순(내림차순)인 실제 구조 그대로."""
    dates = pd.to_datetime(["2026-06-30", "2026-03-31", "2025-12-31", "2025-09-30"])
    gp = [revenue * g for g in gpms]          # gpms: 오래된 것 → 최신
    return pd.DataFrame(
        {d: [revenue, v] for d, v in zip(dates, gp[::-1])},   # 최신순으로 뒤집기
        index=["Total Revenue", "Gross Profit"])


def mk_earnings(days_offset):
    """days_offset: 음수=과거 발표, 양수=미래 발표"""
    base = dt.date.today()
    dates = [base + dt.timedelta(days=days_offset),
             base + dt.timedelta(days=days_offset - 91)]
    return pd.DataFrame({"EPS Estimate": [1.0, 1.0]},
                        index=pd.to_datetime(dates))


class FakeTicker:
    def __init__(self, **kw):
        self._d = kw

    def __getattr__(self, name):
        if name in self._d:
            v = self._d[name]
            if isinstance(v, Exception):
                raise v
            return v
        raise AttributeError(name)


SCENARIOS = {}


def scenario(name, **kw):
    SCENARIOS[name] = FakeTicker(**kw)
    return name


# 1) 기회: 주가는 빠졌는데 EPS 추정치는 그대로, 마진도 개선
scenario("OPPORTUNITY",
         eps_trend=mk_trend(2.05, 2.08),                  # -1.4% 유지
         eps_revisions=mk_revisions(4, 1),
         quarterly_income_stmt=mk_income([0.30, 0.31, 0.32, 0.33]),
         earnings_dates=mk_earnings(45),
         institutional_holders=pd.DataFrame(
             {"Holder": ["Vanguard", "BlackRock"], "pctHeld": [0.11, 0.09],
              "Date Reported": ["2026-06-30", "2026-06-30"]}),
         major_holders=pd.DataFrame({"Value": [0.82]},
                                    index=["institutionsPercentHeld"]))

# 2) 차단: EPS 추정치 급락 — 전망이 실제로 망가짐
scenario("EPS_CRASH",
         eps_trend=mk_trend(1.50, 2.10),                  # -28.6%
         eps_revisions=mk_revisions(0, 9),
         quarterly_income_stmt=mk_income([0.30, 0.30, 0.31, 0.30]),
         earnings_dates=mk_earnings(40))

# 3) 인텔 패턴: EPS는 완만한 하향인데 마진이 3분기 연속 하락
scenario("INTEL_PATTERN",
         eps_trend=mk_trend(1.90, 2.05),                  # -7.3% 완만
         eps_revisions=mk_revisions(1, 5),
         quarterly_income_stmt=mk_income([0.42, 0.39, 0.36, 0.33]),
         earnings_dates=mk_earnings(50))

# 4) 블랙아웃: 이틀 전 실적 발표 — 컨센서스 아직 미반영
scenario("BLACKOUT",
         eps_trend=mk_trend(2.00, 2.00),
         eps_revisions=mk_revisions(2, 2),
         quarterly_income_stmt=mk_income([0.30, 0.31, 0.32, 0.33]),
         earnings_dates=mk_earnings(-2))

# 5) 데이터 없음 — '문제 없음'으로 통과시키면 안 된다
scenario("NO_DATA",
         eps_trend=RuntimeError("no data"),
         eps_revisions=RuntimeError("no data"),
         quarterly_income_stmt=RuntimeError("no data"),
         earnings_dates=RuntimeError("no data"),
         institutional_holders=RuntimeError("no data"))

# 6) 마진만 살짝 하락 — 감점 1개 → 조건부
scenario("SOFT_MARGIN",
         eps_trend=mk_trend(2.00, 2.00),
         eps_revisions=mk_revisions(3, 2),
         quarterly_income_stmt=mk_income([0.34, 0.33, 0.35, 0.32]),
         earnings_dates=mk_earnings(60))


yf.Ticker = lambda t: SCENARIOS[t]

print("=" * 74)
print("④ 악재 스크리닝 판정 테스트")
print("=" * 74)

expected = {
    "OPPORTUNITY": ("통과", "A"),
    "EPS_CRASH": ("차단", "C"),
    "INTEL_PATTERN": ("차단", "C"),
    "BLACKOUT": ("차단", "C"),
    "NO_DATA": ("확인불가", "?"),
    "SOFT_MARGIN": ("조건부", "B"),
}

for name, (want_v, want_g) in expected.items():
    r = F.analyze(name)
    print(f"\n▸ {name}")
    for line in r["reasons"]:
        print(f"    · {line}")
    if r["blocks"]:
        print(f"    차단 사유: {', '.join(r['blocks'])}")
    if r["penalties"]:
        print(f"    감점 사유: {', '.join(r['penalties'])}")
    check(f"  판정", r["verdict"], want_v)
    check(f"  등급", r["grade"], want_g)

# 세부 수치 검증
r = F.analyze("OPPORTUNITY")
check("EPS 개정률 계산 (2.05/2.08-1)", round(r["eps"]["revision"], 4), -0.0144)
check("기관 보유 비중 파싱", r["inst"]["institutions_pct"], 82.0)
check("GPM 최신값 (0.33)", r["gpm"]["latest"], 0.33)

r = F.analyze("INTEL_PATTERN")
check("GPM 연속 하락 카운트", r["gpm"]["consecutive_declines"], 3)
check("GPM 변화폭 (0.33-0.42)", r["gpm"]["change"], -0.09)

r = F.analyze("BLACKOUT")
check("블랙아웃 감지", r["earnings"]["blackout"], True)

print("\n" + "=" * 74)
print(f"결과: {sum(results)}/{len(results)} 통과")
print("=" * 74)
sys.exit(0 if all(results) else 1)
