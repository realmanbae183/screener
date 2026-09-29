"""
알림 로직 검증 — 실제로 전송하지 않고 변화 감지만 확인한다.
핵심: 아무 일 없는 날엔 알림이 0건이어야 한다.
"""
import os
import sys
import tempfile

import pandas as pd

import notify as N

results = []


def check(name, got, want):
    ok = got == want
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL':4} | {name:48} got={got!r:<14} want={want!r}")


def cand(ticker, price, score=12.0, dd=0.35, name=None):
    return {"ticker": ticker, "name": name or f"{ticker} Corp",
            "price": price, "score": score, "dd_52w": dd}


def ladder(px, levels):
    rungs = [{"step": 1, "price": px, "why": "현재가", "weight": 0.4,
              "attractiveness": 70, "expected_return": 0.5, "drop_from_now": 0.0}]
    for i, lv in enumerate(levels, start=2):
        rungs.append({"step": i, "price": lv, "why": "지지선", "weight": 0.3,
                      "attractiveness": 75 + i, "expected_return": 0.6,
                      "drop_from_now": lv / px - 1})
    return {"rungs": rungs, "invalidation": round(min(levels) * 0.88, 2),
            "current_price": px}


DEEP = {"ALB": {"grade": "A", "fin_score": 79.0, "fin_grade": "B",
                "blocks": [], "reasons": ["EPS 추정치 유지 — 밸류에이션만 조정"]}}

tmp = tempfile.mkdtemp()
SP = os.path.join(tmp, "state.json")

print("=" * 76)
print("1일차 — 처음 돌린 날 (기준점만 잡고 조용해야 함)")
print("=" * 76)
c1 = pd.DataFrame([cand("ALB", 107.39), cand("CIEN", 356.91)])
l1 = {"ALB": ladder(107.39, [104.66, 98.57]), "CIEN": ladder(356.91, [340.0, 320.0])}
r = N.notify(c1, DEEP, l1, state_path=SP, dry_run=False)
check("첫 실행 → 알림 0건", r["events"], 0)
check("  기준점으로 2종목 저장", r.get("baseline"), 2)
check("  첫 실행 표시", r.get("first_run"), True)

print("\n" + "=" * 76)
print("1일차 직후 — 새 종목 하나가 추가되면 그건 알림")
print("=" * 76)
c1b = pd.DataFrame([cand("ALB", 107.39), cand("CIEN", 356.91), cand("PCG", 12.34)])
l1b = dict(l1, PCG=ladder(12.34, [11.8, 11.0]))
r = N.notify(c1b, DEEP, l1b, state_path=SP, dry_run=False)
check("신규 1종목 → 알림 1건", r["events"], 1)
check("  종류가 신규", r["kinds"], ["new"])
# 이후 시나리오를 위해 PCG는 목록에서 빼고 기준을 c1 상태로 되돌린다
N.notify(c1, DEEP, l1, state_path=SP, dry_run=False)

print("\n" + "=" * 76)
print("2일차 — 가격 거의 안 움직임 (알림이 오면 안 되는 날)")
print("=" * 76)
c2 = pd.DataFrame([cand("ALB", 107.10), cand("CIEN", 355.00)])
r = N.notify(c2, DEEP, l1, state_path=SP, dry_run=False)
check("변화 없음 → 알림 0건", r["events"], 0)

print("\n" + "=" * 76)
print("3일차 — ALB가 2차 진입가($104.66) 아래로 내려옴")
print("=" * 76)
c3 = pd.DataFrame([cand("ALB", 103.20), cand("CIEN", 354.00)])
r = N.notify(c3, DEEP, l1, state_path=SP, dry_run=False)
check("진입가 도달 → 알림 1건", r["events"], 1)
check("  종류가 rung", r["kinds"], ["rung"])

print("\n" + "=" * 76)
print("4일차 — 같은 가격대 유지 (이미 알린 칸은 다시 안 알림)")
print("=" * 76)
c4 = pd.DataFrame([cand("ALB", 103.00), cand("CIEN", 353.00)])
r = N.notify(c4, DEEP, l1, state_path=SP, dry_run=False)
check("같은 칸 재알림 없음", r["events"], 0)

print("\n" + "=" * 76)
print("5일차 — 폐기선까지 붕괴 + CIEN 후보 이탈")
print("=" * 76)
inv = l1["ALB"]["invalidation"]
c5 = pd.DataFrame([cand("ALB", inv - 2)])
r = N.notify(c5, DEEP, l1, state_path=SP, dry_run=False)
check("폐기선 이탈 + 3차 도달 + 이탈 감지", sorted(set(r["kinds"])),
      ["dropped", "invalid", "rung"])
check("  폐기선 알림이 맨 앞", r["kinds"][0], "invalid")

print("\n" + "=" * 76)
print("메시지 미리보기")
print("=" * 76)
st = N.load_state(SP)
ev, _ = N.diff_events(pd.DataFrame([cand("ALB", 98.00)]), DEEP,
                      {"ALB": ladder(107.39, [104.66, 98.57])}, {"tickers": {}})
print(N.format_event(ev[0], "https://example.github.io/screener")
      .replace("<b>", "").replace("</b>", "").replace("<i>", "").replace("</i>", ""))

check("설정 없으면 전송은 건너뛰고 False", N.send("x", token="", chat_id=""), False)

print("\n" + "=" * 76)
print(f"결과: {sum(results)}/{len(results)} 통과")
print("=" * 76)
sys.exit(0 if all(results) else 1)
