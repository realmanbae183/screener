"""
합성 데이터로 지표 로직 검증.
실시세 없이도 각 함수가 의도대로 동작하는지 확인한다.
"""
import numpy as np
import pandas as pd
import sys

sys.path.insert(0, "/home/claude/trading-bot")
import indicators as I
import config as C


def make_ohlcv(closes, volumes=None, seed=0):
    rng = np.random.default_rng(seed)
    closes = np.asarray(closes, dtype=float)
    n = len(closes)
    noise = rng.uniform(0.002, 0.012, n)
    high = closes * (1 + noise)
    low = closes * (1 - noise)
    op = closes * (1 + rng.uniform(-0.004, 0.004, n))
    vol = np.full(n, 1_000_000.0) if volumes is None else np.asarray(volumes, float)
    idx = pd.bdate_range("2016-01-01", periods=n)
    return pd.DataFrame({"Open": op, "High": high, "Low": low,
                         "Close": closes, "Volume": vol}, index=idx)


results = []


def check(name, got, want):
    ok = got == want
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL':4} | {name:46} got={got!r:>12} want={want!r}")


# ── 1. 상승 추세 종목: 추세 필터 통과해야 함
n = 600
up = 100 * np.exp(np.linspace(0, 0.55, n))
df_up = make_ohlcv(up)
ok, info = I.trend_ok(df_up)
check("상승추세 → 추세필터 통과", ok, True)
check("  200일선 우상향 감지", info["ma200_rising"], True)

# ── 2. 지속 하락 종목: 추세 필터에서 차단돼야 함 (떨어지는 칼날)
down = 200 * np.exp(np.linspace(0, -0.9, n))
df_down = make_ohlcv(down)
ok, info = I.trend_ok(df_down)
check("지속하락 → 추세필터 차단", ok, False)

# ── 3. 상승추세 중 급락(눌림목): 추세 통과 + 과매도 점수 확보
#    실제 급락은 완만한 램프가 아니라 앞쪽에 집중된다(투매 → 진정).
base = 100 * np.exp(np.linspace(0, 1.3, n - 30))
peak = base[-1]
sharp = peak * np.array([0.97, 0.92, 0.86, 0.80, 0.76, 0.735, 0.72, 0.715, 0.71, 0.705,
                         0.702, 0.70, 0.699, 0.701, 0.700, 0.702, 0.703, 0.701, 0.704,
                         0.702, 0.705, 0.703, 0.706, 0.704, 0.707, 0.705, 0.708, 0.706,
                         0.709, 0.707])
closes = np.concatenate([base, sharp])
vols = np.full(n, 1_000_000.0)
vols[n - 30:n - 20] = 4_500_000.0                 # 투매 클라이맥스 10일
vols[n - 20:] = 900_000.0                         # 이후 거래량 소진
df_dip = make_ohlcv(closes, vols, seed=3)

ok, _ = I.trend_ok(df_dip)
check("상승추세 중 눌림목 → 추세필터 통과", ok, True)

res = I.oversold_score(df_dip)
d = res["detail"]
print(f"\n  [눌림목 케이스] 스코어 {res['score']}/{res['max_score']}")
for k, v in d.items():
    val = v["value"] if not isinstance(v["value"], dict) else "…"
    print(f"    {k:20} hit={str(v['hit']):5} score={v['score']} value={val}")

check("  RSI 30 미만 감지", d["rsi"]["hit"], True)
check("  볼린저 하단 이탈 감지", d["bollinger_pct_b"]["hit"], True)
check("  ATR 이격도 2.5 이상 감지", d["atr_distance"]["hit"], True)
check("  52주 낙폭 25% 이상 감지", d["drawdown_52w"]["hit"], True)
check("  거래량 클라이맥스 감지", d["volume_climax"]["hit"], True)
check("  스코어 임계값 통과", res["score"] >= C.SCORE_THRESHOLD, True)

# 부분점수: 경계 근처 값도 점수를 받아야 한다 (O/X면 0점)
if C.PARTIAL_CREDIT:
    near = I._partial(30.1, C.RSI_FULL, C.RSI_ZERO, C.RSI_SCORE)
    check("  RSI 30.1도 부분점수 획득", near > 0, True)
    check("  RSI 45 이상은 0점", I._partial(46.0, C.RSI_FULL, C.RSI_ZERO,
                                          C.RSI_SCORE) == 0, True)
    check("  만점 기준 초과해도 만점 유지",
          I._partial(9.0, C.ATR_FULL, C.ATR_ZERO, C.ATR_SCORE) == C.ATR_SCORE, True)

# ── 4. 평온한 횡보장: 과매도 신호 없어야 함 (오탐 방지)
flat = 100 + np.sin(np.linspace(0, 20, n)) * 1.2
df_flat = make_ohlcv(flat, seed=7)
res_flat = I.oversold_score(df_flat)
check("횡보장 → 과매도 미발동", res_flat["score"] < C.SCORE_THRESHOLD, True)
print(f"  (횡보장 스코어: {res_flat['score']}/{res_flat['max_score']})")

# ── 5. 냉각 기간: 어제 -15% 급락 → 아직 평가 금지
crash = np.concatenate([up[:-1], [up[-2] * 0.85]])
df_crash = make_ohlcv(crash, seed=11)
ok, info = I.cooling_ok(df_crash)
check("급락 직후 → 냉각기간에 걸림", ok, False)
check("  남은 대기일 계산", info["wait_more_days"] > 0, True)

ok, _ = I.cooling_ok(df_up)
check("급락 없음 → 냉각기간 통과", ok, True)

# ── 6. MACD 강세 다이버전스: 가격 신저점 + 모멘텀 개선
#    평탄한 베이스 대신 실제처럼 흔들리는 베이스를 쓴다.
rng6 = np.random.default_rng(21)
base6 = 120 + rng6.normal(0, 1.0, 520).cumsum() * 0.05
seg1 = np.linspace(base6[-1], 92, 40)     # 1차 급락
seg2 = np.linspace(92.6, 106, 25)         # 반등
seg3 = np.linspace(105.4, 88, 30)         # 2차 하락: 더 낮은 저점, 기울기는 완만
seg4 = np.linspace(88.4, 94, 12)          # 저점 확인 후 반등 시작
div_closes = np.concatenate([base6, seg1, seg2, seg3, seg4])
df_div = make_ohlcv(div_closes, seed=13)
hit, dinfo = I.macd_bullish_divergence(df_div)
print(f"\n  [다이버전스] hit={hit} info={dinfo}")
check("가격 신저점 + MACD 개선 → 다이버전스 감지", hit, True)

# ── 7. RSI 경계값 수치 검증 (단조 상승 시 RSI는 100에 수렴)
mono_up = pd.Series(np.linspace(100, 200, 100))
check("단조 상승 시 RSI > 95", float(I.rsi(mono_up).iloc[-1]) > 95, True)
mono_dn = pd.Series(np.linspace(200, 100, 100))
check("단조 하락 시 RSI < 5", float(I.rsi(mono_dn).iloc[-1]) < 5, True)

# ── 8. ATR 정규화: 같은 -30%라도 변동성 큰 종목은 이격도가 작아야 함
calm = np.concatenate([np.full(300, 100.0), np.linspace(100, 70, 30)])
volatile_base = 100 + np.random.default_rng(5).normal(0, 6, 300).cumsum() * 0.05
volatile = np.concatenate([np.abs(volatile_base) + 80, np.linspace(
    float(np.abs(volatile_base[-1]) + 80), float(np.abs(volatile_base[-1]) + 80) * 0.7, 30)])
d_calm = I.atr_distance(make_ohlcv(calm, seed=1))
d_vol = I.atr_distance(make_ohlcv(volatile, seed=2))
print(f"\n  [ATR 정규화] 저변동성 종목 이격도={d_calm:.2f}, 고변동성 종목 이격도={d_vol:.2f}")
check("같은 -30%라도 저변동성 종목 이격도가 더 큼", d_calm > d_vol, True)

print("\n" + "=" * 70)
print(f"결과: {sum(results)}/{len(results)} 통과")
print("=" * 70)
sys.exit(0 if all(results) else 1)
