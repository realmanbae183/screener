"""
⑤ 13F 파싱·매칭 검증.
SEC에 실제로 붙지 않고, 실제 파일과 같은 형태의 XML로 파서를 확인한다.
네임스페이스는 제출자·연도마다 다르므로 여러 변형을 함께 시험한다.
"""
import sys

import numpy as np
import pandas as pd

import thirteenf as F

results = []


def check(name, got, want):
    ok = got == want
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL':4} | {name:46} got={got!r:<16} want={want!r}")


# ── 실제 13F INFORMATION TABLE 형태 (네임스페이스 있음)
NS_XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<informationTable xmlns="http://www.sec.gov/edgar/document/thirteenf/informationtable">
 <infoTable>
  <nameOfIssuer>ALBEMARLE CORP</nameOfIssuer>
  <titleOfClass>COM</titleOfClass>
  <cusip>012653101</cusip>
  <value>184500</value>
  <shrsOrPrnAmt><sshPrnamt>1250000</sshPrnamt><sshPrnamtType>SH</sshPrnamtType></shrsOrPrnAmt>
  <investmentDiscretion>SOLE</investmentDiscretion>
  <votingAuthority><Sole>1250000</Sole><Shared>0</Shared><None>0</None></votingAuthority>
 </infoTable>
 <infoTable>
  <nameOfIssuer>ALBEMARLE CORP</nameOfIssuer>
  <titleOfClass>COM</titleOfClass>
  <cusip>012653101</cusip>
  <value>15500</value>
  <shrsOrPrnAmt><sshPrnamt>105000</sshPrnamt><sshPrnamtType>SH</sshPrnamtType></shrsOrPrnAmt>
 </infoTable>
 <infoTable>
  <nameOfIssuer>APPLE INC</nameOfIssuer>
  <titleOfClass>COM</titleOfClass>
  <cusip>037833100</cusip>
  <value>9200000</value>
  <shrsOrPrnAmt><sshPrnamt>40000000</sshPrnamt><sshPrnamtType>SH</sshPrnamtType></shrsOrPrnAmt>
 </infoTable>
</informationTable>"""

# ── 접두사 네임스페이스 변형 (ns1: 같은 형태로 오는 제출자도 있다)
PREFIX_XML = b"""<?xml version="1.0"?>
<ns1:informationTable xmlns:ns1="http://www.sec.gov/edgar/document/thirteenf/informationtable">
 <ns1:infoTable>
  <ns1:nameOfIssuer>CIENA CORP</ns1:nameOfIssuer>
  <ns1:cusip>171779101</ns1:cusip>
  <ns1:value>52000</ns1:value>
  <ns1:shrsOrPrnAmt><ns1:sshPrnamt>310,000</ns1:sshPrnamt></ns1:shrsOrPrnAmt>
 </ns1:infoTable>
</ns1:informationTable>"""

# ── 네임스페이스 없는 변형
PLAIN_XML = b"""<?xml version="1.0"?>
<informationTable>
 <infoTable>
  <nameOfIssuer>PG&amp;E CORP</nameOfIssuer>
  <cusip>69331C108</cusip>
  <value>31000</value>
  <shrsOrPrnAmt><sshPrnamt>2400000</sshPrnamt></shrsOrPrnAmt>
 </infoTable>
</informationTable>"""

print("=" * 76)
print("13F INFORMATION TABLE 파싱")
print("=" * 76)

t1 = F.parse_info_table(NS_XML)
print(t1.to_string(index=False), "\n")
check("네임스페이스 XML — 종목 수 (중복 합산)", len(t1), 2)
alb = t1[t1.issuer_key == "albemarle"]
check("  같은 종목 두 줄 합산 (1250000+105000)",
      int(alb["shares"].iloc[0]), 1355000)
check("  금액도 합산 (184500+15500)", int(alb["value"].iloc[0]), 200000)

t2 = F.parse_info_table(PREFIX_XML)
check("접두사 네임스페이스 파싱", len(t2), 1)
check("  쉼표 포함 숫자 파싱 (310,000)", int(t2["shares"].iloc[0]), 310000)

t3 = F.parse_info_table(PLAIN_XML)
check("네임스페이스 없는 XML 파싱", len(t3), 1)
check("  앰퍼샌드 이스케이프 처리", t3["issuer"].iloc[0], "PG&E CORP")

check("깨진 XML은 빈 표 반환", F.parse_info_table(b"<not xml").empty, True)

# ── 회사명 정규화 (CUSIP→티커 매핑이 무료로 없어서 이름으로 맞춘다)
print("\n" + "=" * 76)
print("회사명 정규화 — 13F 표기와 우리 유니버스 표기를 맞추기")
print("=" * 76)
PAIRS = [
    ("ALBEMARLE CORP", "Albemarle Corporation", True),
    ("CIENA CORP", "Ciena", True),
    ("PG&E CORP", "PG&E Corporation", True),
    ("OLD DOMINION FREIGHT LINE INC", "Old Dominion", False),
    ("APPLE INC", "Apple Inc.", True),
    ("EDISON INTL", "Edison International", True),
    ("SYNCHRONY FINL", "Synchrony Financial", True),
    ("GENERAC HLDGS INC", "Generac", False),
    ("ATMOS ENERGY CORP", "Atmos Energy", True),
    ("MICROSOFT CORP", "Apple Inc.", False),
]
for a, b, want_same in PAIRS:
    ka, kb = F.normalize_name(a), F.normalize_name(b)
    same = ka == kb
    mark = "일치" if same else ("부분" if (ka in kb or kb in ka) else "불일치")
    print(f"  {a:<32} → {ka:<24} vs {kb:<22} {mark}")
    if want_same:
        results.append(same)

check("서로 다른 회사는 불일치",
      F.normalize_name("MICROSOFT CORP") == F.normalize_name("Apple Inc."), False)

# ── 분기 VWAP으로 추정 평단
print("\n" + "=" * 76)
print("분기 VWAP 추정 평단 (13F는 평단을 공시하지 않는다)")
print("=" * 76)
idx = pd.bdate_range("2026-04-01", "2026-06-30")
n = len(idx)
close = np.linspace(200, 150, n)
vol = np.full(n, 1e6)
vol[-20:] = 5e6                       # 분기 말에 거래가 몰린 경우
px = pd.DataFrame({"Open": close, "High": close * 1.01, "Low": close * 0.99,
                   "Close": close, "Volume": vol}, index=idx)
v = F.quarter_vwap(px, "2026-06-30")
simple = float(close.mean())
print(f"  단순 평균 ${simple:.2f} / 거래량 가중 ${v:.2f}")
check("VWAP이 거래 몰린 쪽으로 당겨짐", v < simple, True)
check("구간 안의 값인가", 150 <= v <= 200, True)
check("데이터 없으면 None", F.quarter_vwap(pd.DataFrame(), "2026-06-30"), None)

# ── 매칭 + 구간 판정
print("\n" + "=" * 76)
print("매칭과 구간 판정")
print("=" * 76)
agg = {
    "albemarle": {"issuer": "ALBEMARLE CORP", "holders": ["Baupost", "Appaloosa"],
                  "holder_count": 2, "added": ["Baupost"], "reduced": [],
                  "new": ["Appaloosa"], "net": 2, "report_date": "2026-06-30",
                  "total_value": 200000},
}
idx2 = pd.bdate_range("2026-04-01", "2026-09-28")
c2 = np.concatenate([np.linspace(180, 160, 60),
                     np.linspace(160, 107.39, len(idx2) - 60)])
px2 = pd.DataFrame({"Open": c2, "High": c2 * 1.01, "Low": c2 * 0.99,
                    "Close": c2, "Volume": np.full(len(idx2), 1e6)}, index=idx2)

m = F.match(agg, "ALB", "Albemarle Corporation", px2)
for k in ("holder_count", "added", "new", "net", "est_avg_price",
          "price_vs_est", "zone", "lag_days"):
    print(f"  {k:<16} {m[k]}")
check("이름으로 매칭 성공", m["available"], True)
check("  보유 기관 수", m["holder_count"], 2)
check("  순증 계산 (증가1+신규1-감소0)", m["net"], 2)
check("  추정 평단이 현재가보다 높음", m["est_avg_price"] > 107.39, True)
check("  '기관도 물림' 구간 판정", m["zone"], "기관도 물림")

m2 = F.match(agg, "XYZ", "Nonexistent Holdings", px2)
check("없는 종목은 available=False", m2["available"], False)

print("\n" + "=" * 76)
print(f"결과: {sum(results)}/{len(results)} 통과")
print("=" * 76)
sys.exit(0 if all(results) else 1)
