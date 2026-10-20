"""lab/replay.py — 최근 며칠 동안 개장 돌파 규칙(v2)을 실제 5분봉으로 따라가 본 결과 (1계약 기준 손익)"""
import json, os
import pandas as pd, numpy as np, yfinance as yf
df = yf.download("NQ=F", period="10d", interval="5m", auto_adjust=False, progress=False, prepost=True)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)
df.index = pd.to_datetime(df.index, utc=True).tz_convert("America/New_York")
df = df[["Open", "High", "Low", "Close"]].dropna()
PT, FEE = 2.0, 2.0
out = []
days = sorted(set(d for d in df.index.date if pd.Timestamp(d).dayofweek < 5))
for d in days:
    day = df[df.index.date == d]
    orb = day.between_time("09:30", "09:59")
    if len(orb) < 5:
        continue
    oh, ol = float(orb["High"].max()), float(orb["Low"].min())
    rec = {"날짜": str(d), "개장최고": oh, "개장최저": ol, "매수": None}
    for hh in (11, 12, 13, 14, 15):
        t = pd.Timestamp(f"{d} {hh}:00", tz="America/New_York")
        prev = day[day.index < t]
        if not len(prev):
            continue
        px = float(prev["Close"].iloc[-1])
        if px > oh:
            rec["매수"] = {"시각": f"{hh}:00", "가격": px}
            break
    if rec["매수"]:
        bt = pd.Timestamp(f"{d} {rec['매수']['시각']}", tz="America/New_York")
        entry = rec["매수"]["가격"]
        stop, tgt = entry * 0.985, entry * 1.02
        # 다음 평일 10:00
        nd = pd.Timestamp(d) + pd.Timedelta(days=1)
        while nd.dayofweek >= 5:
            nd += pd.Timedelta(days=1)
        end = pd.Timestamp(f"{nd.date()} 10:00", tz="America/New_York")
        if pd.Timestamp(d).dayofweek == 4:
            end = pd.Timestamp(f"{d} 15:50", tz="America/New_York")
        path = df[(df.index >= bt) & (df.index < end)]
        exitp, why, et = None, None, None
        for ts, r in path.iterrows():
            if r["Low"] <= stop:
                exitp, why, et = stop, "손절 -1.5%", ts; break
            if r["High"] >= tgt:
                exitp, why, et = tgt, "익절 +2%", ts; break
            if ts.date() == d and ts.minute == 55 and ts.hour in (11, 12, 13, 14) and r["Close"] < ol:
                exitp, why, et = float(r["Close"]), "개장 저점 이탈", ts; break
        if exitp is None:
            if len(path) and path.index[-1] >= end - pd.Timedelta(minutes=10):
                exitp, why, et = float(path["Close"].iloc[-1]), "다음날 10시 정리", path.index[-1]
            else:
                exitp, why, et = float(path["Close"].iloc[-1]) if len(path) else entry, "아직 보유 중(현재가 기준)", path.index[-1] if len(path) else bt
        pnl = (exitp - entry) * PT - FEE
        rec["매도"] = {"시각": str(et), "가격": round(exitp, 2), "이유": why}
        rec["1계약손익$"] = round(pnl, 1)
    d_close = float(day.between_time("09:30", "16:00")["Close"].iloc[-1]) if len(day.between_time("09:30", "16:00")) else None
    d_open = float(orb["Open"].iloc[0])
    rec["그날 정규장 시가→종가(포인트)"] = round(d_close - d_open, 2) if d_close else None
    out.append(rec)
os.makedirs("lab/results", exist_ok=True)
json.dump(out, open("lab/results/replay.json", "w"), ensure_ascii=False, indent=1, default=str)
for r in out: print(r)
