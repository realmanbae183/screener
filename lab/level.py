"""lab/level.py — 나스닥100 선물(NQ=F, 근월물) 최근 종가와 200일 이동평균 출력 (MNQ 대회 5번 규칙용 기준값)"""
import json, os
import yfinance as yf
import pandas as pd
df = yf.download("NQ=F", period="2y", interval="1d", auto_adjust=False, progress=False)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)
c = df["Close"].dropna()
ma = c.rolling(200).mean()
out = {"date": str(c.index[-1].date()), "close": round(float(c.iloc[-1]), 2), "ma200": round(float(ma.iloc[-1]), 2),
       "ma200_5d_ago": round(float(ma.iloc[-6]), 2), "ma200_20d_ago": round(float(ma.iloc[-21]), 2),
       "last5": [[str(i.date()), round(float(v), 2), round(float(m), 2)] for i, v, m in zip(c.index[-5:], c.iloc[-5:], ma.iloc[-5:])]}
os.makedirs("lab/results", exist_ok=True)
json.dump(out, open("lab/results/level.json", "w"), ensure_ascii=False)
print(out)
