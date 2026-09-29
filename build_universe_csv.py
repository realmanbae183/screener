"""동봉 유니버스 CSV 생성기 (회사명 포함). 개발용 — 평소엔 쓸 일 없음."""
import csv
import io
import urllib.request

URL = ("https://raw.githubusercontent.com/datasets/s-and-p-500-companies/"
       "main/data/constituents.csv")

DOW30 = {
    "MMM": "3M", "AXP": "American Express", "AMGN": "Amgen", "AMZN": "Amazon",
    "AAPL": "Apple", "BA": "Boeing", "CAT": "Caterpillar", "CVX": "Chevron",
    "CSCO": "Cisco", "KO": "Coca-Cola", "DIS": "Walt Disney",
    "GS": "Goldman Sachs", "HD": "Home Depot", "HON": "Honeywell", "IBM": "IBM",
    "JNJ": "Johnson & Johnson", "JPM": "JPMorgan Chase", "MCD": "McDonalds",
    "MRK": "Merck", "MSFT": "Microsoft", "NKE": "Nike", "NVDA": "NVIDIA",
    "PG": "Procter & Gamble", "CRM": "Salesforce", "SHW": "Sherwin-Williams",
    "TRV": "Travelers", "UNH": "UnitedHealth", "VZ": "Verizon", "V": "Visa",
    "WMT": "Walmart",
}

req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
rows = list(csv.DictReader(io.StringIO(
    urllib.request.urlopen(req, timeout=60).read().decode())))

uni = {}
for r in rows:
    t = r["Symbol"].strip().upper().replace(".", "-")
    uni[t] = {"ticker": t, "name": r["Security"].strip(), "indices": "SP500",
              "sector": r["GICS Sector"], "industry": r["GICS Sub-Industry"]}

for t, nm in DOW30.items():
    t = t.replace(".", "-")
    if t in uni:
        uni[t]["indices"] = "DOW30," + uni[t]["indices"]
    else:
        uni[t] = {"ticker": t, "name": nm, "indices": "DOW30",
                  "sector": "", "industry": ""}

out = sorted(uni.values(), key=lambda x: x["ticker"])
with open("data/universe.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["ticker", "name", "indices", "sector", "industry"])
    w.writeheader()
    w.writerows(out)

print(f"저장: {len(out)}종목")
for t in ["ALB", "ATO", "CIEN", "GNRC", "EIX", "PCG", "ODFL", "SYF", "PYPL"]:
    r = uni.get(t)
    if r:
        print(f"  {t:<6} {r['name']:<30} {r['sector']}")
