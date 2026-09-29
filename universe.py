"""
유니버스 수집: S&P 500 + 다우 30 (+ 가능하면 나스닥 100).

수집 순서 (앞이 실패하면 다음으로):
  1) 패키지에 동봉된 data/universe.csv  ← 항상 성공. 섹터 정보 포함
  2) 위키피디아 (User-Agent 붙여서)     ← 최신 편입/편출 반영, 나스닥100 추가

위키피디아는 신분증(User-Agent) 없는 프로그램 접속을 403으로 막기 때문에
반드시 헤더를 붙여야 한다.
"""
import io
import os

import pandas as pd

BUNDLED = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "data", "universe.csv")

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/125.0 Safari/537.36")

WIKI = {
    "SP500": ("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", "Symbol"),
    "NDX100": ("https://en.wikipedia.org/wiki/Nasdaq-100", "Ticker"),
    "DOW30": ("https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average", "Symbol"),
}


def _wiki_tickers(url: str, col: str) -> list[str]:
    """위키피디아 표에서 티커 컬럼만 뽑는다. User-Agent 필수."""
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    html = urllib.request.urlopen(req, timeout=30).read().decode("utf-8", "ignore")
    for t in pd.read_html(io.StringIO(html)):
        if col in t.columns:
            vals = [str(s).strip().upper() for s in t[col].dropna()]
            # 티커처럼 생긴 것만 (설명 문장이 섞여 들어오는 표가 있음)
            vals = [v for v in vals if 1 <= len(v) <= 6 and v.replace(".", "").replace("-", "").isalpha()]
            if len(vals) >= 20:
                return vals
    raise ValueError(f"'{col}' 컬럼을 가진 표를 찾지 못함")


def build_universe(verbose: bool = True) -> pd.DataFrame:
    # 1) 동봉 CSV — 이게 기준선이고, 없으면 진행 불가
    if not os.path.exists(BUNDLED):
        raise RuntimeError(f"동봉 목록이 없습니다: {BUNDLED}")
    base = pd.read_csv(BUNDLED)
    base["ticker"] = base["ticker"].astype(str).str.upper()
    if verbose:
        print(f"  동봉 목록: {len(base)}종목 (S&P500 + 다우30)")

    # 2) 위키피디아로 보강 — 실패해도 그냥 넘어간다
    known = set(base["ticker"])
    added, failed = 0, []
    for name, (url, col) in WIKI.items():
        try:
            for t in _wiki_tickers(url, col):
                t = t.replace(".", "-")
                if t not in known:
                    known.add(t)
                    base = pd.concat([base, pd.DataFrame([{
                        "ticker": t, "indices": name, "sector": "", "industry": ""}])],
                        ignore_index=True)
                    added += 1
            if verbose:
                print(f"  {name}: 위키피디아 확인 완료")
        except Exception as e:
            failed.append(name)
            if verbose:
                print(f"  {name}: 위키피디아 건너뜀 ({type(e).__name__}) — 동봉 목록으로 진행")

    base = base.drop_duplicates("ticker").sort_values("ticker").reset_index(drop=True)
    if verbose:
        print(f"  최종 유니버스: {len(base)}종목 (위키피디아로 +{added})")
    return base


if __name__ == "__main__":
    u = build_universe()
    print()
    print(u.head(10).to_string(index=False))
