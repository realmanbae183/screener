"""
시세 데이터 레이어: yfinance 다운로드 + SQLite 캐시.
과거 10년치는 한 번만 받고, 이후엔 새로 생긴 봉만 증분 갱신한다.
(차트 이미지를 매번 가져오는 게 아니라 OHLCV 숫자만 쌓는다)
"""
import os
import sqlite3
import time
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

import config as C

SCHEMA = """
CREATE TABLE IF NOT EXISTS prices (
    ticker TEXT NOT NULL,
    date   TEXT NOT NULL,
    open   REAL, high REAL, low REAL, close REAL, volume REAL,
    PRIMARY KEY (ticker, date)
);
CREATE INDEX IF NOT EXISTS idx_prices_ticker ON prices(ticker);
CREATE TABLE IF NOT EXISTS meta (
    ticker TEXT PRIMARY KEY,
    sector TEXT, industry TEXT, market_cap REAL, updated TEXT
);
"""


def connect(path: str = C.DB_PATH) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    return conn


def _flatten(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """yfinance가 MultiIndex 컬럼을 줄 때가 있어 단일 종목 기준으로 평탄화."""
    if isinstance(df.columns, pd.MultiIndex):
        lvl0 = df.columns.get_level_values(0)
        if ticker in df.columns.get_level_values(-1):
            df = df.xs(ticker, axis=1, level=-1)
        else:
            df.columns = lvl0
    return df


def last_date(conn, ticker: str) -> str | None:
    r = conn.execute("SELECT MAX(date) FROM prices WHERE ticker=?", (ticker,)).fetchone()
    return r[0] if r and r[0] else None


def store(conn, ticker: str, df: pd.DataFrame) -> int:
    if df is None or df.empty:
        return 0
    df = _flatten(df, ticker)
    need = {"Open", "High", "Low", "Close", "Volume"}
    if not need.issubset(set(df.columns)):
        return 0
    rows = [
        (ticker, idx.strftime("%Y-%m-%d"),
         float(r["Open"]), float(r["High"]), float(r["Low"]),
         float(r["Close"]), float(r["Volume"]))
        for idx, r in df.iterrows()
        if pd.notna(r["Close"]) and pd.notna(r["Volume"])
    ]
    conn.executemany(
        "INSERT OR REPLACE INTO prices VALUES (?,?,?,?,?,?,?)", rows)
    conn.commit()
    return len(rows)


def load(conn, ticker: str) -> pd.DataFrame:
    """SQLite에서 OHLCV를 지표 엔진이 기대하는 형태로 읽어온다."""
    df = pd.read_sql_query(
        "SELECT date, open, high, low, close, volume FROM prices "
        "WHERE ticker=? ORDER BY date", conn, params=(ticker,))
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    df.columns = ["Open", "High", "Low", "Close", "Volume"]
    return df


def bar_count(conn, ticker: str) -> int:
    r = conn.execute("SELECT COUNT(*) FROM prices WHERE ticker=?", (ticker,)).fetchone()
    return int(r[0]) if r else 0


def coverage(conn, tickers: list[str], min_bars: int = C.MIN_HISTORY_DAYS) -> dict:
    """DB에 실제로 쓸 만한 데이터가 있는 종목이 몇 개인지 집계한다."""
    have = dict(conn.execute(
        "SELECT ticker, COUNT(*) FROM prices GROUP BY ticker").fetchall())
    ok = [t for t in tickers if have.get(t, 0) >= min_bars]
    thin = [t for t in tickers if 0 < have.get(t, 0) < min_bars]
    missing = [t for t in tickers if have.get(t, 0) == 0]
    return {"ok": ok, "thin": thin, "missing": missing, "counts": have}


def _download_one(ticker: str, period: str) -> pd.DataFrame | None:
    """한 종목만 단독으로 받는다. 배치보다 느리지만 훨씬 안정적."""
    try:
        d = yf.Ticker(ticker).history(period=period, interval="1d",
                                      auto_adjust=True)
        if d is not None and not d.empty:
            return d
    except Exception:
        pass
    try:
        d = yf.download(ticker, period=period, interval="1d",
                        auto_adjust=True, progress=False, threads=False)
        if d is not None and not d.empty:
            return d
    except Exception:
        pass
    return None


def repair(tickers: list[str], conn=None, period: str = C.HISTORY_PERIOD,
           min_bars: int = C.MIN_HISTORY_DAYS, verbose: bool = True) -> dict:
    """
    비어 있거나 부실한 종목만 골라 한 종목씩 다시 받는다.
    배치 다운로드는 야후가 중간에 끊는 일이 잦아서, 빠진 것만 개별로 메운다.
    """
    close_after = conn is None
    conn = conn or connect()
    cov = coverage(conn, tickers, min_bars)
    todo = cov["missing"] + cov["thin"]

    if verbose:
        print(f"  보유 {len(cov['ok'])} / 부실 {len(cov['thin'])} / 없음 {len(cov['missing'])}")
        if not todo:
            print("  메울 것이 없습니다.")

    fixed, still = 0, []
    for rnd in range(1, C.RETRY_ROUNDS + 1):
        if not todo:
            break
        if verbose and todo:
            print(f"  [복구 {rnd}회차] {len(todo)}종목 개별 다운로드...")
        still = []
        for i, t in enumerate(todo, 1):
            d = _download_one(t, period)
            if d is not None and store(conn, t, d) >= min_bars:
                fixed += 1
            else:
                still.append(t)
            if verbose and i % 25 == 0:
                print(f"    {i}/{len(todo)} (복구 {fixed})")
            time.sleep(C.RETRY_DELAY if t in still else 0.15)
        todo = still
        if todo and rnd < C.RETRY_ROUNDS:
            time.sleep(3)

    if verbose:
        print(f"  복구 완료: +{fixed}종목, 남은 실패 {len(still)}종목")
        if still:
            print(f"    {', '.join(still[:20])}{' ...' if len(still) > 20 else ''}")
    if close_after:
        conn.close()
    return {"fixed": fixed, "failed": still}


def sync(tickers: list[str], conn=None, period: str = C.HISTORY_PERIOD,
         batch: int = C.BATCH_SIZE, verbose: bool = True) -> dict:
    """
    전체 동기화. 이미 받아둔 종목은 마지막 날짜 이후만 증분으로 가져온다.
    최초 1회는 오래 걸리고, 이후 일일 갱신은 수십 초면 끝난다.
    """
    close_after = conn is None
    conn = conn or connect()
    stats = {"new": 0, "updated": 0, "failed": []}

    fresh, incremental = [], []
    for t in tickers:
        ld = last_date(conn, t)
        if ld is None:
            fresh.append(t)
        elif datetime.strptime(ld, "%Y-%m-%d").date() < (
                datetime.now().date() - timedelta(days=1)):
            incremental.append((t, ld))

    # 신규 종목: 전체 히스토리 (작은 배치 + 배치 간 휴식)
    for i in range(0, len(fresh), batch):
        chunk = fresh[i:i + batch]
        if verbose:
            print(f"  [신규] {i+1}-{i+len(chunk)}/{len(fresh)} 다운로드 중...")
        try:
            data = yf.download(chunk, period=period, interval="1d",
                               group_by="ticker", auto_adjust=True,
                               progress=False, threads=True)
        except Exception as e:
            stats["failed"] += chunk
            print(f"    배치 실패: {e}")
            time.sleep(C.BATCH_DELAY * 3)
            continue
        for t in chunk:
            try:
                sub = data[t] if isinstance(data.columns, pd.MultiIndex) else data
                n = store(conn, t, sub.dropna(how="all"))
                stats["new"] += 1 if n else 0
                if not n:
                    stats["failed"].append(t)
            except Exception:
                stats["failed"].append(t)
        time.sleep(C.BATCH_DELAY)

    # 기존 종목: 증분
    for t, ld in incremental:
        start = (datetime.strptime(ld, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
        try:
            d = yf.download(t, start=start, interval="1d", auto_adjust=True,
                            progress=False)
            if store(conn, t, d):
                stats["updated"] += 1
        except Exception:
            stats["failed"].append(t)

    if verbose:
        print(f"  동기화 완료 — 신규 {stats['new']}, 갱신 {stats['updated']}, "
              f"실패 {len(stats['failed'])}")

    # 배치에서 빠진 것들을 개별로 메운다 (야후가 끊은 종목 회수)
    rep = repair(tickers, conn=conn, period=period, verbose=verbose)
    stats["repaired"] = rep["fixed"]
    stats["failed"] = rep["failed"]

    if close_after:
        conn.close()
    return stats


def fetch_meta(tickers: list[str], conn=None, verbose: bool = True) -> int:
    """섹터/산업/시총 — ①유니버스 필터에 쓰인다. 느리므로 주 1회면 충분."""
    close_after = conn is None
    conn = conn or connect()
    ok = 0
    for i, t in enumerate(tickers, 1):
        try:
            info = yf.Ticker(t).get_info()
            conn.execute(
                "INSERT OR REPLACE INTO meta VALUES (?,?,?,?,?)",
                (t, info.get("sector"), info.get("industry"),
                 info.get("marketCap"), datetime.now().strftime("%Y-%m-%d")))
            ok += 1
        except Exception:
            pass
        if verbose and i % 50 == 0:
            print(f"  메타데이터 {i}/{len(tickers)}")
            conn.commit()
    conn.commit()
    if close_after:
        conn.close()
    return ok


def load_meta(conn) -> pd.DataFrame:
    return pd.read_sql_query("SELECT * FROM meta", conn)
