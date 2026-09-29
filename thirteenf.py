"""
⑤ 13F — 기관 보유 확인.

미국 기관투자자는 분기마다 보유 종목을 SEC에 신고한다(13F-HR).
그걸 읽어서 "이 종목을 주요 기관들이 늘리고 있나 줄이고 있나"를 본다.

현실적인 한계 (설계에 반영됨)
  · 최대 45일 지연 — 이미 팔았을 수 있다
  · 롱 포지션만 공시 — 헤지의 한쪽 다리일 수 있다
  · 평단은 공시되지 않는다 → 분기 VWAP로 추정한다
그래서 이 단계는 '진입 필터'이지 '홀딩 정당화'가 아니다.
이미 물린 종목을 합리화하는 데 쓰면 확증편향을 자동화하게 된다.

SEC 요구사항: User-Agent 헤더에 연락처를 넣어야 하고, 초당 10건 이하로 요청한다.
"""
import json
import re
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime

import numpy as np
import pandas as pd

import config as C

SEC = "https://data.sec.gov"
WWW = "https://www.sec.gov"


class SECError(RuntimeError):
    """SEC 응답 오류 — 상태 코드를 담아 원인을 알 수 있게 한다."""

    def __init__(self, url: str, status, detail: str = ""):
        self.url, self.status, self.detail = url, status, detail
        super().__init__(f"{status} — {url}" + (f" ({detail})" if detail else ""))


BROWSER_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36")

# SEC의 방화벽은 요청 모양을 보고 막는다. 어떤 조합이 통하는지는
# 망·지역에 따라 달라서, 되는 걸 찾을 때까지 순서대로 시도한다.
#
# ★ 순서가 중요하다. 실패하는 방식을 앞에 두면 요청마다 403을 한 번씩
#   맞고 넘어가는데, 기관 20곳이면 헛발질이 수십 번이라 SEC가 아예
#   차단할 수 있다. 실제로 통과한 '브라우저 위장'을 맨 앞에 둔다.
#   (SEC가 권장하는 '이름 이메일' UA를 자기네 방화벽이 봇으로 잡는다)
HEADER_PROFILES = [
    ("브라우저 위장", lambda: {
        "User-Agent": BROWSER_UA,
        "Accept": "application/json,text/html,application/xhtml+xml,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate",
        "Referer": "https://www.sec.gov/",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-site",
    }),
    ("SEC UA + 브라우저 헤더", lambda: {
        "User-Agent": C.SEC_USER_AGENT,
        "Accept": "application/json,text/html,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate",
        "Referer": "https://www.sec.gov/",
    }),
    ("SEC 권장", lambda: {
        "User-Agent": C.SEC_USER_AGENT,
        "Accept-Encoding": "gzip, deflate",
    }),
    ("SEC 권장 최소", lambda: {
        "User-Agent": C.SEC_USER_AGENT,
    }),
    ("완전 브라우저", lambda: {
        "User-Agent": BROWSER_UA,
        "Accept": ("text/html,application/xhtml+xml,application/xml;q=0.9,"
                   "image/avif,image/webp,*/*;q=0.8"),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Ch-Ua": '"Chromium";v="129", "Not=A?Brand";v="8"',
        "Sec-Ch-Ua-Mobile": "?0",
        "Sec-Ch-Ua-Platform": '"Windows"',
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
    }),
]

# ★ 통하는 방식을 '서버별로' 기억한다.
#   data.sec.gov 와 www.sec.gov 는 방화벽이 달라서 원하는 모양이 다르다.
#   (data 쪽은 브라우저 위장이 통하는데 www 쪽은 그걸 봇으로 잡는 식)
#   하나로 묶어두면 한쪽에서 찾은 방식을 다른 쪽에 계속 들이밀게 된다.
_working_profile: dict[str, int] = {}


def _host(url: str) -> str:
    return url.split("/")[2]


def _decompress(raw: bytes) -> bytes:
    if raw[:2] == b"\x1f\x8b":
        import gzip
        try:
            return gzip.decompress(raw)
        except Exception:
            return raw
    if raw[:1] == b"\x78":
        import zlib
        try:
            return zlib.decompress(raw)
        except Exception:
            return raw
    return raw


def _attempt(url: str, headers: dict, timeout: int) -> bytes:
    req = urllib.request.Request(
        url, headers={k: v for k, v in headers.items() if v is not None})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return _decompress(r.read())
    except urllib.error.HTTPError as e:
        body = ""
        try:
            # 에러 본문도 압축돼 오므로 풀어야 읽을 수 있다
            body = _decompress(e.read())[:300].decode("utf-8", "ignore")
            body = re.sub(r"<[^>]+>", " ", body)
            body = re.sub(r"\s+", " ", body).strip()
        except Exception:
            pass
        raise SECError(url, e.code, body) from None
    except urllib.error.URLError as e:
        raise SECError(url, "연결 실패", str(e.reason)) from None


def _get(url: str, timeout: int = 30, retry_403: bool = True) -> bytes:
    """
    되는 헤더 조합을 찾을 때까지 시도한다. 서버별로 따로 기억한다.
    403이 계속되면 속도 제한일 수 있어 한 번 쉬었다 다시 해본다.
    """
    host = _host(url)
    pref = _working_profile.get(host, 0)
    order = [pref] + [i for i in range(len(HEADER_PROFILES)) if i != pref]

    last = None
    for i in order:
        try:
            raw = _attempt(url, HEADER_PROFILES[i][1](), timeout)
            _working_profile[host] = i
            time.sleep(C.SEC_DELAY)
            return raw
        except SECError as e:
            last = e
            if e.status == "연결 실패":
                raise                  # 망 차단이면 다른 헤더도 소용없다
            time.sleep(0.4)

    # 전부 403이면 속도 제한일 수 있다. 한 번만 쉬었다 재시도.
    if retry_403 and last is not None and last.status == 403:
        time.sleep(C.SEC_RETRY_PAUSE)
        return _get(url, timeout, retry_403=False)
    raise last


def diagnose(verbose: bool = True) -> dict:
    """
    SEC 접속이 왜 막히는지 확인한다.
    전부 같은 이유로 실패하면 User-Agent 거부(403)일 가능성이 높다.
    """
    # 두 서버를 따로 본다. 방화벽이 달라서 한쪽만 뚫리는 경우가 실제로 있다.
    targets = [
        ("기관 목록 (data.sec.gov)",
         f"{SEC}/submissions/CIK0001067983.json"),
        ("보유 내역 (www.sec.gov)",
         f"{WWW}/Archives/edgar/data/1067983/index.json"),
    ]
    out = {"user_agent": C.SEC_USER_AGENT, "servers": []}
    if verbose:
        print(f"  UA: {C.SEC_USER_AGENT}\n")

    for label, url in targets:
        if verbose:
            print(f"  ── {label}")
        found, results = None, []
        for i, (name, make) in enumerate(HEADER_PROFILES):
            try:
                raw = _attempt(url, make(), 25)
                results.append({"profile": name, "status": 200, "bytes": len(raw)})
                if found is None:
                    found = i
                if verbose:
                    print(f"     ✅ {name:<20} 200 OK ({len(raw):,} bytes)")
                break                      # 하나 통하면 더 볼 필요 없다
            except SECError as e:
                results.append({"profile": name, "status": e.status,
                                "detail": e.detail})
                if verbose:
                    print(f"     ❌ {name:<20} {e.status}"
                          + (f"  {e.detail[:60]}" if e.detail else ""))
            time.sleep(0.4)

        if found is not None:
            _working_profile[_host(url)] = found
        out["servers"].append({"label": label, "url": url,
                               "ok": found is not None,
                               "profile": (HEADER_PROFILES[found][0]
                                           if found is not None else None),
                               "results": results})
        if verbose:
            print()

    ok = [s["ok"] for s in out["servers"]]
    if all(ok):
        names = " / ".join(f"{s['label'].split()[0]}: {s['profile']}"
                           for s in out["servers"])
        out["verdict"] = f"정상 — 두 서버 모두 접속됩니다 ({names})"
    elif not any(ok):
        out["verdict"] = ("두 서버 모두 차단 — 회선·지역 차단일 수 있습니다. "
                          "클라우드(GitHub Actions)에서는 보통 통과합니다.")
    else:
        blocked = next(s["label"] for s in out["servers"] if not s["ok"])
        out["verdict"] = (f"{blocked} 만 차단 — 13F는 보유 내역을 못 읽어 "
                          f"0건으로 나옵니다. 클라우드에서 돌리면 해결됩니다.")
    if verbose:
        print(f"  판정: {out['verdict']}")
    return out


def latest_13f_filings(cik: str, count: int = 2) -> list[dict]:
    """한 기관의 최근 13F-HR 제출 건 목록."""
    cik10 = str(cik).zfill(10)
    data = json.loads(_get(f"{SEC}/submissions/CIK{cik10}.json").decode())
    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    out = []
    for i, f in enumerate(forms):
        if f != "13F-HR":
            continue
        out.append({
            "accession": recent["accessionNumber"][i].replace("-", ""),
            "report_date": recent.get("reportDate", [None] * len(forms))[i],
            "filing_date": recent.get("filingDate", [None] * len(forms))[i],
        })
        if len(out) >= count:
            break
    return out


def _info_table_url(cik: str, accession: str) -> str | None:
    """
    제출 건 안에서 INFORMATION TABLE XML 파일을 찾는다.
    ★ 여기서 나는 오류를 삼키면 안 된다. 목록(data.sec.gov)은 읽히는데
      본문(www.sec.gov)은 막히는 경우가 있어서, 조용히 넘어가면
      '20곳 성공, 종목 0개'라는 말이 안 되는 결과가 나온다.
    """
    base = f"{WWW}/Archives/edgar/data/{int(cik)}/{accession}"
    idx = json.loads(_get(f"{base}/index.json").decode())   # 실패하면 SECError

    best = None
    for item in idx.get("directory", {}).get("item", []):
        n = item.get("name", "")
        low = n.lower()
        if not low.endswith(".xml"):
            continue
        flat = low.replace("_", "").replace("-", "")
        # 파일명은 제출자마다 다르다: form13fInfoTable.xml, informationtable.xml 등
        if "infotable" in flat or "informationtable" in flat:
            return f"{base}/{n}"
        if "primary" not in low:
            best = best or f"{base}/{n}"
    return best


def debug_one(name: str | None = None, verbose: bool = True) -> dict:
    """
    기관 한 곳을 끝까지 추적해 어디서 끊기는지 본다.
    13F가 0건 나올 때 원인을 찾는 용도.
    """
    name = name or next(iter(C.MANAGERS))
    cik = C.MANAGERS[name]
    out = {"manager": name, "cik": cik, "steps": []}

    def step(label, ok, note=""):
        out["steps"].append({"label": label, "ok": ok, "note": note})
        if verbose:
            print(f"  {'✅' if ok else '❌'} {label:<26} {note}")
        return ok

    if verbose:
        print(f"  대상: {name} (CIK {cik})\n")

    # 1) 제출 목록
    try:
        fils = latest_13f_filings(cik, count=2)
    except SECError as e:
        step("제출 목록 조회", False, f"{e.status} {e.detail[:60]}")
        out["verdict"] = "data.sec.gov 조회 실패"
        if verbose:
            print(f"\n  판정: {out['verdict']}")
        return out

    if not step("제출 목록 조회", bool(fils),
                f"13F-HR {len(fils)}건" if fils else "13F-HR 없음"):
        out["verdict"] = "이 기관은 13F-HR을 제출하지 않습니다 (CIK 확인 필요)"
        if verbose:
            print(f"\n  판정: {out['verdict']}")
        return out

    f = fils[0]
    step("최근 제출 건", True,
         f"{f['report_date']} 분기 (제출 {f['filing_date']})")

    # 2) 본문 파일 찾기 — 여기가 다른 서버(www.sec.gov)다
    try:
        url = _info_table_url(cik, f["accession"])
    except SECError as e:
        step("본문 목록 조회 (www)", False, f"{e.status} {e.detail[:60]}")
        out["verdict"] = ("www.sec.gov 가 막혔습니다. 목록은 되는데 본문이 "
                          "안 되면 종목이 0개로 나옵니다.")
        if verbose:
            print(f"\n  판정: {out['verdict']}")
        return out

    if not step("본문 파일 찾기", bool(url), url.rsplit("/", 1)[-1] if url else "XML 없음"):
        out["verdict"] = "제출 건 안에서 보유내역 XML을 못 찾았습니다"
        if verbose:
            print(f"\n  판정: {out['verdict']}")
        return out

    # 3) 내려받아 파싱
    try:
        raw = _get(url)
    except SECError as e:
        step("본문 다운로드", False, f"{e.status}")
        out["verdict"] = "본문 다운로드 실패"
        if verbose:
            print(f"\n  판정: {out['verdict']}")
        return out
    step("본문 다운로드", True, f"{len(raw):,} bytes")

    tbl = parse_info_table(raw)
    if not step("보유내역 파싱", not tbl.empty,
                f"{len(tbl)}개 종목" if not tbl.empty else "0개 — 파서 문제"):
        out["sample_xml"] = raw[:400].decode("utf-8", "ignore")
        out["verdict"] = "XML 형식이 예상과 다릅니다 (파서 수정 필요)"
        if verbose:
            print(f"\n  XML 앞부분:\n  {out['sample_xml'][:300]}")
            print(f"\n  판정: {out['verdict']}")
        return out

    out["sample"] = tbl.head(5)[["issuer", "issuer_key"]].to_dict("records")
    if verbose:
        print("\n  보유 종목 예시:")
        for r in out["sample"]:
            print(f"    {r['issuer']:<36} → {r['issuer_key']}")
    out["verdict"] = f"정상 — {len(tbl)}개 종목을 읽었습니다"
    if verbose:
        print(f"\n  판정: {out['verdict']}")
    return out


def parse_info_table(xml_bytes: bytes) -> pd.DataFrame:
    """
    13F INFORMATION TABLE 파싱.
    네임스페이스가 연도·제출자마다 달라서 태그 끝부분만 보고 찾는다.
    """
    def tag(e):
        return e.tag.rsplit("}", 1)[-1].lower()

    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return pd.DataFrame()

    rows = []
    for el in root.iter():
        if tag(el) != "infotable":
            continue
        rec = {}
        for child in el.iter():
            t = tag(child)
            txt = (child.text or "").strip()
            if t == "nameofissuer":
                rec["issuer"] = txt
            elif t == "cusip":
                rec["cusip"] = txt.upper()
            elif t == "value":
                rec["value"] = txt
            elif t == "sshprnamt":
                rec["shares"] = txt
            elif t == "titleofclass":
                rec["class"] = txt
        if rec.get("issuer"):
            rows.append(rec)

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    for c in ("value", "shares"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", ""),
                                  errors="coerce")
    df["issuer_key"] = df["issuer"].map(normalize_name)
    grp = {"shares": "sum", "value": "sum", "issuer": "first"}
    if "cusip" in df.columns:
        grp["cusip"] = "first"
    return df.groupby("issuer_key", as_index=False).agg(grp)


_SUFFIX = re.compile(
    r"\b(inc|incorporated|corp|corporation|co|company|plc|ltd|limited|"
    r"holdings?|hldgs?|group|grp|the|sa|nv|ag|lp|llc|cl|class|com|new|adr|"
    r"shs|ord|cap|stk)\b", re.I)

# 13F는 약어를 많이 쓴다. 우리 유니버스 표기와 맞추려면 풀어줘야 한다.
_ABBREV = {
    "intl": "international", "int l": "international",
    "tech": "technologies", "techs": "technologies", "technology": "technologies",
    "svcs": "services", "svc": "services", "serv": "services",
    "sys": "systems", "syst": "systems",
    "ind": "industries", "inds": "industries", "indus": "industries",
    "pharm": "pharmaceuticals", "pharma": "pharmaceuticals",
    "fin": "financial", "finl": "financial", "fncl": "financial",
    "amer": "american", "amn": "american",
    "natl": "national", "nat": "national",
    "mtrs": "motors", "mfg": "manufacturing",
    "res": "resources", "energ": "energy",
    "commun": "communications", "comm": "communications",
    "elec": "electric", "entpr": "enterprises", "entp": "enterprises",
    "lab": "laboratories", "labs": "laboratories",
    "prods": "products", "prod": "products",
    "stores": "stores", "strs": "stores",
}


def normalize_name(s: str) -> str:
    """
    회사명을 비교 가능한 형태로 정규화.
    CUSIP→티커 매핑이 무료로 없어서 이름으로 맞춰야 하고,
    13F 표기("EDISON INTL")와 우리 표기("Edison International")가 달라서
    법인 접미사를 떼고 약어를 풀어준다.
    """
    s = re.sub(r"[^A-Za-z0-9 ]", " ", str(s))
    s = re.sub(r"\s+", " ", s).strip().lower()
    words = [_ABBREV.get(w, w) for w in s.split()]
    s = _SUFFIX.sub(" ", " ".join(words))
    return re.sub(r"\s+", " ", s).strip()


def quarter_vwap(df: pd.DataFrame, report_date: str) -> float | None:
    """
    13F는 평단을 공시하지 않는다. 대신 그 분기의 거래량가중평균가를 써서
    '대략 이 가격대에서 담았다'를 추정한다. 어디까지나 근사치다.
    """
    if df is None or df.empty or not report_date:
        return None
    try:
        end = pd.Timestamp(report_date)
    except Exception:
        return None
    start = end - pd.Timedelta(days=92)
    win = df.loc[(df.index > start) & (df.index <= end)]
    if win.empty or win["Volume"].sum() <= 0:
        return None
    typical = (win["High"] + win["Low"] + win["Close"]) / 3
    return float((typical * win["Volume"]).sum() / win["Volume"].sum())


def collect(managers: dict[str, str] | None = None,
            verbose: bool = True) -> dict:
    """
    주요 기관들의 최근 2개 분기 13F를 모아 종목별 집계를 만든다.
    반환: {issuer_key: {holders, added, reduced, new, details, report_date}}
    """
    managers = managers or C.MANAGERS
    latest, prior = {}, {}
    meta = {"report_dates": [], "ok": [], "failed": []}

    # ★ 보유 내역 서버(www.sec.gov)가 막혔으면 20곳을 다 돌아봐야 헛수고다.
    #   먼저 한 번 찔러보고 막혔으면 바로 접는다. (로컬에서 4분 낭비 방지)
    try:
        _get(f"{WWW}/Archives/edgar/data/1067983/index.json", timeout=20)
    except SECError as e:
        meta["failed"] = list(managers)
        meta["errors"] = [str(e.status)]
        meta["blocked_host"] = "www.sec.gov"
        meta["universe_size"] = 0
        if verbose:
            print(f"    보유 내역 서버 차단 ({e.status}) — 13F를 건너뜁니다")
            print(f"    로컬에서 막히는 경우입니다. 클라우드에서는 보통 통과합니다.")
        return {"holdings": {}, "meta": meta}

    for name, cik in managers.items():
        try:
            fils = latest_13f_filings(cik, count=2)
            if not fils:
                meta["failed"].append(name)
                if verbose:
                    print(f"    {name}: 13F-HR 제출 내역 없음 (CIK 확인 필요)")
                continue

            got = 0
            for slot, f in zip((latest, prior), fils):
                url = _info_table_url(cik, f["accession"])
                if not url:
                    continue
                tbl = parse_info_table(_get(url))
                if tbl.empty:
                    continue
                if slot is latest:
                    meta["report_dates"].append(f["report_date"])
                    got = len(tbl)
                for _, r in tbl.iterrows():
                    slot.setdefault(r["issuer_key"], {})[name] = {
                        "shares": float(r.get("shares") or 0),
                        "value": float(r.get("value") or 0),
                        "issuer": r.get("issuer"),
                        "report_date": f["report_date"],
                    }

            # 종목을 하나도 못 읽었으면 '성공'이 아니다.
            # 이걸 성공으로 치면 '20곳 성공, 종목 0개'라는 모순이 생긴다.
            if got == 0:
                meta["failed"].append(name)
                meta.setdefault("errors", []).append("본문 읽기 실패")
                if verbose:
                    print(f"    {name}: 보유내역을 읽지 못함")
                continue

            meta["ok"].append(name)
            if verbose:
                print(f"    {name}: {got}개 종목")
        except SECError as e:
            meta["failed"].append(name)
            meta.setdefault("errors", []).append(str(e.status))
            if verbose:
                print(f"    {name}: 실패 ({e.status})"
                      + (f" {e.detail[:60]}" if e.detail else ""))
        except Exception as e:
            meta["failed"].append(name)
            meta.setdefault("errors", []).append(type(e).__name__)
            if verbose:
                print(f"    {name}: 실패 ({type(e).__name__})")

    agg = {}
    for key, holders in latest.items():
        prev = prior.get(key, {})
        added, reduced, new = [], [], []
        for m, cur in holders.items():
            p = prev.get(m)
            if p is None:
                new.append(m)
            elif cur["shares"] > p["shares"] * 1.02:
                added.append(m)
            elif cur["shares"] < p["shares"] * 0.98:
                reduced.append(m)
        agg[key] = {
            "issuer": next(iter(holders.values()))["issuer"],
            "holders": sorted(holders.keys()),
            "holder_count": len(holders),
            "added": sorted(added), "reduced": sorted(reduced),
            "new": sorted(new),
            "net": len(added) + len(new) - len(reduced),
            "report_date": next(iter(holders.values()))["report_date"],
            "total_value": sum(h["value"] for h in holders.values()),
        }
    meta["universe_size"] = len(agg)
    return {"holdings": agg, "meta": meta}


def match(agg: dict, ticker: str, company_name: str,
          price_df: pd.DataFrame | None = None) -> dict:
    """
    회사명으로 13F 집계와 매칭하고, 분기 VWAP으로 추정 평단을 계산한다.
    CUSIP→티커 매핑이 무료로 없어서 이름 기준으로 맞춘다.
    """
    key = normalize_name(company_name)
    rec = agg.get(key)

    if rec is None:                      # 부분 일치 시도
        cands = [k for k in agg if key and (key in k or k in key)]
        if len(cands) == 1:
            rec = agg[cands[0]]

    if rec is None:
        return {"available": False, "ticker": ticker,
                "reason": "주요 기관 보유 목록에 없음"}

    est = quarter_vwap(price_df, rec["report_date"]) if price_df is not None else None
    px = float(price_df["Close"].iloc[-1]) if price_df is not None and not price_df.empty else None
    vs = (px / est - 1) if (est and px) else None

    lag = None
    try:
        lag = (datetime.now() - datetime.strptime(rec["report_date"], "%Y-%m-%d")).days
    except Exception:
        pass

    # 기관별 행동 + 대표 인물 + 운용 성향 (리포트에 보여줄 형태로)
    act = {}
    for m in rec["new"]:
        act[m] = "신규"
    for m in rec["added"]:
        act[m] = "증가"
    for m in rec["reduced"]:
        act[m] = "감소"

    def rank(m):
        a = act.get(m, "유지")
        return ({"신규": 0, "증가": 1, "유지": 2, "감소": 3}[a],
                C.STYLE_ORDER.index(C.MANAGER_STYLE.get(m, "퀀트"))
                if C.MANAGER_STYLE.get(m) in C.STYLE_ORDER else 9)

    detail = [{
        "manager": m,
        "person": C.MANAGER_PERSON.get(m, ""),
        "style": C.MANAGER_STYLE.get(m, ""),
        "action": act.get(m, "유지"),
    } for m in sorted(rec["holders"], key=rank)]

    styles = {}
    for d in detail:
        styles[d["style"]] = styles.get(d["style"], 0) + 1

    return {
        "available": True, "ticker": ticker,
        "issuer": rec["issuer"],
        "holder_count": rec["holder_count"],
        "holders": rec["holders"][:8],
        "detail": detail,
        "styles": styles,
        "added": rec["added"], "reduced": rec["reduced"], "new": rec["new"],
        "net": rec["net"],
        "report_date": rec["report_date"],
        "lag_days": lag,
        "est_avg_price": None if est is None else round(est, 2),
        "price_vs_est": None if vs is None else round(vs, 4),
        "zone": _zone(vs),
    }


def _zone(vs: float | None) -> str:
    """추정 기관 평단 대비 현재가 위치."""
    if vs is None:
        return "평단 추정불가"
    if vs > C.F13_TOO_HIGH:
        return "고점"          # 기관 평단보다 한참 위 — 진입 부적합
    if vs < -C.F13_DEEP_BELOW:
        return "기관도 물림"   # 기관 평단보다 한참 아래
    return "적정"              # 기관과 비슷한 가격대
