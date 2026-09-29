"""
후보 종목을 읽기 좋은 HTML 리포트로 만든다.
표에 티커만 나열하는 대신, 회사명·섹터·낙폭을 한눈에 보이게 한다.
생성 후 브라우저에서 바로 열린다.
"""
import datetime as dt
import html
import os
import webbrowser

import pandas as pd

import config as C

SECTOR_KO = {
    "Information Technology": "정보기술",
    "Health Care": "헬스케어",
    "Financials": "금융",
    "Consumer Discretionary": "경기소비재",
    "Communication Services": "커뮤니케이션",
    "Industrials": "산업재",
    "Consumer Staples": "필수소비재",
    "Energy": "에너지",
    "Utilities": "유틸리티",
    "Real Estate": "부동산",
    "Materials": "소재",
}

# 지표 키 → (표시명, 배점, 한 줄 설명)
IND = {
    "atr_distance":    ("ATR 이격도", C.ATR_SCORE, "그 종목 변동성 대비 과하게 빠졌나"),
    "volume_climax":   ("거래량 소진", C.VOL_SCORE, "투매 터진 뒤 거래량이 줄었나"),
    "macd_divergence": ("MACD 다이버전스", C.MACD_DIV_SCORE, "하락 힘이 빠지고 있나"),
    "rsi":             ("RSI 과매도", C.RSI_SCORE, "2주간 매도 압력이 소진됐나"),
    "bollinger_pct_b": ("볼린저 이탈", C.BB_SCORE, "통계적으로 이상치 구간인가"),
    "drawdown_52w":    ("52주 낙폭", C.DRAWDOWN_SCORE, "고점 대비 충분히 빠졌나"),
    "fib_retracement": ("피보나치 되돌림", C.FIB_SCORE, "건전한 조정 범위인가"),
}


def _tier(score: int) -> tuple[str, str]:
    if score >= 14:
        return "최우선", "t1"
    if score >= 12:
        return "관심", "t2"
    return "참고", "t3"


def _severity(dd: float) -> str:
    """낙폭 깊이에 따른 색 단계."""
    if dd is None or pd.isna(dd):
        return "s0"
    if dd >= 0.40:
        return "s3"
    if dd >= 0.25:
        return "s2"
    return "s1"


GRADE_CLS = {"A": "g-a", "B": "g-b", "C": "g-c", "?": "g-u"}


def _fund_block(f: dict | None) -> str:
    """④단계 결과를 카드 안에 붙인다. 없으면 안내 한 줄."""
    if not f:
        return ('<div class="fund none">④ 악재 스크리닝 미실행 '
                '<code>--deep</code></div>')

    grade = f.get("grade", "?")
    verdict = html.escape(str(f.get("verdict", "")))
    cls = GRADE_CLS.get(grade, "g-u")
    lines = "".join(f"<li>{html.escape(x)}</li>" for x in f.get("reasons", [])[:5])
    blocked = f.get("blocks") or []
    tag = (f'<span class="blk">{html.escape(", ".join(blocked))}</span>'
           if blocked else "")

    fin = f.get("fin") or {}
    fin_html = ""
    if fin.get("available"):
        sc = fin["score"]
        items = "".join(
            f'<div class="fi"><span>{html.escape(v["label"])}</span>'
            f'<u><i style="width:{v["points"]/v["max"]*100:.0f}%"></i></u></div>'
            for v in fin["items"].values())
        fin_html = f"""<div class="fin">
  <div class="fin-h"><b>재무 점수</b>
    <span class="fin-n">{sc:.0f}<small>/100 · {fin['grade']}</small></span></div>
  <div class="fin-bar"><i style="width:{sc:.0f}%"></i></div>
  <div class="fin-items">{items}</div>
</div>"""

    return f"""<div class="fund {cls}">
  {fin_html}
  <div class="fund-h"><b>④ 악재 스크리닝</b>
    <span class="gr">{grade} · {verdict}</span></div>
  {tag}<ul>{lines}</ul>
</div>"""


def _ladder_block(lad: dict | None) -> str:
    """진입 사다리 — 이 리포트에서 사람이 실제로 쓰는 부분."""
    if not lad or not lad.get("rungs"):
        return ""
    px = lad["current_price"]
    rows = []
    for x in lad["rungs"]:
        now = x["step"] == 1
        drop = "현재" if now else f'{x["drop_from_now"]*100:+.1f}%'
        rows.append(
            f'<tr class="{"now" if now else ""}">'
            f'<td class="st">{x["step"]}차</td>'
            f'<td class="pr">${x["price"]:,.2f}</td>'
            f'<td class="dr">{drop}</td>'
            f'<td class="wt"><u><i style="width:{x["weight"]*100:.0f}%"></i></u>'
            f'<b>{x["weight"]*100:.0f}%</b></td>'
            f'<td class="at">{x["attractiveness"]:.0f}</td></tr>')

    chan = ('<span class="chan">하락 채널 — 1차 비중 축소</span>'
            if lad.get("in_channel") else "")
    return f"""<div class="lad">
  <div class="lad-h"><b>진입 사다리</b>{chan}</div>
  <table><thead><tr><th></th><th>가격</th><th>하락</th>
    <th>비중</th><th>매력도</th></tr></thead>
    <tbody>{''.join(rows)}</tbody></table>
  <div class="inval">폐기선 <b>${lad['invalidation']:,.2f}</b>
    ({lad['invalidation_drop']*100:+.1f}%) · 회복 목표 ${lad['recovery_target']:,.2f}</div>
</div>"""


ZONE_CLS = {"적정": "z-ok", "고점": "z-hi", "기관도 물림": "z-lo",
            "평단 추정불가": "z-u", "unknown": "z-u"}
ACT_CLS = {"신규": "new", "증가": "add", "감소": "cut", "유지": "keep"}
STYLE_CLS = {"가치투자": "val", "액티비스트": "act", "턴어라운드": "turn",
             "성장주": "grow", "매크로": "macro", "퀀트": "quant",
             "멀티전략": "multi"}


def _f13_block(f: dict | None) -> str:
    """⑤ 13F — 진입 필터로만 쓴다는 전제를 화면에도 남긴다."""
    if not f:
        return ""
    if not f.get("available"):
        return (f'<div class="f13 z-u"><b>⑤ 13F</b> '
                f'<span>{html.escape(str(f.get("reason", "자료 없음")))}</span></div>')

    zone = f.get("zone", "unknown")
    cls = ZONE_CLS.get(zone, "z-u")
    vs = f.get("price_vs_est")
    vs_txt = f"{vs*100:+.1f}%" if vs is not None else "—"
    est = f.get("est_avg_price")
    moves = []
    if f.get("new"):
        moves.append(f"신규 {len(f['new'])}")
    if f.get("added"):
        moves.append(f"증가 {len(f['added'])}")
    if f.get("reduced"):
        moves.append(f"감소 {len(f['reduced'])}")

    # 기관 목록 — 이름 + 대표 인물 + 운용 성향
    rows = ""
    for d in (f.get("detail") or [])[:6]:
        style = d.get("style") or ""
        act = d.get("action") or ""
        rows += (
            f'<li class="s-{ACT_CLS.get(act, "keep")}">'
            f'<span class="mgr">{html.escape(d["manager"])}'
            + (f' <em>{html.escape(d["person"])}</em>' if d.get("person") else "")
            + f'</span>'
            f'<span class="sty y-{STYLE_CLS.get(style, "etc")}">{html.escape(style)}</span>'
            f'<span class="act">{html.escape(act)}</span></li>')
    more = max(0, len(f.get("detail") or []) - 6)
    if more:
        rows += f'<li class="s-keep"><span class="mgr">외 {more}곳</span></li>'

    return f"""<div class="f13 {cls}">
  <div class="f13-h"><b>⑤ 13F 기관 보유</b><span class="zn">{html.escape(zone)}</span></div>
  <div class="f13-r"><span>보유 기관</span><b>{f['holder_count']}곳</b>
    <span class="mv">{' · '.join(moves) or '변동 없음'}</span></div>
  <ul class="mgrs">{rows}</ul>
  <div class="f13-r"><span>추정 평단</span>
    <b>{'—' if est is None else f'${est:,.2f}'}</b>
    <span class="mv">현재가 {vs_txt}</span></div>
  <div class="f13-note">{f.get('report_date','')} 분기 보유 신고 ·
    {f.get('lag_days','?')}일 지연 · 평단은 분기 거래량가중평균가로 <b>추정</b>한 값<br>
    <b>보유 신고일 뿐 해당 기관의 추천·의견이 아닙니다.</b>
    진입 시점 판단에만 쓰고, 이미 보유한 종목을 계속 들고 있을 근거로는 쓰지 않습니다.</div>
</div>"""


def _card(r: pd.Series, deep: dict | None = None,
          ladders: dict | None = None, f13: dict | None = None) -> str:
    t = html.escape(str(r["ticker"]))
    name = html.escape(str(r.get("name") or t))
    sector = str(r.get("sector") or "")
    sec_ko = SECTOR_KO.get(sector, sector)
    score = int(r["score"])
    tier_ko, tier_cls = _tier(score)
    hits = set(str(r.get("hits") or "").split(","))

    dd = r.get("dd_52w")
    dd_pct = f"{dd*100:.1f}%" if pd.notna(dd) else "—"
    sev = _severity(dd)

    vs = r.get("vs_ma200")
    slope = r.get("ma200_slope")
    rsi = r.get("rsi")
    atr = r.get("atr_dist")

    # 지표별 획득 점수 (부분점수제)
    earned = {}
    for part in str(r.get("scores") or "").split(","):
        if ":" in part:
            k, _, v = part.partition(":")
            try:
                earned[k] = float(v)
            except ValueError:
                pass

    chips = []
    for key, (label, pts, why) in IND.items():
        got = earned.get(key)
        on = key in hits if got is None else got >= pts * C.HIT_FRACTION
        shown = (pts if on else 0) if got is None else got
        cls = "on" if on else ("part" if shown > 0 else "off")
        chips.append(
            f'<span class="chip {cls}" title="{html.escape(why)} (배점 {pts})">'
            f'{html.escape(label)}<b>{shown:g}</b></span>')

    pct = score / C.MAX_SCORE * 100
    return f"""
<article class="card {tier_cls}">
  <header class="card-h">
    <div class="id">
      <div class="tk">{t}</div>
      <div class="nm">{name}</div>
      <div class="sec">{html.escape(sec_ko)}</div>
    </div>
    <div class="sc">
      <div class="sc-n"><b>{score}</b><span>/{C.MAX_SCORE}</span></div>
      <div class="sc-t">{tier_ko}</div>
    </div>
  </header>

  <div class="bar"><i style="width:{pct:.1f}%"></i></div>

  <dl class="stats">
    <div><dt>현재가</dt><dd>${r['price']:,.2f}</dd></div>
    <div class="{sev}"><dt>52주 고점 대비</dt><dd class="dd">−{dd_pct}</dd></div>
    <div><dt>200일선 대비</dt><dd>{vs*100:+.1f}%</dd></div>
    <div><dt>200일선 기울기</dt><dd class="{'up' if slope and slope > 0 else 'dn'}">{slope*100:+.2f}%</dd></div>
    <div><dt>RSI</dt><dd>{rsi if pd.notna(rsi) else '—'}</dd></div>
    <div><dt>ATR 이격</dt><dd>{atr if pd.notna(atr) else '—'}배</dd></div>
  </dl>

  <div class="chips">{''.join(chips)}</div>
  {_ladder_block((ladders or {}).get(t))}
  {_fund_block((deep or {}).get(t))}
  {_f13_block((f13 or {}).get(t))}
</article>"""


CSS = """
/* 레이아웃: 요약 스트립 위, 종목 카드 그리드 아래 — 화면 폭에 따라 1~3열 */
:root{
  --bg:#f6f5f2; --panel:#fffefb; --line:#e3e0d8; --line2:#d3cfc4;
  --fg:#1c1b18; --mut:#6f6b61; --faint:#98938a;
  --ink:#2b3a4a; --gold:#a8762a; --up:#1f7a4d; --dn:#b03c2e;
  --s1:#8a8578; --s2:#c07a1f; --s3:#b03c2e;
  --mono:'IBM Plex Mono','Consolas',monospace;
  --sans:'IBM Plex Sans','Malgun Gothic','맑은 고딕',system-ui,sans-serif;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --bg:#14151a; --panel:#1c1e25; --line:#2b2e37; --line2:#3a3e49;
  --fg:#eceae4; --mut:#9a9689; --faint:#6e6a61;
  --ink:#a9c2dc; --gold:#d3a457; --up:#4fbc84; --dn:#e0715c;
  --s1:#7d786c; --s2:#d3a457; --s3:#e0715c;
  color-scheme:dark;
}}
:root[data-theme="dark"]{
  --bg:#14151a; --panel:#1c1e25; --line:#2b2e37; --line2:#3a3e49;
  --fg:#eceae4; --mut:#9a9689; --faint:#6e6a61;
  --ink:#a9c2dc; --gold:#d3a457; --up:#4fbc84; --dn:#e0715c;
  --s1:#7d786c; --s2:#d3a457; --s3:#e0715c;
  color-scheme:dark;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font-family:var(--sans);
     -webkit-font-smoothing:antialiased}
.wrap{max-width:1180px;margin:0 auto;padding:0 16px;padding-block:32px 64px}

.head{border-bottom:2px solid var(--fg);padding-bottom:18px;margin-bottom:22px}
.kicker{font-size:11px;letter-spacing:.18em;text-transform:uppercase;
        color:var(--gold);font-weight:600}
h1{font-size:clamp(24px,4vw,34px);margin:8px 0 6px;letter-spacing:-.02em;
   text-wrap:balance}
.sub{color:var(--mut);font-size:13px;margin:0}

.funnel{display:flex;flex-wrap:wrap;gap:1px;background:var(--line);
        border:1px solid var(--line);margin:22px 0 10px;border-radius:3px;
        overflow:hidden}
.fn{flex:1 1 130px;min-width:0;background:var(--panel);padding:12px 14px}
.fn dt{font-size:11px;color:var(--mut);margin-bottom:5px}
.fn dd{margin:0;font-family:var(--mono);font-size:21px;font-weight:600;
       font-variant-numeric:tabular-nums}
.fn.hl dd{color:var(--gold)}
.note{font-size:12px;color:var(--faint);margin:0 0 30px}

.grid{display:grid;gap:14px;grid-template-columns:1fr}
@media(min-width:680px){.grid{grid-template-columns:repeat(2,1fr)}}
@media(min-width:1000px){.grid{grid-template-columns:repeat(3,1fr)}}

.card{background:var(--panel);border:1px solid var(--line);border-radius:4px;
      padding:16px;display:flex;flex-direction:column;gap:12px;min-width:0}
.card.t1{border-color:var(--gold);box-shadow:0 0 0 1px var(--gold) inset}
.card.t2{border-color:var(--line2)}

.card-h{display:flex;justify-content:space-between;align-items:flex-start;gap:10px}
.id{min-width:0}
.tk{font-family:var(--mono);font-size:19px;font-weight:600;letter-spacing:.01em}
.nm{font-size:13px;color:var(--fg);margin-top:1px;overflow-wrap:anywhere}
.sec{font-size:11px;color:var(--faint);margin-top:3px}
.sc{text-align:right;flex-shrink:0}
.sc-n{font-family:var(--mono);font-variant-numeric:tabular-nums;line-height:1}
.sc-n b{font-size:26px;font-weight:600}
.sc-n span{font-size:12px;color:var(--faint)}
.sc-t{font-size:10px;letter-spacing:.1em;color:var(--mut);margin-top:5px}
.t1 .sc-n b{color:var(--gold)}

.bar{height:4px;background:var(--line);border-radius:2px;overflow:hidden}
.bar i{display:block;height:100%;background:var(--ink)}
.t1 .bar i{background:var(--gold)}

.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:10px 8px;margin:0}
.stats>div{min-width:0}
.stats dt{font-size:10px;color:var(--faint);margin-bottom:3px;
          white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.stats dd{margin:0;font-family:var(--mono);font-size:13px;font-weight:500;
          font-variant-numeric:tabular-nums}
.s1 .dd{color:var(--s1)} .s2 .dd{color:var(--s2)} .s3 .dd{color:var(--s3);font-weight:600}
.up{color:var(--up)} .dn{color:var(--dn)}

.chips{display:flex;flex-wrap:wrap;gap:5px;border-top:1px solid var(--line);
       padding-top:11px}
.fund{border-top:1px solid var(--line);padding-top:11px;font-size:11.5px;
      margin-top:auto}
.fund.none{color:var(--faint)}
.fund code{font-family:var(--mono);font-size:10.5px;background:var(--line);
           padding:1px 4px;border-radius:2px}
.fund-h{display:flex;justify-content:space-between;align-items:center;gap:8px;
        margin-bottom:7px}
.fund-h b{font-size:11px;font-weight:600}
.gr{font-family:var(--mono);font-size:10.5px;font-weight:600;padding:2px 7px;
    border-radius:2px;white-space:nowrap}
.g-a .gr{background:color-mix(in srgb,var(--up) 18%,transparent);color:var(--up)}
.g-b .gr{background:color-mix(in srgb,var(--s2) 18%,transparent);color:var(--s2)}
.g-c .gr{background:color-mix(in srgb,var(--dn) 18%,transparent);color:var(--dn)}
.g-u .gr{background:var(--line);color:var(--mut)}
.blk{display:inline-block;font-size:10.5px;color:var(--dn);font-weight:600;
     margin-bottom:5px}
.fund ul{margin:0;padding-left:15px;color:var(--mut);line-height:1.65}
.fund li{margin-bottom:2px}
.g-c{opacity:.72}

.fin{margin-bottom:12px}
.fin-h{display:flex;justify-content:space-between;align-items:baseline;gap:8px;
       margin-bottom:6px}
.fin-h b{font-size:11px;font-weight:600}
.fin-n{font-family:var(--mono);font-size:19px;font-weight:600;
       font-variant-numeric:tabular-nums;color:var(--ink)}
.fin-n small{font-size:10px;color:var(--faint);font-weight:500;margin-left:2px}
.fin-bar{height:6px;background:var(--line);border-radius:3px;overflow:hidden;
         margin-bottom:9px}
.fin-bar i{display:block;height:100%;background:var(--ink)}
/* 진입 사다리 */
.lad{border-top:1px solid var(--line);padding-top:11px;margin-top:11px}
.lad-h{display:flex;justify-content:space-between;align-items:center;gap:8px;
       margin-bottom:7px;font-size:11px}
.lad-h b{font-weight:600}
.chan{font-size:10px;color:var(--s2);font-weight:600}
.lad table{width:100%;border-collapse:collapse;font-size:11px}
.lad th{font-size:9.5px;color:var(--faint);font-weight:500;text-align:right;
        padding-bottom:3px}
.lad th:first-child,.lad td.st{text-align:left}
.lad td{padding:2.5px 0;font-family:var(--mono);font-variant-numeric:tabular-nums;
        text-align:right;color:var(--mut)}
.lad td.st{font-family:var(--sans);font-size:10px;color:var(--faint)}
.lad tr.now td{color:var(--fg);font-weight:600}
.lad tr.now td.st::after{content:" ●";color:var(--gold);font-size:8px}
.lad td.wt{display:flex;align-items:center;gap:5px;justify-content:flex-end}
.lad td.wt u{width:34px;height:3px;background:var(--line);border-radius:2px;
             text-decoration:none;display:block;flex:0 0 34px}
.lad td.wt i{display:block;height:100%;background:var(--ink);border-radius:2px;
             opacity:.65}
.lad td.wt b{font-weight:500;font-size:10px;min-width:28px}
.lad td.at{color:var(--ink);font-weight:600}
.inval{font-size:10px;color:var(--faint);margin-top:6px}

/* 13F */
.f13{border-top:1px solid var(--line);padding-top:11px;margin-top:11px;font-size:11px}
.f13-h{display:flex;justify-content:space-between;align-items:center;gap:8px;
       margin-bottom:6px}
.f13-h b{font-size:11px;font-weight:600}
.zn{font-size:10px;font-weight:600;padding:2px 7px;border-radius:2px}
.z-ok .zn{background:color-mix(in srgb,var(--up) 18%,transparent);color:var(--up)}
.z-hi .zn{background:color-mix(in srgb,var(--dn) 18%,transparent);color:var(--dn)}
.z-lo .zn{background:color-mix(in srgb,var(--s2) 18%,transparent);color:var(--s2)}
.z-u{color:var(--faint)}
.f13-r{display:flex;align-items:baseline;gap:7px;color:var(--mut);
       font-size:10.5px;margin-bottom:2px}
.f13-r>span:first-child{flex:0 0 56px;color:var(--faint)}
.f13-r b{font-family:var(--mono);font-variant-numeric:tabular-nums;color:var(--fg)}
.mv{font-size:10px;color:var(--faint)}
.f13-note{font-size:9.5px;color:var(--faint);margin-top:6px;line-height:1.55}
.f13-note b{color:var(--mut)}

/* 기관 목록 — 이름 + 대표 인물 + 운용 성향 */
.mgrs{list-style:none;margin:7px 0;padding:0;display:flex;flex-direction:column;
      gap:3px}
.mgrs li{display:flex;align-items:center;gap:6px;font-size:10.5px;min-width:0}
.mgr{flex:1;min-width:0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;
     color:var(--fg);font-weight:500}
.mgr em{font-style:normal;color:var(--faint);font-weight:400}
.sty{flex:0 0 auto;font-size:9px;padding:1.5px 5px;border-radius:2px;
     white-space:nowrap}
.y-val{background:color-mix(in srgb,var(--gold) 20%,transparent);color:var(--gold)}
.y-act{background:color-mix(in srgb,var(--dn) 18%,transparent);color:var(--dn)}
.y-turn{background:color-mix(in srgb,var(--s2) 18%,transparent);color:var(--s2)}
.y-grow{background:color-mix(in srgb,var(--up) 18%,transparent);color:var(--up)}
.y-macro,.y-quant,.y-multi,.y-etc{background:var(--line);color:var(--mut)}
.act{flex:0 0 26px;text-align:right;font-size:9.5px;font-weight:600}
.s-new .act{color:var(--up)}
.s-add .act{color:var(--up);opacity:.8}
.s-cut .act{color:var(--dn)}
.s-keep .act{color:var(--faint);font-weight:400}
.s-keep .mgr{color:var(--mut);font-weight:400}

.fin-items{display:grid;grid-template-columns:repeat(2,1fr);gap:3px 10px}
.fi{display:flex;align-items:center;gap:6px;font-size:10px;color:var(--mut)}
.fi span{flex:0 0 54px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.fi u{flex:1;min-width:0;height:3px;background:var(--line);border-radius:2px;
      text-decoration:none;display:block}
.fi i{display:block;height:100%;background:var(--ink);border-radius:2px;
      opacity:.6}
.chip{font-size:10.5px;padding:3px 7px;border-radius:2px;border:1px solid transparent;
      display:inline-flex;gap:5px;align-items:center;cursor:help}
.chip b{font-family:var(--mono);font-weight:600}
.chip.on{background:color-mix(in srgb,var(--ink) 13%,transparent);
         border-color:color-mix(in srgb,var(--ink) 32%,transparent);color:var(--fg)}
.chip.part{background:color-mix(in srgb,var(--ink) 6%,transparent);
           border-color:var(--line2);color:var(--mut)}
.chip.off{color:var(--faint);border-color:var(--line);text-decoration:line-through;
          text-decoration-color:var(--line2)}
.chip.off b{opacity:.45}

.empty{background:var(--panel);border:1px dashed var(--line2);border-radius:4px;
       padding:44px 20px;text-align:center;color:var(--mut)}
.foot{margin-top:40px;padding-top:16px;border-top:1px solid var(--line);
      font-size:11.5px;color:var(--faint);line-height:1.75}
@media print{body{background:#fff}.card{break-inside:avoid}}
"""


def _deep_line(deep: dict | None) -> str:
    if not deep:
        return ""
    g = [v.get("grade") for v in deep.values()]
    return (f' · ④단계 통과 {g.count("A")} / 조건부 {g.count("B")} / '
            f'차단 {g.count("C")}')


def build(cands: pd.DataFrame, stats: dict, out: str = "report.html",
          deep: dict | None = None, ladders: dict | None = None,
          f13: dict | None = None, open_browser: bool = True) -> str:
    f = stats.get("funnel", {})
    now = dt.datetime.now()
    nodata = f.get("nodata", 0) + f.get("history", 0)
    total = sum(f.get(k, 0) for k in
                ["nodata", "history", "trend", "structure", "cooling",
                 "score", "passed"])
    evaluated = total - nodata

    if cands.empty:
        body = ('<div class="empty"><b>오늘은 조건을 만족하는 종목이 없습니다.</b>'
                '<br>시장이 충분히 빠지지 않았거나, 빠진 종목의 추세가 이미 꺾였습니다.</div>')
    else:
        body = ('<div class="grid">'
                + "".join(_card(r, deep, ladders, f13)
                          for _, r in cands.iterrows())
                + "</div>")

    warn = ""
    if nodata and total and nodata / total > 0.05:
        warn = (f'<p class="note" style="color:var(--dn)">⚠ {nodata}종목의 데이터가 비어 '
                f'있어 시장 전체를 대표하지 못합니다. <code>py run.py --repair</code> 실행 필요.</p>')

    n1 = int((cands["score"] >= 14).sum()) if not cands.empty else 0
    n2 = int(((cands["score"] >= 12) & (cands["score"] < 14)).sum()) if not cands.empty else 0

    doc = f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>과매도 스크리너 — {now:%Y-%m-%d}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>{CSS}</style></head><body>
<div class="wrap">
  <header class="head">
    <div class="kicker">Oversold Screener · 미국 주식</div>
    <h1>추세는 살아있고 과하게 빠진 종목</h1>
    <p class="sub">{now:%Y년 %m월 %d일 %H:%M} 기준 · S&amp;P 500 + 다우 30 · 커트라인 {C.SCORE_THRESHOLD}점</p>
  </header>

  <dl class="funnel">
    <div class="fn"><dt>평가 종목</dt><dd>{evaluated}</dd></div>
    <div class="fn"><dt>추세 이탈 차단</dt><dd>{f.get('trend',0)}</dd></div>
    <div class="fn"><dt>하락 채널 차단</dt><dd>{f.get('structure',0)}</dd></div>
    <div class="fn"><dt>급락 직후 대기</dt><dd>{f.get('cooling',0)}</dd></div>
    <div class="fn"><dt>점수 미달</dt><dd>{f.get('score',0)}</dd></div>
    <div class="fn hl"><dt>최종 후보</dt><dd>{f.get('passed',0)}</dd></div>
  </dl>
  <p class="note">최우선 {n1}종목 · 관심 {n2}종목{_deep_line(deep)} ·
     지표 칩에 마우스를 올리면 뜻이 나옵니다.</p>
  {warn}

  {body}

  <footer class="foot">
    <b>읽는 법</b> — 차트 점수({C.MAX_SCORE}점 만점)는 <b>언제</b> 사느냐를,
    재무 점수(100점)는 <b>무엇을</b> 사느냐를 말합니다. 순위는 재무 점수 순이고,
    차트 점수는 같은 등급 안에서의 보조 기준입니다.<br>
    <b>진입 사다리</b> — 손절선이 없는 전략에서 진입 가격은 유일한 방어선입니다.
    그래서 한 점이 아니라 구간으로 나눠 삽니다. 매력도는 회복 목표가까지의
    기대수익에 차트 타이밍과 재무 품질을 더한 값입니다. 하락 채널로 판정된 종목은
    차단하지 않고 1차 비중을 줄이는 대신 아래 칸을 두껍게 잡습니다.<br>
    <b>13F의 한계</b> — 기관 공시는 최대 45일 늦고 롱 포지션만 나오며 평단은
    공시되지 않습니다(분기 거래량가중평균가로 추정). 그래서 진입 판단에만 쓰고,
    이미 보유한 종목을 계속 들고 있을 근거로는 쓰지 않습니다.<br>
    이 리포트는 후보를 추린 것이지 매수 신호가 아닙니다.
  </footer>
</div></body></html>"""

    with open(out, "w", encoding="utf-8") as fp:
        fp.write(doc)

    if open_browser:
        try:
            webbrowser.open("file://" + os.path.abspath(out))
        except Exception:
            pass
    return os.path.abspath(out)
