"""
텔레그램 알림.

알림 피로를 막는 게 이 파일의 핵심 설계다.
매일 "오늘 후보 12개입니다"를 보내면 일주일이면 아무도 안 본다.
그래서 '상태'가 아니라 '변화'만 보낸다.

  · 신규 진입   후보 목록에 처음 올라온 종목
  · 진입가 도달 네가 정해둔 사다리 칸에 가격이 닿음
  · 폐기선 이탈 가설이 틀렸다고 볼 선을 깸
  · 이탈       후보에서 빠짐

아무 일도 없는 날은 알림이 아예 가지 않는다. 그게 정상이다.
"""
import json
import os
import urllib.parse
import urllib.request
from datetime import datetime

import config as C

STATE_PATH = "data/alert_state.json"
API = "https://api.telegram.org/bot{token}/{method}"


# ═══════════════════════════════════════════
# 상태 저장 — '변화'를 알려면 어제를 기억해야 한다
# ═══════════════════════════════════════════

def load_state(path: str = STATE_PATH) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"tickers": {}, "updated": None}


def save_state(state: dict, path: str = STATE_PATH) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    state["updated"] = datetime.now().isoformat(timespec="seconds")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=1)


# ═══════════════════════════════════════════
# 변화 감지
# ═══════════════════════════════════════════

def diff_events(cands, deep: dict, ladders: dict, state: dict) -> list[dict]:
    """
    어제 상태와 비교해 알릴 만한 변화만 뽑는다.
    cands: 후보 DataFrame (ticker, name, score, price ...)
    """
    prev = state.get("tickers", {})
    events, now = [], {}

    for _, r in cands.iterrows():
        t = str(r["ticker"])
        px = float(r["price"])
        lad = ladders.get(t) or {}
        d = deep.get(t) or {}
        rungs = lad.get("rungs", [])

        now[t] = {
            "price": px,
            "score": float(r.get("score") or 0),
            "fin_score": d.get("fin_score"),
            "grade": d.get("grade"),
            "rung_prices": [x["price"] for x in rungs],
            "invalidation": lad.get("invalidation"),
            "reached": list((prev.get(t) or {}).get("reached", [])),
        }
        p = prev.get(t)

        if p is None:
            events.append({"kind": "new", "ticker": t, "row": r,
                           "deep": d, "ladder": lad})
            continue

        # 사다리 칸 도달 — 위에서 아래로 내려온 경우만
        for rung in rungs[1:]:
            lvl = rung["price"]
            key = f"{lvl:.2f}"
            if key in now[t]["reached"]:
                continue
            if p["price"] > lvl >= px:
                now[t]["reached"].append(key)
                events.append({"kind": "rung", "ticker": t, "row": r,
                               "deep": d, "ladder": lad, "rung": rung})

        inv = lad.get("invalidation")
        if inv and p["price"] > inv >= px:
            events.append({"kind": "invalid", "ticker": t, "row": r,
                           "deep": d, "ladder": lad})

    for t, p in prev.items():
        if t not in now:
            events.append({"kind": "dropped", "ticker": t,
                           "last_price": p.get("price")})

    return events, {"tickers": now}


# ═══════════════════════════════════════════
# 메시지 작성
# ═══════════════════════════════════════════

def _esc(s) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def format_event(e: dict, report_url: str | None = None) -> str:
    k = e["kind"]
    t = _esc(e["ticker"])

    if k == "dropped":
        return f"➖ <b>{t}</b> 후보에서 빠졌습니다 (마지막 ${e['last_price']:,.2f})"

    r, d, lad = e["row"], e.get("deep") or {}, e.get("ladder") or {}
    name = _esc(r.get("name") or t)
    px = float(r["price"])
    fin = d.get("fin_score")
    fin_txt = f"재무 {fin:.0f}점 {d.get('fin_grade') or ''}" if fin else "재무 미확인"
    chart = f"차트 {float(r.get('score') or 0):.1f}/{C.MAX_SCORE}"

    if k == "new":
        head = f"🆕 <b>{t}</b> 신규 후보\n<i>{name}</i>"
    elif k == "rung":
        step = e["rung"]["step"]
        head = (f"🔔 <b>{t}</b> {step}차 진입가 도달\n<i>{name}</i>")
    else:
        head = f"⚠️ <b>{t}</b> 폐기선 이탈\n<i>{name}</i>"

    lines = [head, ""]
    lines.append(f"현재가 <b>${px:,.2f}</b>"
                 + (f" · 52주 고점 대비 {-float(r['dd_52w'])*100:.1f}%"
                    if r.get("dd_52w") is not None else ""))
    lines.append(f"{fin_txt} · {chart}")

    if k == "invalid":
        lines.append("")
        lines.append("판단 근거가 깨졌습니다. 재검토하세요.")
    elif lad.get("rungs"):
        lines.append("")
        lines.append("<b>진입 사다리</b>")
        for x in lad["rungs"]:
            mark = "▸" if abs(x["price"] - px) / max(px, 1) < 0.01 else " "
            drop = "현재" if x["step"] == 1 else f"{x['drop_from_now']*100:+.1f}%"
            lines.append(f"{mark} {x['step']}차 ${x['price']:,.2f} ({drop}) "
                         f"비중 {x['weight']*100:.0f}% · 매력도 {x['attractiveness']:.0f}")
        if lad.get("invalidation"):
            lines.append(f"  폐기선 ${lad['invalidation']:,.2f}")

    if d.get("blocks"):
        lines.append("")
        lines.append("⛔ " + _esc(", ".join(d["blocks"])))
    elif d.get("reasons"):
        lines.append("")
        lines.append("· " + _esc(d["reasons"][0]))

    if report_url:
        lines.append("")
        lines.append(f'<a href="{_esc(report_url)}">전체 리포트 보기</a>')

    return "\n".join(lines)


# ═══════════════════════════════════════════
# 전송
# ═══════════════════════════════════════════

def send(text: str, token: str | None = None, chat_id: str | None = None,
         timeout: int = 20) -> bool:
    token = token or os.environ.get("TELEGRAM_TOKEN", "")
    chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        print("  (텔레그램 설정 없음 — 전송 건너뜀)")
        return False

    data = urllib.parse.urlencode({
        "chat_id": chat_id, "text": text, "parse_mode": "HTML",
        "disable_web_page_preview": "true",
    }).encode()
    try:
        req = urllib.request.Request(API.format(token=token, method="sendMessage"),
                                     data=data)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode()).get("ok", False)
    except Exception as e:
        print(f"  텔레그램 전송 실패: {type(e).__name__}")
        return False


def notify(cands, deep: dict, ladders: dict, report_url: str | None = None,
           state_path: str = STATE_PATH, dry_run: bool = False) -> dict:
    """변화를 감지해 알림을 보내고 상태를 저장한다."""
    state = load_state(state_path)
    first_run = not state.get("tickers")
    events, new_state = diff_events(cands, deep, ladders, state)

    # ★ 첫 실행은 기준점만 잡는다.
    #   비교할 어제가 없으니 후보 전부가 '신규'로 잡혀 알림이 쏟아진다.
    #   그러면 첫날부터 알림을 안 보게 되므로, 조용히 상태만 저장한다.
    if first_run:
        if not dry_run:
            save_state(new_state, state_path)
        return {"events": 0, "sent": 0, "kinds": [], "first_run": True,
                "baseline": len(new_state["tickers"])}

    order = {"invalid": 0, "rung": 1, "new": 2, "dropped": 3}
    events.sort(key=lambda e: order.get(e["kind"], 9))
    events = events[:C.ALERT_MAX_PER_RUN]

    sent = 0
    for e in events:
        msg = format_event(e, report_url)
        if dry_run:
            print("─" * 50)
            print(msg.replace("<b>", "").replace("</b>", "")
                  .replace("<i>", "").replace("</i>", ""))
        elif send(msg):
            sent += 1

    if not dry_run:
        save_state(new_state, state_path)

    return {"events": len(events), "sent": sent,
            "kinds": [e["kind"] for e in events]}
