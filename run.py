#!/usr/bin/env python3
"""
메인 엔트리.

  python run.py --setup     최초 1회: 유니버스 + 10년치 시세 + 메타데이터 (30~60분)
  python run.py             일일 실행: 증분 갱신 후 스크리닝 (1~2분)
  python run.py --no-sync   저장된 데이터로만 스크리닝 (몇 초)
"""
import argparse
import datetime as dt
import os
import sys
import webbrowser


def safe_write(path: str, writer) -> str:
    """
    파일이 엑셀·브라우저 등에서 열려 있어 잠긴 경우,
    이름 뒤에 시각을 붙인 새 파일로 저장한다. (작업이 통째로 날아가지 않게)
    """
    try:
        writer(path)
        return path
    except PermissionError:
        stem, ext = os.path.splitext(path)
        alt = f"{stem}_{dt.datetime.now():%H%M%S}{ext}"
        writer(alt)
        print(f"  ⚠ {os.path.basename(path)} 이(가) 열려 있어 "
              f"{os.path.basename(alt)} 로 저장했습니다.")
        return alt

import config as C
import data as D
import fundamental as F
import ladder as L
import notify as N
import report as R
import screener as S
import thirteenf as T13
import universe as U


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setup", action="store_true", help="최초 전체 구축")
    ap.add_argument("--no-sync", action="store_true", help="다운로드 없이 스크리닝만")
    ap.add_argument("--repair", action="store_true",
                    help="빠진 종목만 개별 재다운로드 (스크리닝 안 함)")
    ap.add_argument("--status", action="store_true",
                    help="데이터 보유 현황만 확인")
    ap.add_argument("--refresh-meta", action="store_true", help="섹터/시총 갱신 (주 1회)")
    ap.add_argument("--csv", default="candidates.csv", help="후보 저장 경로")
    ap.add_argument("--report", default="report.html", help="리포트 저장 경로")
    ap.add_argument("--deep", action="store_true",
                    help="④악재 스크리닝·재무 점수까지 실행")
    ap.add_argument("--deep-top", type=int, default=15,
                    help="④단계를 적용할 상위 종목 수 (기본 15)")
    ap.add_argument("--f13", action="store_true",
                    help="⑤13F 기관 보유 확인 (SEC 조회, 2~4분)")
    ap.add_argument("--f13-test", action="store_true",
                    help="SEC 접속만 점검 (왜 막히는지 확인)")
    ap.add_argument("--f13-debug", nargs="?", const="", metavar="기관명",
                    help="기관 한 곳을 끝까지 추적 (어디서 끊기는지 확인)")
    ap.add_argument("--notify", action="store_true",
                    help="변화가 있으면 텔레그램 알림 전송")
    ap.add_argument("--notify-dry", action="store_true",
                    help="알림 내용만 화면에 출력 (전송 안 함)")
    ap.add_argument("--report-url", default=os.environ.get("REPORT_URL", ""),
                    help="알림에 넣을 공개 리포트 주소")
    ap.add_argument("--all", action="store_true",
                    help="--deep --f13 --notify 를 한 번에")
    ap.add_argument("--no-report", action="store_true", help="리포트 생성 안 함")
    ap.add_argument("--no-open", action="store_true", help="브라우저 자동 실행 안 함")
    args = ap.parse_args()
    if args.all:
        args.deep = args.f13 = args.notify = True

    if args.f13_test:
        print("SEC 접속 점검\n" + "=" * 60)
        T13.diagnose()
        return 0

    if args.f13_debug is not None:
        name = args.f13_debug or None
        if name and name not in C.MANAGERS:
            print(f"'{name}' 은 등록된 기관이 아닙니다. 사용 가능:")
            for k in C.MANAGERS:
                print(f"  {k}")
            return 1
        print("13F 단계별 추적\n" + "=" * 60)
        T13.debug_one(name)
        return 0

    print("유니버스 구성 중...")
    uni = U.build_universe()
    tickers = uni["ticker"].tolist()

    conn = D.connect()

    if args.status:
        cov = D.coverage(conn, tickers)
        print(f"\n데이터 보유 현황 (전체 {len(tickers)}종목)")
        print(f"  사용 가능 : {len(cov['ok'])}")
        print(f"  히스토리 부족 : {len(cov['thin'])}")
        print(f"  아예 없음 : {len(cov['missing'])}")
        if cov["missing"]:
            print(f"    {', '.join(cov['missing'][:25])}"
                  f"{' ...' if len(cov['missing']) > 25 else ''}")
        conn.close()
        return 0

    if args.repair:
        print(f"\n빠진 데이터 복구 중...")
        D.repair(tickers, conn=conn)
        conn.close()
        return 0

    if not args.no_sync:
        print(f"\n시세 동기화 ({len(tickers)}종목)...")
        D.sync(tickers, conn=conn)

    if args.setup or args.refresh_meta:
        print(f"\n메타데이터 수집 (섹터/시총)...")
        n = D.fetch_meta(tickers, conn=conn)
        print(f"  {n}종목 완료")

    print(f"\n스크리닝...")
    cands, stats = S.run(uni, conn)
    print()
    print(S.format_report(cands, stats))

    deep = {}
    if args.deep and not cands.empty:
        top = cands.head(args.deep_top)["ticker"].tolist()
        print(f"\n④ 악재 스크리닝 ({len(top)}종목 — EPS·마진·실적일·기관)...")
        deep = F.analyze_many(top)
        order = {"A": 0, "B": 1, "?": 2, "C": 3}
        cands["grade"] = cands["ticker"].map(
            lambda t: deep.get(t, {}).get("grade", ""))
        cands["fund_verdict"] = cands["ticker"].map(
            lambda t: deep.get(t, {}).get("verdict", ""))
        cands["fin_score"] = cands["ticker"].map(
            lambda t: deep.get(t, {}).get("fin_score"))
        cands["fin_grade"] = cands["ticker"].map(
            lambda t: deep.get(t, {}).get("fin_grade"))

        # ★ 순위의 주 기준은 재무다. 차트 점수는 동점일 때의 보조 기준.
        #   (차트는 '언제 사느냐'를 말할 뿐, '무엇을 사느냐'는 재무가 정한다)
        cands["_veto"] = cands["grade"].map(lambda g: order.get(g, 2))
        cands["_fin"] = cands["fin_score"].fillna(-1)
        cands = (cands.sort_values(["_veto", "_fin", "score"],
                                   ascending=[True, False, False])
                 .drop(columns=["_veto", "_fin"])
                 .reset_index(drop=True))

        n = {g: int((cands["grade"] == g).sum()) for g in ("A", "B", "C", "?")}
        print(f"  통과 {n['A']} / 조건부·주의 {n['B']} / 차단 {n['C']} / 확인불가 {n['?']}")
        scored = cands["fin_score"].dropna()
        if len(scored):
            print(f"  재무 점수 — 최고 {scored.max():.0f} / 중앙값 {scored.median():.0f} "
                  f"/ 최저 {scored.min():.0f}")

    # ⑤ 13F — 기관 보유 확인 (진입 필터로만 사용)
    f13 = {}
    if args.f13 and not cands.empty:
        print(f"\n⑤ 13F 기관 보유 확인 (SEC 조회 {len(C.MANAGERS)}곳)...")
        try:
            col = T13.collect(verbose=True)
            agg, meta = col["holdings"], col["meta"]
            print(f"  {len(meta['ok'])}곳 성공 / {len(meta['failed'])}곳 실패, "
                  f"종목 {meta['universe_size']}개 집계")

            if not meta["ok"]:
                # 한 곳도 못 읽었으면 '보유 기관 없음'이 아니라 '조회 실패'다.
                # 이걸 구분하지 않으면 리포트가 사실과 다른 말을 하게 된다.
                codes = sorted(set(meta.get("errors", []))) or ["원인 불명"]
                note = f"SEC 조회 실패 ({', '.join(codes)}) — py run.py --f13-test 로 점검"
                print(f"  ⚠ {note}")
                f13 = {str(r['ticker']): {"available": False, "reason": note}
                       for _, r in cands.iterrows()}
            else:
                for _, r in cands.iterrows():
                    t = str(r["ticker"])
                    f13[t] = T13.match(agg, t, str(r.get("name") or t),
                                       D.load(conn, t))
                hit = sum(1 for v in f13.values() if v.get("available"))
                print(f"  후보 {len(f13)}종목 중 {hit}종목 매칭")
        except Exception as e:
            note = f"SEC 조회 실패 ({type(e).__name__}) — py run.py --f13-test 로 점검"
            print(f"  ⚠ {note}")
            f13 = {str(r['ticker']): {"available": False, "reason": note}
                   for _, r in cands.iterrows()}

    # 진입 사다리 — 후보별 분할 매수 계획
    ladders = {}
    if not cands.empty:
        for _, r in cands.iterrows():
            t = str(r["ticker"])
            df = D.load(conn, t)
            if df.empty:
                continue
            ladders[t] = L.build_ladder(
                df, chart_score=float(r.get("score") or 0),
                fin_score=(deep.get(t) or {}).get("fin_score"),
                in_channel=bool(r.get("in_channel", False)))
        cands["first_weight"] = cands["ticker"].map(
            lambda t: (ladders.get(t) or {}).get("first_weight"))
        cands["invalidation"] = cands["ticker"].map(
            lambda t: (ladders.get(t) or {}).get("invalidation"))

    if not cands.empty:
        p = safe_write(args.csv,
                       lambda f: cands.to_csv(f, index=False, encoding="utf-8-sig"))
        print(f"\n후보 저장: {p}")

    if not args.no_report:
        path = safe_write(args.report,
                          lambda f: R.build(cands, stats, out=f, deep=deep,
                                            ladders=ladders, f13=f13,
                                            open_browser=False))
        print(f"리포트 생성: {os.path.abspath(path)}")
        if not args.no_open:
            try:
                webbrowser.open("file://" + os.path.abspath(path))
                print("  (브라우저가 자동으로 열립니다)")
            except Exception:
                pass

    # 알림 — 변화가 있을 때만
    if (args.notify or args.notify_dry) and not cands.empty:
        print("\n알림 확인...")
        res = N.notify(cands, deep, ladders,
                       report_url=args.report_url or None,
                       dry_run=args.notify_dry)
        if res.get("first_run"):
            print(f"  첫 실행 — 후보 {res['baseline']}종목을 기준점으로 저장했습니다.")
            print("  내일부터 '변화'가 생기면 알림이 갑니다.")
        elif res["events"] == 0:
            print("  변화 없음 — 알림 없음")
        else:
            print(f"  변화 {res['events']}건 ({', '.join(res['kinds'])}) "
                  f"— 전송 {res['sent']}건")

    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
