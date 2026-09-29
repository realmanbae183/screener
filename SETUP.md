# 세팅 안내 (코딩 몰라도 됨)

두 가지를 세팅한다. **텔레그램 알림**(10분)과 **클라우드 자동실행**(30분).
클라우드까지 하면 컴퓨터를 꺼둬도 매일 아침 결과가 나오고, 남들한테 보낼
링크도 생긴다.

---

## A. 텔레그램 알림 (10분)

### 1) 봇 만들기

텔레그램에서 **@BotFather** 를 검색해서 대화를 연다.

```
/newbot
```

- 봇 이름을 물어보면 아무거나 (예: 내 스크리너)
- 아이디를 물어보면 `_bot` 으로 끝나게 (예: jihoon_screener_bot)

그러면 이런 줄을 준다. **이게 토큰이다. 남한테 보여주면 안 된다.**

```
8123456789:AAH-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

### 2) 받을 방 만들기

혼자 받을 거면 방금 만든 봇과 대화를 열고 아무 말이나 보낸다.
친구·가족과 같이 받을 거면 **그룹방을 만들어 봇을 초대**하고 아무 말이나 보낸다.

### 3) 방 번호 알아내기

브라우저 주소창에 아래를 붙여넣는다. `<토큰>` 자리에 1)에서 받은 토큰을 넣는다.

```
https://api.telegram.org/bot<토큰>/getUpdates
```

나오는 글자 중에서 `"chat":{"id":` 뒤의 숫자가 방 번호다.
그룹방이면 `-1001234567890` 처럼 **마이너스로 시작**한다. 그대로 쓴다.

### 4) 내 컴퓨터에서 테스트

검은 창에서 (토큰과 번호는 본인 것으로):

```
set TELEGRAM_TOKEN=8123456789:AAH-xxxxx
set TELEGRAM_CHAT_ID=-1001234567890
py run.py --no-sync --deep --notify
```

알림이 오면 성공이다. 전송 없이 내용만 보려면 `--notify-dry` 를 쓴다.

---

## B. 클라우드 자동실행 (30분, 무료)

컴퓨터를 꺼둬도 매일 아침 7시에 자동으로 돌고, 결과 페이지 링크가 생긴다.

### 1) GitHub 계정 만들기

github.com 에서 가입한다.

### 2) 저장소 만들기

우측 상단 **+ → New repository**

- 이름: `screener` (아무거나)
- **Private** 선택 (비공개)
- Create repository

### 3) 파일 올리기

만들어진 저장소 화면에서 **uploading an existing file** 링크를 누르고,
이 폴더의 파일을 전부 끌어다 놓는다. 아래 두 개는 빠지기 쉬우니 확인한다.

- `.github/workflows/daily.yml` (자동실행 설정)
- `data/universe.csv` (종목 목록)

`.github` 폴더는 탐색기에서 숨겨져 보일 수 있다. 안 올라가면 저장소 화면에서
**Add file → Create new file** 로 들어가 파일명 칸에
`.github/workflows/daily.yml` 을 치고 내용을 붙여넣는다.

### 4) 비밀값 넣기

저장소 **Settings → Secrets and variables → Actions**

**Secrets** 탭에서 **New repository secret** 을 두 번:

| Name | Secret |
|---|---|
| `TELEGRAM_TOKEN` | A-1)에서 받은 토큰 |
| `TELEGRAM_CHAT_ID` | A-3)에서 받은 방 번호 |

**Variables** 탭에서 **New repository variable** 하나:

| Name | Value |
|---|---|
| `REPORT_URL` | `https://<내아이디>.github.io/screener/` |

### 5) 결과 페이지 켜기

**Settings → Pages**

- Source: `Deploy from a branch`
- Branch: `main` / 폴더는 `/docs`
- Save

몇 분 뒤 `https://<내아이디>.github.io/screener/` 로 리포트가 열린다.
이 링크를 카카오톡으로 보내면 누구나 볼 수 있다. 매일 같은 주소에
내용만 갱신되니 한 번 저장해두면 된다.

> 저장소를 Private으로 만들었는데 Pages가 안 열리면, Pages를 쓰려면
> Public이어야 하는 경우가 있다. 코드에 개인정보는 없으니 Public으로
> 바꿔도 되고 (토큰은 Secrets에 있어서 코드에 안 들어간다),
> 링크 공유가 굳이 필요 없으면 Pages는 건너뛰어도 알림은 온다.

### 6) 첫 실행

**Actions** 탭 → 왼쪽 **매일 스크리닝** → **Run workflow** 버튼.

처음엔 10년치를 받느라 20~40분 걸린다. 그 다음부터는 몇 분이면 끝난다.
초록색 체크가 뜨면 성공이다.

---

## 실행 명령 정리

```
py run.py --setup             최초 1회 (시세 10년치)
py run.py --status            데이터 보유 현황
py run.py --repair            빠진 종목만 다시 받기

py run.py --no-sync           차트만 (몇 초)
py run.py --no-sync --deep    + 재무·악재 (1~2분)
py run.py --all               전부 + 13F + 알림
py run.py --all --notify-dry  알림 내용만 미리보기
```

## 알림이 오는 경우

- 🆕 **신규 후보** — 처음 목록에 오른 종목
- 🔔 **진입가 도달** — 사다리 칸에 가격이 닿음
- ⚠️ **폐기선 이탈** — 판단 근거가 깨짐
- ➖ **후보 이탈** — 목록에서 빠짐

**아무 일 없는 날은 알림이 오지 않는다.** 그게 정상이다.
매일 목록을 보내면 일주일이면 안 보게 되기 때문에 변화만 보낸다.
