# Deploy — 24/7 자동화 가이드

이 문서는 24시간 동작하는 머신에 quant 프로젝트를 띄우고
**일일 모니터링 + 매월 자동 리밸런싱 + Discord 알림**을 설정하는 절차입니다.

---

## 1. 머신에 quant 배치

```bash
# 적당한 위치에 클론
git clone git@github.com:ToySin/quant.git ~/quant
cd ~/quant

# Python 3.11+ venv 만들기
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'

# 테스트로 동작 확인
pytest -q
```

## 2. `.env` 두 개 만들기

quant 코드는 `~/repositories/assisthub-ws-quant/.env`를 우선 읽고, 없으면
`~/quant/.env`(현재 레포 내부)를 읽습니다. 24/7 머신에는 워크스페이스가
없을 테니 `~/quant/.env`로 만들면 편해요.

```bash
cat > ~/quant/.env <<'EOF'
ALPACA_PAPER_KEY=PK...
ALPACA_PAPER_SECRET=...
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
EOF
chmod 600 ~/quant/.env   # 권한 잠그기
```

⚠️ 절대 commit 금지. `.env`는 이미 `.gitignore`에 있음.

## 3. Discord webhook 만들기

1. Discord에서 받을 채널 선택 → 톱니바퀴 (채널 편집)
2. **Integrations → Webhooks → New Webhook**
3. 이름 / 아이콘 설정 (예: "Quant Bot")
4. **Copy Webhook URL** → `.env`의 `DISCORD_WEBHOOK_URL=` 뒤에 붙여넣기

## 4. 동작 확인

```bash
cd ~/quant

# Alpaca 연결
.venv/bin/python -m scripts.check_alpaca

# Discord 한 번 쏴보기 (no-post 빼고)
.venv/bin/python -m scripts.daily_monitor

# Auto-rebal dry-run
.venv/bin/python -m scripts.auto_monthly_rebalance --dry-run --force
```

## 5. cron 설정

```bash
crontab -e
```

아래 내용을 추가 (경로는 본인 실제 경로로 수정):

```cron
# Quant paper trading automation
SHELL=/bin/bash
PATH=/usr/local/bin:/usr/bin:/bin

# Daily monitor: every weekday at 6am KST (= ~5pm ET, after US market close)
0 6 * * 2-6 cd /home/USER/quant && .venv/bin/python -m scripts.daily_monitor >> /tmp/quant_monitor.log 2>&1

# Monthly auto-rebalance: every weekday at 11pm KST (= ~10am ET, after market open).
# State file makes this idempotent — only the first weekday of the month actually rebalances.
0 23 * * 1-5 cd /home/USER/quant && .venv/bin/python -m scripts.auto_monthly_rebalance >> /tmp/quant_rebal.log 2>&1
```

### 시간대 설명

cron은 머신의 로컬 타임존을 따릅니다. 위 예시는 KST 머신 기준:
- `0 6 * * 2-6`: 화-토 06:00 KST = 월-금 17:00 ET (US 시장 마감 직후)
- `0 23 * * 1-5`: 월-금 23:00 KST = 월-금 10:00 ET (US 시장 개장 직후)

머신 시간대 확인: `date +%Z` 또는 `timedatectl`

## 6. 모니터링

### 로그
```bash
tail -f /tmp/quant_monitor.log
tail -f /tmp/quant_rebal.log
```

### 상태 파일
```bash
cat ~/quant/data/state/portfolio.json
```
`last_rebalance_month`, `peak_equity`, `history` 확인.

### Discord
- 매일 아침 daily snapshot (equity / PnL / 포지션 수 / drawdown)
- 매월 1주차 weekday 리밸런싱 알림 (체결된 trade 리스트)
- ALERT 색깔: 양수 = 초록, 음수 = 빨강

### 알림 임계값 조정
`scripts/daily_monitor.py` 상단에서 변경:
- `ALERT_MOVE_THRESHOLD` — 일일 변동 ±몇 %부터 alert
- `ALERT_DRAWDOWN_THRESHOLD` — 고점 대비 -몇 %부터 alert

## 7. 운용 중 확인

### 정상 동작
- 매일 아침 Discord에 snapshot 들어와야 함
- 매월 첫 영업일에 Discord에 rebalance 알림 들어와야 함
- 로그에 에러 없어야 함

### 문제 시
```bash
# 수동 실행
.venv/bin/python -m scripts.daily_monitor          # 즉시 snapshot
.venv/bin/python -m scripts.auto_monthly_rebalance --dry-run --force   # 강제 실행 (체결 X)

# 상태 리셋 (다시 이번 달 리밸런싱하고 싶을 때)
rm ~/quant/data/state/portfolio.json
```

### 장기 (코드 업데이트)
```bash
cd ~/quant
git pull
.venv/bin/pip install -e '.[dev]'   # 의존성 변경 시
```

## 8. 6개월 후 평가

`data/state/portfolio.json`의 `history`로 equity 변화 시계열 추출 가능:

```python
import json, pandas as pd
data = json.load(open("data/state/portfolio.json"))
df = pd.DataFrame(data["history"])
df["date"] = pd.to_datetime(df["date"])
df["return"] = df["equity"].pct_change()

# 누적 수익률
total = df["equity"].iloc[-1] / df["equity"].iloc[0] - 1
# 일간 수익률 통계
sharpe = df["return"].mean() / df["return"].std() * (252 ** 0.5)
```

이걸 backtest에서 예상한 CAGR/Sharpe와 비교 → 슬리피지·실행지연 영향 측정.
