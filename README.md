# coin-auto-trading

Binance Futures 기반 자동매매 실험, 백테스트, 테스트넷/메인넷 런타임, 운영 대시보드를 하나로 묶은 Python 프로젝트입니다.

이 저장소는 코드와 문서 중심으로 관리합니다. 실제 `.env`, 거래소 API 키, SQLite 런타임 DB, 로그, parquet 시장 데이터, 백테스트 산출물은 Git에 올리지 않도록 `.gitignore`에 제외되어 있습니다.

## 주요 기능

- Binance Futures 데이터 수집
  - contract kline, mark price kline, funding rate, open interest history 수집
  - parquet 저장과 백필 지원
- 룰 기반 전략 엔진
  - 단일 프로필 전략
  - `best_pair_v1` 프로필 기반 전략
  - 다중 심볼 우선순위 메인넷 사이클
  - LONG/SHORT 방향, funding, taker buy ratio, EMA spread, volume 조건
- 백테스트
  - contract candle, mark price, funding rate 기반 replay
  - 손익, drawdown, win rate, profit factor, trade log 산출
  - exit policy 비교와 조건 조합 탐색 도구 포함
- 테스트넷/메인넷 실행
  - 거래소 설정 확인 및 보정
  - 진입 주문과 보호 stop 번들 검증
  - 로컬 expected state와 거래소 remote state reconciliation
  - 포지션 관리, ATR trail, fixed take-profit, time stop
- 리스크 관리
  - 일일 손실 한도
  - 연속 손실 제한
  - 손실 후 cooldown
  - 수동 pause/resume lockout
  - stale stream, orphan position, missing protective stop 감지
- 로컬 AI 게이트
  - OpenAI 호환 `/v1/chat/completions` 형태의 로컬 모델 서버
  - 룰 기반 진입 신호를 AI가 보조 검토
  - `AI_GATE_ENABLED`로 선택 사용
- 운영 대시보드
  - runtime status, service status, account USDT, equity history
  - remote exposure, risk lockout, recent incidents, recent trades
  - 수동 pause/resume, remote state refresh, monitor once, cycle once

## 주의

이 프로젝트는 자동매매 시스템 구현체입니다. 메인넷 명령은 실제 주문과 손실을 만들 수 있습니다.

- `.env`에 들어가는 API 키는 절대 커밋하지 마세요.
- 처음에는 테스트넷에서만 검증하세요.
- 메인넷 실행 전 `--confirm-mainnet-live`가 필요한 명령만 사용하세요.
- `entry_margin_fraction`, leverage, stop policy, daily loss cap을 보수적으로 설정하세요.
- 이 저장소의 코드는 투자 조언이 아니며, 사용 책임은 운영자에게 있습니다.

## 프로젝트 구조

```text
src/ai_auto_trading/
  ai/                 로컬 AI 추론 클라이언트와 로컬 모델 서버
  backtest/           백테스트 replay와 profile runner
  data/               Binance historical downloader, mark price recorder
  execution/          Binance Futures testnet/mainnet 클라이언트와 주문 실행 엔진
  features/           candle feature와 indicator snapshot
  risk/               hard stop 등 리스크 보조 로직
  runtime/            dashboard, orchestrator, runtime state, streams, position manager
  strategy/           rule-based strategy, runtime profile, trade management
  cli.py              전체 CLI 진입점
  settings.py         `.env` 및 환경변수 기반 설정

tests/                단위 테스트와 fixture
tools/                탐색/비교/운영 보조 스크립트
docs/                 설계 문서와 스키마
data/                 로컬 실행 산출물 전용, Git 제외
```

## 설치

Python 3.9 이상이 필요합니다.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -e .
```

대시보드 smoke test를 실행하려면 Node 의존성도 설치합니다.

```bash
npm ci
```

로컬 AI 서버를 직접 띄우는 경우에는 별도 환경에 `transformers`, `torch`, 모델 파일이 필요합니다. 기본 런타임과 테스트에는 로컬 AI 서버가 필수는 아닙니다.

## 초기 설정

샘플 환경 파일을 복사해서 값을 채웁니다.

```bash
cp .env.example .env
```

중요 설정:

```text
APP_ENV=development
LOG_LEVEL=INFO
STRATEGY_MODE=best_pair_v1

BINANCE_TESTNET_API_KEY=
BINANCE_TESTNET_API_SECRET=

BINANCE_API_KEY=
BINANCE_API_SECRET=

TRADING_SYMBOL=BTCUSDT
LIVE_START_LEVERAGE=5
SYSTEM_LEVERAGE_CAP=10
MAX_DAILY_LOSS_R=3.0
MAX_CONSECUTIVE_LOSSES=3
COOLDOWN_AFTER_LOSS_MINUTES=30

ALLOW_LONG_ENTRIES=false
ALLOW_SHORT_ENTRIES=true
AI_GATE_ENABLED=true
```

디렉터리 구조 확인:

```bash
ai-auto-trading check-layout
```

또는 editable 설치 전에는 모듈 실행으로도 가능합니다.

```bash
python3 -m ai_auto_trading.cli check-layout
```

## 테스트

전체 단위 테스트:

```bash
python3 -m unittest discover -s tests
```

대시보드 관련 테스트:

```bash
python3 -m unittest tests.test_dashboard
```

설정 확인:

```bash
ai-auto-trading show-config
```

## 데이터 수집

최근 데이터를 한 번 가져오기:

```bash
ai-auto-trading fetch-historical \
  --dataset contract_klines \
  --symbol BTCUSDT \
  --interval 1m \
  --limit 500 \
  --output data/raw/binance/contract_klines/btcusdt/latest.parquet
```

기간 백필:

```bash
ai-auto-trading backfill-historical \
  --dataset mark_price_klines \
  --symbol BTCUSDT \
  --interval 1m \
  --start-time 1704067200000 \
  --end-time 1704153600000 \
  --output data/raw/binance/mark_price_klines/btcusdt/1m_sample.parquet
```

Funding rate:

```bash
ai-auto-trading backfill-historical \
  --dataset funding_rate \
  --symbol BTCUSDT \
  --start-time 1704067200000 \
  --end-time 1704153600000 \
  --output data/raw/binance/funding_rate/btcusdt/sample.parquet
```

## 백테스트

`best_pair_v1`는 funding parquet이 필요합니다.

```bash
ai-auto-trading backtest-run \
  --symbol BTCUSDT \
  --strategy-mode best_pair_v1 \
  --contract-parquet data/raw/binance/contract_klines/btcusdt/1m_5y_v1.parquet \
  --mark-parquet data/raw/binance/mark_price_klines/btcusdt/1m_5y_v1.parquet \
  --funding-parquet data/raw/binance/funding_rate/btcusdt/5y_v1.parquet \
  --output-dir data/backtests/btcusdt-best-pair
```

단일 프로필 모드 예:

```bash
ai-auto-trading backtest-run \
  --symbol BTCUSDT \
  --strategy-mode single_profile \
  --contract-parquet data/raw/binance/contract_klines/btcusdt/1m_sample.parquet \
  --mark-parquet data/raw/binance/mark_price_klines/btcusdt/1m_sample.parquet \
  --execution-timeframe 3m \
  --micro-timeframe 5m \
  --confirmation-timeframe 15m \
  --macro-timeframe 1h \
  --output-dir data/backtests/btcusdt-single-profile
```

백테스트 산출물은 `data/backtests/` 아래에 저장되며 Git에는 포함하지 않습니다.

## 테스트넷 운영

테스트넷 연결 확인:

```bash
ai-auto-trading testnet-check --symbol BTCUSDT
```

거래소 계정 설정 보정:

```bash
ai-auto-trading testnet-ensure-config \
  --symbol BTCUSDT \
  --leverage 5 \
  --margin-mode ISOLATED \
  --position-mode ONE_WAY
```

테스트넷 자동 사이클 1회:

```bash
ai-auto-trading testnet-auto-cycle-once \
  --symbol BTCUSDT \
  --entry-notional-usdt 1000 \
  --candle-limit 120
```

테스트넷 루프:

```bash
ai-auto-trading testnet-auto-cycle-loop \
  --symbol BTCUSDT \
  --entry-notional-usdt 1000 \
  --interval-seconds 5
```

수동 중지/재개:

```bash
ai-auto-trading testnet-manual-pause --symbol BTCUSDT
ai-auto-trading testnet-manual-resume --symbol BTCUSDT
```

## 메인넷 운영

메인넷은 실제 주문을 낼 수 있으므로 테스트넷과 소액 검증 후 사용합니다.

상태 확인:

```bash
ai-auto-trading mainnet-check --symbol BTCUSDT
```

설정 보정:

```bash
ai-auto-trading mainnet-ensure-config \
  --symbol BTCUSDT \
  --leverage 5 \
  --margin-mode ISOLATED \
  --position-mode ONE_WAY \
  --confirm-mainnet-config
```

메인넷 자동 사이클 1회:

```bash
ai-auto-trading mainnet-auto-cycle-once \
  --symbol BTCUSDT \
  --entry-margin-fraction 0.25 \
  --candle-limit 120 \
  --confirm-mainnet-live
```

다중 심볼 우선순위 루프:

```bash
ai-auto-trading mainnet-priority-auto-cycle-loop \
  --symbols BTCUSDT,ETHUSDT,SOLUSDT \
  --entry-margin-fraction 0.25 \
  --interval-seconds 5 \
  --idle-reconcile-seconds 60 \
  --confirm-mainnet-live
```

`mainnet-auto-cycle-*` 명령에서 fixed `--entry-notional-usdt`는 비활성화되어 있으며, 사용 가능 USDT와 `entry_margin_fraction`, leverage 기반으로 sizing합니다.

## 운영 대시보드

테스트넷 대시보드:

```bash
ai-auto-trading testnet-dashboard \
  --host 127.0.0.1 \
  --port 8765 \
  --symbols BTCUSDT,ETHUSDT,SOLUSDT
```

메인넷 대시보드:

```bash
ai-auto-trading mainnet-dashboard \
  --host 127.0.0.1 \
  --port 8765 \
  --symbols BTCUSDT,ETHUSDT,SOLUSDT
```

브라우저에서 `http://127.0.0.1:8765`를 엽니다.

대시보드에는 다음 정보가 표시됩니다.

- 거래 가능 여부
- 자동 루프 실행 상태
- 런타임 상태와 stream freshness
- USDT account balance와 equity history
- remote position/order exposure
- risk lockout과 최근 손익
- Binance income 기반 실제 순손익
- 최근 거래 기록과 손실 리뷰
- runtime summary, remote state, raw JSON

Node/Playwright 기반 smoke test:

```bash
npm run dashboard:smoke
```

## 로컬 AI 게이트

AI 게이트는 룰 기반 신호를 보조 검토합니다. 필수 구성은 아니며 `.env`에서 꺼둘 수 있습니다.

```text
AI_GATE_ENABLED=false
```

로컬 서버를 사용할 경우:

```bash
ai-auto-trading local-ai-check
ai-auto-trading local-ai-bench --iterations 3 --warmup 1
```

모델 로딩 서버는 `src/ai_auto_trading/ai/local_server.py`에 구현되어 있으며, `LOCAL_MODEL_PATH`, `LOCAL_MODEL_ENDPOINT`, memory 설정을 사용합니다.

## Runtime 데이터 정책

다음 경로는 로컬 전용입니다.

```text
data/raw/       market parquet
data/features/  generated feature data
data/backtests/ backtest reports and trade logs
data/runtime/   SQLite runtime DB, launchd logs, screenshots
```

이 파일들은 계좌 상태, 주문 기록, runtime 로그, 대용량 시장 데이터가 포함될 수 있으므로 Git에 올리지 않습니다.

Git에는 다음만 포함합니다.

- source code
- tests and fixtures
- docs and schema
- tool scripts
- `.env.example`
- empty `.gitkeep` directory markers

## macOS launchd 보조 스크립트

`tools/launchd/`에는 로컬 서비스 실행을 위한 plist와 shell script가 있습니다.

- testnet/mainnet dashboard 실행
- testnet/mainnet auto-cycle loop 실행
- local AI server 실행
- service root 배포 보조

plist 내부 경로는 로컬 운영 환경에 맞게 수정해야 합니다.

## 개발 메모

- CLI 진입점: `ai-auto-trading`
- Python 패키지: `src/ai_auto_trading`
- 설정 로딩: repo root의 `.env`를 자동 로드
- 기본 runtime DB: `data/runtime/execution/testnet_execution.sqlite3`
- 메인넷 runtime DB: `data/runtime/execution/mainnet_execution.sqlite3`
- Dashboard template: `src/ai_auto_trading/runtime/dashboard_template.html`
- Trade log schema: `docs/schemas/trade_log_v1.schema.json`

## GitHub 업로드 전 체크리스트

```bash
python3 -m unittest discover -s tests
python3 -m py_compile src/ai_auto_trading/runtime/dashboard.py src/ai_auto_trading/cli.py
git status --ignored --short
```

확인해야 할 항목:

- `.env`가 untracked/ignored인지
- `data/` 산출물이 ignored인지
- `node_modules/`가 ignored인지
- SQLite, log, parquet 파일이 staged 되지 않았는지
