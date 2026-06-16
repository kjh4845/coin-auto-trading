# BTCUSDT Auto Trading Bot Implementation Workflow

## 1. 목적

이 문서는 구현 순서를 잘게 나누고, 각 단계마다:

- 무엇을 만들지
- 무엇이 끝나야 다음 단계로 가는지
- 어떻게 검증할지
- 무엇을 산출물로 남길지

를 한 파일에서 관리하기 위한 실행용 워크플로다.

핵심 원칙:

- 한 번에 전체를 만들지 않는다.
- 이전 단계의 체크리스트를 통과해야 다음 단계로 간다.
- AI는 뒤에 붙인다.
- 실거래는 더 뒤에 붙인다.
- `signal은 contract price`, `risk는 mark price` 규칙을 전 단계에서 유지한다.

## 2. 고정 전제

아래는 구현 중 바뀌면 안 되는 현재 기준이다.

- symbol: `BTCUSDT`
- market: Binance USD-M Futures
- mode: `one-way`
- margin: `isolated`
- live start leverage: `2x`
- leverage cap: `10x`
- one position at a time
- hard stop basis: `entry-time fixed initial margin`
- hard stop trigger: `mark price unrealized PnL <= -5% of entry-time fixed initial margin`
- exchange hard stop trigger: `workingType=MARK_PRICE`
- emergency fallback exit: reduce-only `MARKET`
- signal source: contract-price candles
- ATR source: contract-price candles
- ATR exit: local strategy exit, not primary exchange safety stop
- split-entry: disabled in approved live baseline
- immediate live auto-promotion: disabled

## 3. 운영 규칙

- 모든 단계는 체크리스트 기반으로 완료 처리한다.
- 체크리스트 항목은 코드, 파일, 테스트, 로그 중 하나로 증명 가능해야 한다.
- 구현이 끝나도 `통과 기준`을 만족하지 못하면 완료로 치지 않는다.
- 실험 결과는 `policy/experiment/`에만 둔다.
- 승인된 라이브 정책은 `policy/approved_live/`에만 둔다.
- 거래 로그는 `docs/schemas/trade_log_v1.schema.json`을 따라야 한다.

## 4. 전역 완료 기준

아래 항목은 전체 프로젝트에서 계속 유지돼야 한다.

- [ ] 하드 손절 로직은 `mark price` 기준으로만 평가된다.
- [ ] `entry_initial_margin_fixed`는 체결 시점에 고정 저장된다.
- [ ] 거래 로그에는 버전 메타데이터가 항상 들어간다.
- [ ] `rule_only` 거래도 `model_base="rule_only"`로 기록된다.
- [ ] `dataset_version`은 항상 존재하며, 미승격 거래는 `null`이다.
- [ ] ATR 청산은 하드 손절 계약을 덮어쓰지 않는다.
- [ ] 리뷰/재학습 결과는 즉시 라이브 반영되지 않는다.

## 5. 단계별 워크플로

### Phase 0. 프로젝트 스캐폴드

목표:

- 구현 가능한 기본 폴더 구조와 실행 진입점을 만든다.

체크리스트:

- [x] `src/` 폴더 생성
- [x] `config/` 폴더 생성
- [x] `policy/approved_live/`, `policy/experiment/`, `policy/training_candidates/`, `policy/rollback/` 생성
- [x] `data/raw/`, `data/features/`, `data/backtests/`, `data/runtime/` 생성
- [x] `tests/` 폴더 생성
- [x] `.env.example` 생성
- [x] Python 패키지 진입점 생성

통과 기준:

- 프로젝트가 모듈로 import 가능해야 한다.
- 설정 파일이 비어 있어도 기본 로딩 경로가 깨지지 않아야 한다.

검증 방법:

- `python -m <package> --help` 실행 가능
- 기본 설정 로더 단위 테스트 통과

산출물:

- 디렉토리 구조
- 설정 로더
- 기본 CLI 진입점

선행조건:

- 없음

### Phase 1. 도메인 모델과 스키마 연결

목표:

- 거래, 포지션, 주문, 정책 버전 구조를 코드로 고정한다.

체크리스트:

- [x] `TradeRecord` 모델 생성
- [x] `PositionState` 모델 생성
- [x] `OrderIntent` 모델 생성
- [x] `PolicyVersionInfo` 모델 생성
- [x] `trade_log_v1.schema.json` 검증기 연결
- [x] `model_base="rule_only"` 강제 규칙 구현
- [x] `dataset_version=null` 기본값 규칙 구현

통과 기준:

- 샘플 거래 로그를 생성하면 스키마 검증을 통과해야 한다.
- AI 미사용 거래는 `model_base`를 비우지 않고 `rule_only`가 들어가야 한다.

검증 방법:

- 스키마 검증 테스트
- rule-only 거래 생성 테스트
- dataset_version 기본값 테스트

산출물:

- 도메인 모델 코드
- 스키마 검증 코드
- 예제 trade record fixture

선행조건:

- Phase 0 완료

### Phase 2. 과거 데이터 수집기

목표:

- 백테스트에 필요한 과거 데이터를 자동 수집한다.

체크리스트:

- [x] contract-price OHLCV downloader 구현
- [x] mark-price candle downloader 구현
- [x] funding rate downloader 구현
- [x] open interest downloader 구현
- [x] parquet 저장 로직 구현
- [x] 기간 단위 backfill 실행기 구현

통과 기준:

- `BTCUSDT` 기준으로 `1m`, `3m`, `15m`, `1h` contract-price candles 저장 가능
- `1m` mark-price candles 저장 가능
- funding/open interest 저장 가능

검증 방법:

- downloader integration test
- parquet file 생성 확인
- 중복 실행 시 append 또는 dedupe 정책 확인

산출물:

- historical downloader
- raw parquet dataset

선행조건:

- Phase 1 완료

### Phase 3. 실시간 mark price 기록기

목표:

- intrabar 손절 검증을 위한 실시간 mark price 히스토리를 저장한다.

체크리스트:

- [x] WebSocket mark price recorder 구현
- [x] 재연결 처리 구현
- [x] 파일 또는 SQLite append 저장 구현
- [x] 끊김 구간 감지 구현
- [x] 런타임 헬스 로그 구현

통과 기준:

- `BTCUSDT` mark price를 지속 수집 가능
- 연결이 끊겨도 자동 복구 가능
- 데이터 누락 구간을 로그로 남김

검증 방법:

- 재연결 시뮬레이션
- 5분 이상 수집 후 저장 확인
- 누락 구간 경고 테스트

산출물:

- live mark price recorder
- runtime mark price store

선행조건:

- Phase 2 완료

### Phase 4. 피처 엔진

목표:

- 전략과 리뷰, AI 입력에 필요한 피처를 일관되게 계산한다.

체크리스트:

- [x] contract-price 기반 ATR 구현
- [x] EMA, RSI 구현
- [x] VWAP, swing level 계산 구현
- [x] multi-timeframe feature merge 구현
- [x] feature schema version 관리 구현

통과 기준:

- 같은 입력에 대해 항상 같은 피처가 생성돼야 한다.
- 피처 결과에 `feature_schema_version`이 붙어야 한다.

검증 방법:

- indicator unit test
- fixture 기반 deterministic test

산출물:

- feature builder
- versioned feature snapshot

선행조건:

- Phase 2 완료

### Phase 5. 하드 손절 엔진

목표:

- 문서에 정의된 하드 손절 계약을 코드로 정확히 고정한다.

체크리스트:

- [x] `entry_initial_margin_fixed` 계산기 구현
- [x] `hard_stop_loss_usdt` 계산기 구현
- [x] mark-price unrealized PnL 계산기 구현
- [x] 손절 트리거 evaluator 구현
- [x] `workingType=MARK_PRICE` 주문 파라미터 생성기 구현
- [x] 로컬 reduce-only market fail-safe trigger 구현

통과 기준:

- 같은 체결값과 mark price 입력에 대해 손절 여부가 일관되게 판정돼야 한다.
- 거래소 주문용 손절 파라미터와 로컬 fail-safe 판정이 같은 계약을 가리켜야 한다.

검증 방법:

- calculator unit test
- edge-case test: leverage up, partial fill, slippage assumption
- stop trigger parity test

산출물:

- hard stop module
- stop order payload builder

선행조건:

- Phase 1, 3, 4 완료

### Phase 6. 룰 기반 시그널 엔진

목표:

- AI 없이도 작동하는 베이스 전략을 만든다.

체크리스트:

- [x] trend-following pullback long rule 구현
- [x] short rule 구현
- [x] ATR volatility filter 구현
- [x] cooldown/no-entry filter 구현
- [x] signal reason code 출력 구현

통과 기준:

- 입력 데이터에 대해 `LONG`, `SHORT`, `NO_TRADE`가 재현 가능하게 출력된다.
- 모든 신호에는 reason code가 붙는다.

검증 방법:

- fixture 전략 테스트
- hand-crafted scenario test

산출물:

- deterministic signal engine

선행조건:

- Phase 4, 5 완료

### Phase 7. 백테스터

목표:

- 현재 전략/손절 정의가 과거 데이터에서 어떻게 동작하는지 검증한다.

체크리스트:

- [x] hybrid replay engine 구현
- [x] signal은 execution timeframe contract-price 기준 적용
- [x] 손절은 lower-timeframe mark-price breach 기준 적용
- [x] slippage/fee model 구현
- [x] ATR local exit 구현
- [x] time stop 구현
- [x] trade log export 구현
- [x] backtest decision report 생성 구현
- [x] 채택/폐기 근거 explanation 생성 구현

통과 기준:

- 거래 하나마다 `trade_log_v1.schema.json`을 만족하는 로그를 생성해야 한다.
- intrabar breach가 발생한 경우 손절이 close-only보다 보수적으로 적용돼야 한다.
- 백테스트 결과물에는 `왜 이 전략/파라미터를 유지 또는 폐기하는지`가 reason code와 지표로 설명돼야 한다.

검증 방법:

- replay integration test
- slippage sensitivity test
- schema validation on exported trade logs
- explanation report snapshot test

산출물:

- backtest runner
- backtest result report
- versioned trade logs
- decision explanation report

선행조건:

- Phase 1~6 완료

### Phase 8. 손실 리뷰 파이프라인

목표:

- 손실 원인을 자동 태깅하고 재학습 후보를 추출한다.

체크리스트:

- [ ] deterministic loss tagger 구현
- [ ] reason code reporter 구현
- [ ] grouped failure summary 구현
- [ ] candidate dataset selector 구현
- [ ] `dataset_version` lineage writer 구현

통과 기준:

- 손실 거래를 읽어 최소 1개 이상의 reason code를 산출해야 한다.
- dataset 승격 시 `dataset_version`이 null에서 값으로 바뀌어야 한다.

검증 방법:

- review pipeline integration test
- dataset lineage test

산출물:

- loss review report
- training candidate dataset manifest

선행조건:

- Phase 7 완료

### Phase 9. 페이퍼 런타임

목표:

- 실시간에서 전략+리스크가 실제처럼 돌아가는지 주문 없이 검증한다.

체크리스트:

- [x] live data subscription 구현
- [x] signal engine 연결
- [x] hard stop evaluator 연결
- [x] ATR local exit 연결
- [x] paper position manager 구현
- [x] trade log 저장 구현

통과 기준:

- 실시간 입력에서 paper trade가 생성되고 종료까지 추적 가능해야 한다.
- rule-only 기준 로그가 스키마를 통과해야 한다.

검증 방법:

- live dry-run
- schema validation
- restart recovery test

산출물:

- paper runtime
- paper trade logs

선행조건:

- Phase 7, 8 완료

### Phase 10. Testnet 실행 엔진

목표:

- 실제 주문 경로와 보호 스탑 동작을 테스트넷에서 검증한다.

체크리스트:

- [x] Binance testnet client 구현
- [x] entry order 구현
- [x] `STOP_MARKET reduce-only` hard stop 구현
- [x] `workingType=MARK_PRICE` 확인
- [x] local fail-safe market exit 구현
- [x] order/position reconciliation 구현

통과 기준:

- entry -> hard stop placement -> exit 흐름이 일관되게 동작해야 한다.
- mark price breach 시 거래소 또는 로컬 fail-safe 둘 중 하나가 반드시 포지션을 정리해야 한다.

검증 방법:

- testnet end-to-end run
- restart reconciliation test
- forced breach scenario test

현재 상태 메모:

- mock transport 기반 검증은 완료
- public `serverTime` testnet 호출은 확인 완료
- testnet CLI 진입점과 SQLite runtime state / incident log는 구현 완료
- restart reconciliation 경로는 저장된 runtime state 기준으로 구현 및 테스트 완료
- 실제 signed end-to-end testnet 주문 검증은 API 키가 없어서 아직 미실행

산출물:

- testnet execution runtime
- execution incident logs

선행조건:

- Phase 5, 7, 9 완료

### Phase 11. SuperGemma4 통합

목표:

- 로컬 추론을 보조 게이트로 붙인다.

체크리스트:

- [x] local inference adapter 구현
- [x] JSON output parser 구현
- [x] invalid output fallback 구현
- [x] regime gate 연결
- [x] setup quality filter 연결
- [x] rule-only vs rule-plus-AI 비교 리포트 구현

통과 기준:

- AI 출력 실패 시 시스템이 rule-only로 안전하게 복귀해야 한다.
- AI 개입 여부가 trade log에 기록돼야 한다.

검증 방법:

- malformed output test
- latency budget test
- A/B backtest comparison

산출물:

- local inference adapter
- AI comparison report

선행조건:

- Phase 8, 9 완료

### Phase 12. 승격과 운영

목표:

- 리뷰/재학습 결과를 즉시 반영하지 않고 안전하게 승격한다.

체크리스트:

- [ ] `policy/experiment/` 산출물 관리 구현
- [ ] `policy/approved_live/` 승격 규칙 구현
- [ ] `policy/rollback/` 롤백 포인터 구현
- [ ] shadow/paper holdout gate 구현
- [ ] weekly promotion gate 구현

통과 기준:

- 실험 결과가 승인 폴더를 직접 덮어쓰지 못해야 한다.
- 새 모델/새 전략은 지연 승격 규칙 없이는 라이브 진입이 불가능해야 한다.

검증 방법:

- promotion guard test
- rollback test
- policy overwrite prevention test

산출물:

- promotion manager
- rollback procedure

선행조건:

- Phase 10, 11 완료

## 6. 지금 당장 시작할 범위

첫 개발 사이클은 여기까지만 한다.

- [x] Phase 0 완료
- [x] Phase 1 완료
- [x] Phase 2 완료

이 3개가 끝나야:

- 데이터가 쌓이고
- 도메인 모델이 고정되고
- 이후 백테스트 구현이 가능해진다

## 7. 이 체크리스트는 모두 통과 가능한가

현재 기준으로는 `예`다.

이유:

- 선행조건이 순환 참조가 아니다.
- 각 단계에 명확한 산출물과 검증 방법이 있다.
- 실거래에 들어가기 전 단계별로 중단 가능한 구조다.
- 문서상 불변 코어 룰과 구현 순서가 충돌하지 않는다.

단, 아래가 전제다:

- Binance API 접근이 정상이어야 한다.
- mark price 데이터 적재가 안정적이어야 한다.
- 단계별 테스트를 생략하지 않아야 한다.

## 8. 구현 시작 규칙

앞으로 구현은 아래 원칙으로 진행한다.

- 현재 단계 외 기능은 건드리지 않는다.
- 각 단계가 끝나면 체크리스트를 갱신한다.
- 통과 기준을 만족하지 못하면 다음 단계로 가지 않는다.
- 작은 단위 PR 또는 변경 세트처럼 작업한다.
