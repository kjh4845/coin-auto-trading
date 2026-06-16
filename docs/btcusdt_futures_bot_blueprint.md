# BTCUSDT Futures AI Auto Trading Bot Blueprint

## 1. Goal

Build an automated trading bot for `BTCUSDT` perpetual futures on Binance that:

- trades only `BTCUSDT`
- targets short-term trades, but not ultra-low-latency scalping
- stores as much relevant BTC market history as the data sources allow
- uses local `Gemma4` as a narrow trading assistant, not as the account owner
- enforces a hard stop-loss at `-5%` of margin used for the trade
- uses an `ATR-based trailing stop` as the main profit capture method
- supports leverage up to `10x`
- runs safely on a spare M2 MacBook Air with local inference

The first milestone is not "make profit immediately".
The first milestone is:

- build a correct historical dataset
- create a reproducible backtest
- prove the risk engine behaves correctly
- validate execution on testnet before live capital

## 2. Core Design Principles

### 2.1 The model does not own the account

The AI model can:

- classify market regime
- score setup quality
- veto low-quality entries
- suggest tighter exit posture
- summarize trade context

The AI model cannot:

- bypass leverage limits
- bypass stop-loss rules
- place orders directly without the risk manager
- widen stop-loss protection
- override kill switches

The final trading authority is:

`Signal Engine -> Risk Manager -> Execution Engine`

### 2.2 Store all relevant data, but train on structured windows

The system should backfill as much relevant BTC data as possible:

- futures contract-price OHLCV
- mark price
- funding rate
- open interest
- taker buy/sell volume
- optional spot BTC context

But `Gemma4` should not be trained on raw unlimited chart dumps.
Instead:

- store broad historical data
- convert it into structured training samples
- train on rolling market windows with labels
- run inference on compact feature snapshots

This keeps the model grounded and makes the output testable.

### 2.3 Separate signal price from risk price

Do not use one price source for every purpose.

Recommended rule:

- signal generation uses executable market data such as contract-price candles
- ATR, structure, momentum, and entry timing use contract-price candles
- hard-stop trigger, unrealized PnL, and liquidation-risk checks use mark price
- post-trade review stores both contract-price and mark-price context

Short form:

`signals on contract price, risk on mark price`

### 2.4 Start with one strategy

Version 1 should use one strategy only:

`trend-following pullback entry with volatility and regime filters`

Do not mix multiple styles in version 1:

- breakout
- mean reversion
- reversal
- fully autonomous AI entries

### 2.5 Keep the execution path deterministic

Every live order must be explainable by explicit rules.

The bot should always be able to answer:

- why the trade was entered
- why the size was chosen
- where the stop was placed
- why the stop moved
- why the trade exited

### 2.6 Core live rules must remain immutable

The review and retraining loop may suggest changes, but it may never silently weaken or erase the baseline live policy.

Immutable baseline rules for version 1:

- trade `BTCUSDT` only
- use `isolated` margin only
- never exceed the approved leverage cap
- keep one position at a time
- keep the entry-time fixed initial-margin `-5%` hard stop
- keep hard-stop trigger evaluation on mark price
- keep protective stop coverage at all times
- keep martingale and uncontrolled averaging down disabled
- keep self-modifying live strategy updates disabled

The review system may:

- propose tighter filters
- propose safer parameter ranges
- propose additional diagnostics
- propose candidate strategy versions for offline testing

The review system may not:

- delete the hard stop
- widen leverage limits automatically
- remove mandatory protective stops
- enable new strategy families directly in live mode
- activate split-entry logic without a separately approved version

## 3. Trading Style Definition

This bot is designed for:

- holding periods from several minutes to several hours
- decisions on candle close, not every tick
- execution speed that matters, but does not require HFT infrastructure

Recommended timeframe stack:

- execution timeframe: `3m`
- confirmation timeframe: `15m`
- macro intraday filter: `1h`

Optional later extension:

- add `5m` as a secondary confirmation layer

## 4. Market and Account Constraints

### 4.1 Market scope

- exchange: Binance USD-M Futures
- symbol: `BTCUSDT`
- contract type: perpetual
- one-way mode only in version 1
- only one open position at a time in version 1
- margin mode: `isolated`

### 4.2 Leverage

- maximum supported leverage in the system: `10x`
- recommended live starting leverage: `2x` to `3x`
- increase leverage only after enough testnet and small-live samples

Important consequence:

- a stop at `-5%` of initial margin becomes very tight as leverage rises
- at `10x`, a `-5%` margin-loss stop is roughly near a `0.5%` adverse move before fees and slippage

This is acceptable as a system rule only if it survives backtesting.

### 4.3 Position sizing

Position size must be driven by risk, not by available leverage.

Recommended control bands:

- risk per trade: `0.5%` to `0.75%` of account equity
- daily realized loss limit: `2%`
- weekly drawdown pause: `5%`
- one symbol only: yes
- one concurrent position only: yes

Position size is derived from:

- entry price
- initial margin budget
- leverage cap
- exchange lot size and minimum notional rules
- stop policy

## 5. Stop-Loss Policy

### 5.1 Hard stop definition

Per your latest rule, the hard stop is:

`close the trade if mark-price unrealized PnL <= -5% of the initial margin fixed at entry time`

This `-5%` value is a trigger threshold, not a guaranteed cap on final realized loss.
Realized loss can be worse after slippage, fees, gaps, or delayed execution.

Capture and freeze this value when the entry fill is confirmed:

- `filled_entry_notional = average_fill_price * filled_quantity`
- `entry_initial_margin_fixed = filled_entry_notional / leverage_at_entry`
- `hard_stop_loss_usdt = entry_initial_margin_fixed * 0.05`

Exit trigger:

- long or short position must be force-closed if `mark_price_unrealized_pnl <= -hard_stop_loss_usdt`

### 5.2 Why this must be implemented carefully

The stop is not based on:

- raw chart percentage move
- account equity drawdown
- ATR distance

It is specifically based on the `initial margin fixed at entry time`.

That means the trigger becomes tighter when leverage increases.

Do not use Binance position fields such as current `initialMargin` or current `positionInitialMargin` as the source of truth for this rule.
Those fields are documented relative to the current mark price, not the frozen entry-time margin contract you want to enforce.

### 5.3 Live implementation rule

Use a two-layer stop structure:

1. `Exchange-side protective stop`
- place a `STOP_MARKET` reduce-only protective stop immediately after entry
- convert the fixed-margin hard-stop rule into a price level at order time
- set the conditional trigger to `workingType=MARK_PRICE`

2. `Local fail-safe check`
- on every execution candle close and on major account updates
- recalculate mark-price unrealized PnL against the frozen `-5%` margin-loss rule
- if triggered, force close even if the stop order is missing or stale
- if mark-price loss is already beyond the threshold, send a reduce-only `MARKET` exit immediately as the last-resort safety action

### 5.4 Additional stop-related guardrails

- if the protective stop cannot be placed, flatten immediately
- if the protective stop exists but is not executed after the local mark-price breach is confirmed, flatten with a market order
- if local state and exchange state differ, pause new entries
- if slippage on exits exceeds threshold repeatedly, reduce size or stop trading

## 6. Take-Profit and Exit Policy

### 6.1 Primary profit-taking method

Per your latest rule, profit-taking should be based on an `ATR trailing stop`.

The main exit logic is:

- enter the trade with a hard loss stop
- let profitable trades run
- ratchet the stop using ATR as the trade moves in favor

Recommended handling:

- the exchange-side protective stop belongs to the mark-price hard-stop contract
- ATR trailing is strategy-driven exit logic, not the primary exchange safety stop
- when ATR or time-stop logic says to exit, the local engine should send a reduce-only exit order

### 6.2 ATR trailing stop definition

Recommended version 1 formula:

- ATR source: `ATR(14)` on contract-price candles from the execution timeframe
- default multiplier: `2.5`

Long trailing stop:

- `trail_stop = highest_high_since_entry - ATR(14) * 2.5`

Short trailing stop:

- `trail_stop = lowest_low_since_entry + ATR(14) * 2.5`

Rules:

- update only on candle close
- the trailing stop may tighten, but never loosen
- ATR logic is signal-side logic and should not replace the mark-price hard-stop trigger
- by default, ATR exit is executed as a local reduce-only `MARKET` exit on candle close
- an optional later optimization is a short-timeout reduce-only limit exit with market fallback

### 6.3 Exit hierarchy

Exit priority should be:

1. exchange or local hard stop from `-5%` margin-loss rule
2. local ATR trailing exit
3. time stop
4. AI recommendation to tighten or exit

### 6.4 Additional exit rules worth adding

Recommended additions:

- `break-even rule`: once trade reaches `+1 ATR` in favor, stop can move to entry or slightly positive after fees
- `time stop`: if the trade does not progress within `N` candles, exit or reduce
- `session stop`: do not hold if the bot enters a restricted maintenance or error state

Recommended version 1 defaults:

- break-even enabled: yes
- time stop enabled: yes
- fixed take-profit target: no

### 6.5 Scale-in and partial exit rules

Recommended live default:

- `scale-in disabled` in the first live version unless backtests prove it adds edge
- `scale-in disabled` in the approved live baseline while the hard stop is defined from entry-time fixed initial margin

If scale-in is enabled later, use these rules:

- maximum entry tranches: `2` or `3`
- never add to a losing position outside a pre-defined entry zone
- never add if the original invalidation has already been broken
- total position risk after all tranches must still respect the account risk cap
- no martingale and no uncontrolled averaging down

Additional accounting rule:

- each tranche must keep its own `entry_initial_margin_fixed`, fill price, and stop contract in the trade ledger
- do not enable live scale-in until per-tranche stop accounting and reconciliation are implemented

Recommended confirmation-based scale-in template:

- tranche 1: `50%` of planned size on the base signal
- tranche 2: `30%` only if the setup remains valid and price confirms with a reclaim, hold, or volume confirmation
- tranche 3: `20%` only if the trade moves at least `0.5 ATR` in favor or confirms continuation

Do not fill remaining tranches if:

- spread or slippage expands
- ATR leaves the allowed band
- the stop distance becomes unacceptable
- the AI regime gate turns negative

Partial exit rules can coexist with ATR trailing:

- optional early reduction: `20%` to `30%` after `+1 ATR` and move the stop toward break-even
- optional weakness reduction: `20%` to `30%` if trend quality degrades or the AI exit posture turns defensive
- final remainder exits through the ATR trailing stop

## 7. Strategy Skeleton for Version 1

Use one strategy:

`trend-following pullback entry with regime and volatility filters`

### 7.1 Long setup

All of the following must be true:

- `15m` EMA fast is above EMA slow
- `1h` trend filter is not bearish
- `3m` pullback returns to EMA zone or VWAP area
- `3m` RSI cools down toward neutral and turns up again
- candle closes back above the local trigger level
- ATR is inside the allowed volatility band
- AI regime gate does not veto the trade

### 7.2 Short setup

All of the following must be true:

- `15m` EMA fast is below EMA slow
- `1h` trend filter is not bullish
- `3m` pullback returns to EMA zone or VWAP area
- `3m` RSI resets upward and turns down again
- candle closes back below the local trigger level
- ATR is inside the allowed volatility band
- AI regime gate does not veto the trade

### 7.3 Entry filters

Avoid entries when:

- ATR is too low and the market is dead
- ATR is too high and structure is unstable
- open interest spikes without price confirmation
- funding behavior is distorted
- recent losses triggered a cooldown
- spread or slippage exceeds threshold

### 7.4 Optional filters worth adding

These are good additions if implementation time allows:

- no-entry blackout around major macro events
- no re-entry for `N` candles after a stop-out
- daily max trade count
- no new entries if websocket health is degraded

## 8. Gemma4 Training and Inference Design

### 8.1 What "training Gemma4 on BTC data" should mean

Do not interpret this as training a base model from scratch on chart history.
That is not practical for this project.

For this bot, `training Gemma4` should mean:

- build a labeled BTC market dataset
- fine-tune or adapt the model with `LoRA` or `QLoRA`
- train it to classify regimes and trade quality
- train it to produce strict JSON outputs

### 8.2 Target local model

The planned local inference target is:

- model: `Jiunsong/supergemma4-26b-uncensored-mlx-4bit-v2`
- base model listed on the model card: `google/gemma-4-26B-A4B-it`
- format: MLX 4-bit
- serving mode: local text-generation inference on Apple Silicon

This model should be treated as:

- a local inference engine for regime, quality, and exit-posture tasks
- an offline fine-tuning target only after the data and review pipeline are stable

### 8.3 Training objectives

Recommended training targets:

- `market_regime`: trend up, trend down, range, noisy, breakout risk
- `setup_quality`: strong, medium, weak
- `entry_veto`: allow, veto, reduce_size
- `exit_posture`: hold, tighten_stop, exit

Version 1 should not train the model to output raw order instructions.

### 8.4 Training sample design

Each training sample should be a structured market window, for example:

- recent `3m`, `15m`, `1h` candle summaries
- ATR, EMA slope, RSI, ADX, VWAP distance
- funding rate and delta
- open interest delta
- taker buy/sell ratio
- current trend state
- optional position context

Label sources can come from:

- deterministic rule-engine tags
- backtest outcomes
- manually reviewed high-quality trade examples

Every training sample should also carry:

- `policy_version`
- `strategy_version`
- `feature_schema_version`
- `labeler_version`
- `dataset_version`
- `model_base`
- `adapter_version`

### 8.5 Data split rules

To avoid leakage:

- split train, validation, and test by time
- never shuffle future windows into older training periods
- evaluate on unseen date ranges
- review performance by regime, not just total PnL

### 8.6 Inference input format

Feed the model structured JSON, not raw chart dumps.

Example input groups:

- latest OHLCV summary
- indicator snapshot by timeframe
- ATR, EMA slope, RSI, ADX, VWAP distance
- funding rate
- open interest change
- taker buy/sell ratio
- recent swing structure
- current position and unrealized PnL
- time since entry

### 8.7 Output contract

Require strict JSON output:

```json
{
  "regime": "trend_up|trend_down|range|high_volatility|unclear",
  "setup_quality": 0.0,
  "entry_action": "allow|veto|reduce_size",
  "exit_action": "hold|tighten_stop|full_exit",
  "confidence": 0.0,
  "reason_codes": ["trend_alignment", "weak_volume"]
}
```

Reject free-form text in the live execution path.

### 8.8 When to run local inference

Run AI only on meaningful events:

- candle close on execution timeframe
- new setup candidate detected
- open position state change
- ATR trail update decision point

Do not run the model on every tick.

### 8.9 AI safety rules

If the AI output is invalid:

- ignore it
- fall back to deterministic logic
- log the failure

If AI confidence is below threshold:

- ignore the recommendation

If AI says to widen risk:

- reject the recommendation

## 9. Data Pipeline Design

### 9.1 Historical data to collect

For `BTCUSDT`, backfill all available relevant history from the chosen sources:

- futures contract-price OHLCV for `1m`, `3m`, `5m`, `15m`, `1h`
- mark price candles
- funding rate history
- open interest history
- taker buy/sell volume
- long/short ratio data if used later
- exchange rule metadata

Optional secondary context:

- spot BTCUSDT candles for longer historical context
- premium index or basis series

### 9.2 Real-time data to collect

- live kline stream for execution and filter timeframes
- mark price stream
- user data stream for order and position events
- periodic account snapshot

### 9.3 Storage stack

Recommended simple storage stack:

- raw historical files: `Parquet`
- analytics and joins: `DuckDB`
- runtime state and logs: `SQLite`

Suggested directory layout:

- `data/raw/`
- `data/features/`
- `data/backtests/`
- `data/runtime/`

### 9.4 Feature groups

Version 1 feature groups:

- trend: EMA, SMA slope, higher-high and higher-low detection
- momentum: RSI, MACD histogram, rate of change
- volatility: ATR, Bollinger width
- volume: candle volume z-score, taker ratio
- derivatives context: funding rate, open interest delta
- structure: VWAP distance, swing levels, session range

## 10. Backtest and Validation Framework

### 10.1 Backtest requirements

The backtester must support:

- hybrid replay using contract-price candles for signals and lower-timeframe mark-price replay for stop detection
- fee model
- slippage model
- leverage-aware sizing
- hard stop based on `-5%` of entry-time fixed initial margin
- mark-price unrealized PnL trigger logic
- ATR trailing exits
- time stops
- cooldown rules
- daily loss lockout

Preferred replay model:

- signal engine runs on the intended execution timeframe such as `3m`
- mark-price stop detection runs on lower-timeframe replay such as `1m` or finer when available
- if only candle history is available, use a conservative breach rule instead of close-only logic

Conservative fallback rule:

- if lower-timeframe mark-price high or low breaches the hard-stop threshold intrabar, assume the stop was triggered
- fill at the worse of stop price or next executable contract-price level, plus configured slippage

### 10.2 Validation stages

Use this progression:

1. indicator correctness test
2. strategy replay on historical data
3. walk-forward validation
4. paper trading on live data
5. Binance futures testnet
6. very small live capital

### 10.3 Metrics to track

Minimum evaluation metrics:

- win rate
- average R multiple
- profit factor
- max drawdown
- average holding time
- long and short split
- regime-specific performance
- performance after fees and slippage

### 10.4 Anti-overfitting rules

- keep optimized parameters limited
- lock train and validation windows
- reject fragile parameter sets
- compare rule-only vs rule-plus-AI fairly

## 11. Execution Engine Design

### 11.1 Required capabilities

- set leverage
- place market or limit entry
- place protective stop immediately after fill
- evaluate ATR-based trailing exits on candle close
- manage reduce-only exits
- reconcile exchange state with local state
- recover after restart

### 11.2 Order lifecycle

Recommended order flow:

1. detect valid setup
2. compute allowed notional and size
3. compute margin-based hard stop price
4. place entry order
5. confirm fill
6. place protective hard stop immediately
7. begin ATR trail evaluation after position is live
8. if ATR, time, or AI exit logic triggers, send a reduce-only exit order
9. close and archive the trade

### 11.3 Stop update rules

- never remove protection while a position is open
- never widen the hard-stop contract after entry
- ATR exit logic should not overwrite the hard-stop contract by default
- if a later safety tightening rule is added, it may only tighten the hard-stop contract
- if stop update fails, pause further entries and alert

### 11.4 Exchange reconciliation

At a fixed interval, reconcile:

- local open position vs exchange position
- local open orders vs exchange open orders
- expected stop price vs actual protective stop
- leverage and margin mode vs expected configuration

If mismatch exists:

- pause new entries
- resolve inconsistency first

## 12. Runtime Risk Manager

This is the most important module in the system.

### 12.1 Hard rules

- never exceed configured leverage cap
- never open a second position in version 1
- never trade if account state is stale
- never trade without stop protection
- never widen risk after entry

### 12.2 Session guards

- daily realized loss threshold -> stop trading for the day
- max consecutive losses -> cooldown
- abnormal API error threshold -> pause bot
- slippage threshold breach -> block entries
- websocket desync -> pause trading

### 12.3 Kill switches

Kill switch must flatten or pause under:

- missing protective stop
- user data stream failure
- repeated exchange rejection
- local state corruption
- model output parse failure above threshold
- drawdown breach

## 13. Observability and Logging

### 13.1 Trade log contents

Every trade should log:

- signal reason
- AI output snapshot
- entry and exit timestamps
- entry and exit prices
- entry and exit mark prices
- leverage
- size
- entry-time fixed initial margin
- hard stop level
- ATR trail levels over time
- policy version
- strategy version
- feature schema version
- model base
- adapter version
- dataset version
- realized PnL
- slippage
- fees
- reason for exit

Concrete schema artifact:

- version 1 trade records should conform to `docs/schemas/trade_log_v1.schema.json`
- if a trade was taken without model influence, store `model_base="rule_only"`
- store `dataset_version=null` until the trade is promoted into a training dataset lineage

### 13.2 Operational logs

Also log:

- websocket reconnects
- API failures
- leverage changes
- position reconciliation events
- risk lockouts
- stop update failures
- AI inference failures

### 13.3 Review dashboard

Version 1 can use:

- CLI status view
- local HTML report
- daily summary file

At minimum, show:

- recent trades
- current position
- daily PnL
- active stop level
- strategy status
- risk status
- last AI decision

### 13.4 Loss analysis pipeline

Every losing trade should be reviewed automatically.

For each losing trade, derive and store:

- market regime at entry and exit
- whether all rule conditions were actually satisfied
- whether AI agreed, vetoed, or was ignored
- slippage and fee contribution
- maximum favorable excursion and maximum adverse excursion
- whether the hard stop was too tight for realized volatility
- whether the entry was early, late, or structurally invalid

Assign deterministic reason codes first, such as:

- `regime_misread`
- `volatility_expansion`
- `false_breakout`
- `entered_too_late`
- `overtrading`
- `slippage_too_high`
- `stop_too_tight`
- `execution_failure`
- `ignored_cooldown`
- `model_veto_missed`

The AI can add a secondary explanation layer, but deterministic tags should remain the primary truth source.

### 13.5 Self-backtest and retraining loop

Recommended improvement loop:

1. collect completed trades, feature snapshots, and AI outputs
2. auto-tag losing trades with deterministic checks
3. let AI summarize the likely loss pattern
4. run an offline batch job daily or weekly to group recurring failure modes
5. generate candidate rule or parameter changes
6. run backtests on those candidates
7. validate winners on out-of-sample and walk-forward windows
8. stage the result in shadow or paper mode for a minimum holdout period
9. only then promote selected samples into the AI training dataset
10. train or refresh the model offline
11. never let the live bot rewrite its own live rules without offline validation
12. never auto-promote a newly trained model directly into real-capital trading

Acceptance gates for any change:

- improvement after fees and slippage
- no unacceptable increase in max drawdown
- stability across multiple date ranges and market regimes
- no increase in tail-loss behavior

Recommended deployment delay:

- do not promote a new model or rule package faster than a weekly cycle
- require at least `7` days of shadow or paper observation before live promotion
- require a minimum completed-trade sample threshold before retraining or promotion

### 13.6 Rule preservation and promotion governance

To prevent review from overwhelming the original rules, separate the system into three layers:

1. `Immutable baseline policy`
- hard safety rules and baseline trading constraints
- cannot be edited by the review loop

2. `Experiment candidates`
- proposed parameter changes, filters, or model variants
- tested only in offline backtests, shadow runs, or paper mode

3. `Approved live versions`
- manually promoted strategy packages with explicit version numbers
- the only versions allowed to run with real capital

Mandatory governance rules:

- every live strategy must have a version id and changelog
- baseline rules must be stored separately from experiment output
- candidate changes must never overwrite the last approved live configuration
- every trade, training sample, and model artifact must carry explicit version metadata
- rule-only trades must still write explicit sentinel metadata instead of omitting fields
- only one major strategy change should be promoted per review cycle
- every promotion must pass a baseline regression suite on fixed historical windows
- every promotion must keep a rollback target to the previous approved version
- if a new version violates core rules, promotion is rejected automatically

Operational implication:

- review is advisory
- offline research is exploratory
- live policy changes are gated and explicit

## 14. Proposed System Modules

Suggested module split:

- `config/`
- `policy/`
- `data/`
- `features/`
- `strategy/`
- `ai/`
- `risk/`
- `execution/`
- `backtest/`
- `runtime/`
- `monitoring/`
- `review/`

Suggested responsibilities:

- `policy/`: immutable live rules, approved strategy versions, rollback targets
- `data/`: REST and WebSocket ingestion, persistence
- `features/`: indicators and feature snapshots
- `strategy/`: deterministic setup detection
- `ai/`: Gemma4 fine-tuning prep, inference, JSON parsing
- `risk/`: sizing, stop rules, kill switches
- `execution/`: order placement and reconciliation
- `backtest/`: simulation and reports
- `runtime/`: scheduler and orchestration
- `monitoring/`: logs, alerts, summaries
- `review/`: loss analysis, reason tagging, retraining dataset generation

Suggested policy storage split:

- `policy/approved_live/`
- `policy/experiment/`
- `policy/training_candidates/`
- `policy/rollback/`

## 15. Development Roadmap

### Phase 1: foundation

- create project structure
- add config and secrets management
- connect to Binance market data
- backfill historical BTCUSDT data
- calculate indicators correctly

Deliverable:

- reproducible local dataset and feature pipeline

### Phase 2: strategy and backtest

- implement the rule-based strategy
- implement fixed entry-margin hard stop logic
- implement mark-price PnL stop trigger logic
- implement ATR trailing logic
- build a backtester with fees and slippage

Deliverable:

- historical backtest report with stable metrics

### Phase 3: testnet execution

- implement execution engine
- wire account and order streams
- validate immediate stop placement
- validate `workingType=MARK_PRICE` stop behavior
- validate local market-order fail-safe when mark-price loss breach is confirmed
- validate ATR stop updates
- run on Binance futures testnet

Deliverable:

- stable testnet runtime with safe recovery behavior

### Phase 4: Gemma4 integration

- build structured market-window dataset
- label training samples
- add local `Jiunsong/supergemma4-26b-uncensored-mlx-4bit-v2` inference wrapper
- fine-tune for regime and setup-quality tasks
- add post-trade loss tagging and retraining dataset generation
- compare rule-only vs rule-plus-AI

Deliverable:

- measurable AI impact report

### Phase 5: small live deployment

- deploy with low capital
- enforce low leverage
- review execution quality and slippage
- tune only after enough sample size

Deliverable:

- live trade journal and operating checklist

## 16. Recommended Version 1 Decisions

To avoid scope explosion, lock these first:

- symbol: `BTCUSDT`
- execution timeframe: `3m`
- confirmation timeframe: `15m`
- macro filter: `1h`
- mode: one-way
- margin: isolated
- live start leverage: `2x`
- system leverage cap: `10x`
- one position at a time: yes
- entry style: trend-following pullback
- hard stop basis: `-5%` of entry-time fixed initial margin
- hard stop trigger: `mark price` unrealized PnL
- exchange stop trigger mode: `workingType=MARK_PRICE`
- emergency fallback exit: reduce-only market exit after confirmed mark-price breach
- signal and ATR source: contract-price candles
- profit capture: ATR trailing stop on contract-price candles
- split entry: disabled by default, enable only after backtest proof
- partial exit: optional `20%` to `30%` reduction after `+1 ATR`
- AI role: regime gate and setup-quality filter
- live model promotion: delayed, never immediate

## 17. Extra Recommendations Worth Adding

These are not mandatory for day one, but they are strong additions:

- news blackout around major macro releases
- daily max trade count
- separate maker and taker fee assumptions in backtest
- model retraining cadence such as weekly or monthly offline refresh
- manual approval gate for rule changes suggested by the research loop
- manual emergency flat button
- health check for disk space and process memory on the M2 MacBook Air

## 18. Final Recommendation

The right architecture is:

`Data -> Features -> Rule Strategy -> AI Gate -> Policy -> Risk Manager -> Execution -> Monitoring -> Review`

The best version 1 is not the smartest bot.
Review may improve it, but review may not erase its safety contract.
It is the bot that:

- always has stop protection
- exits exactly when the rules say it should
- survives API and websocket failures
- produces reproducible logs
- can be backtested honestly
- can later absorb AI safely
