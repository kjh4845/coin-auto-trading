from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_repo_env(repo_root: Path) -> dict[str, str]:
    env_path = repo_root / ".env"
    if not env_path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        value = value.strip()
        if not name:
            continue
        if value and value[0] in {"'", '"'} and value[-1:] == value[:1]:
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        values[name] = value
    return values


def _read_env(name: str, default: str, *, dotenv: dict[str, str]) -> str:
    value = os.getenv(name)
    if value not in (None, ""):
        return value
    value = dotenv.get(name)
    return value if value not in (None, "") else default


def _read_env_first(names: list[str], default: str, *, dotenv: dict[str, str]) -> str:
    for name in names:
        value = os.getenv(name)
        if value not in (None, ""):
            return value
        value = dotenv.get(name)
        if value not in (None, ""):
            return value
    return default


def _read_env_bool(name: str, default: bool, *, dotenv: dict[str, str]) -> bool:
    value = os.getenv(name)
    if value in (None, ""):
        value = dotenv.get(name)
    if value in (None, ""):
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _default_local_model_path() -> str:
    llm_root = Path.home() / "llm"
    preferred = llm_root / "gemma4-e2b" / "model"
    if (preferred / "config.json").exists():
        return str(preferred)

    if not llm_root.exists():
        return ""

    for child in sorted(llm_root.iterdir()):
        if not child.is_dir():
            continue
        lowered = child.name.lower()
        if "gemma" not in lowered or "e2b" not in lowered:
            continue
        for candidate in (child / "model", child):
            if (candidate / "config.json").exists():
                return str(candidate)
    return ""


def _default_local_model_python() -> str:
    preferred = Path.home() / "llm" / "gemma4-e2b" / ".venv" / "bin" / "python"
    if preferred.exists():
        return str(preferred)
    return ""


@dataclass(frozen=True)
class Settings:
    app_env: str
    log_level: str
    strategy_mode: str
    repo_root: Path
    config_dir: Path
    policy_dir: Path
    data_dir: Path
    tests_dir: Path
    trading_symbol: str
    execution_timeframe: str
    micro_timeframe: str
    confirmation_timeframe: str
    macro_timeframe: str
    regime_timeframe: str
    anchor_timeframe: str
    atr_trailing_multiplier: float
    atr_trail_activation_profit_r: float
    atr_trail_min_bars: int
    exit_policy: str
    fixed_take_profit_r: float
    max_holding_bars: int
    live_start_leverage: int
    system_leverage_cap: int
    binance_futures_base_url: str
    binance_futures_ws_base_url: str
    binance_futures_testnet_base_url: str
    binance_futures_testnet_ws_base_url: str
    binance_api_key: str
    binance_api_secret: str
    binance_testnet_api_key: str
    binance_testnet_api_secret: str
    local_model_id: str
    local_model_base: str
    local_model_path: str
    local_model_python: str
    local_model_endpoint: str
    local_model_mps_max_memory: str
    local_model_cpu_max_memory: str
    ai_gate_enabled: bool
    ai_gate_fail_open: bool
    ai_gate_min_setup_quality: float
    ai_reduce_size_fraction: float
    ai_request_timeout_seconds: float
    ai_max_tokens: int
    ai_max_latency_ms: int
    allow_long_entries: bool
    allow_short_entries: bool
    max_long_funding_rate: float
    min_short_funding_rate: float
    min_long_taker_buy_ratio: float
    max_short_taker_buy_ratio: float
    min_micro_long_ema_spread_pct: float
    min_micro_short_ema_spread_pct: float
    min_higher_tf_ema_spread_pct: float
    min_volume_ratio_20: float
    max_daily_loss_r: float
    max_consecutive_losses: int
    cooldown_after_loss_minutes: int


def load_settings() -> Settings:
    repo_root = _repo_root()
    dotenv = _load_repo_env(repo_root)
    settings = Settings(
        app_env=_read_env("APP_ENV", "development", dotenv=dotenv),
        log_level=_read_env("LOG_LEVEL", "INFO", dotenv=dotenv),
        strategy_mode=_read_env("STRATEGY_MODE", "single_profile", dotenv=dotenv),
        repo_root=repo_root,
        config_dir=repo_root / "config",
        policy_dir=repo_root / "policy",
        data_dir=repo_root / "data",
        tests_dir=repo_root / "tests",
        trading_symbol=_read_env("TRADING_SYMBOL", "BTCUSDT", dotenv=dotenv),
        execution_timeframe=_read_env("EXECUTION_TIMEFRAME", "3m", dotenv=dotenv),
        micro_timeframe=_read_env("MICRO_TIMEFRAME", "5m", dotenv=dotenv),
        confirmation_timeframe=_read_env(
            "CONFIRMATION_TIMEFRAME", "15m", dotenv=dotenv
        ),
        macro_timeframe=_read_env("MACRO_TIMEFRAME", "1h", dotenv=dotenv),
        regime_timeframe=_read_env("REGIME_TIMEFRAME", "4h", dotenv=dotenv),
        anchor_timeframe=_read_env("ANCHOR_TIMEFRAME", "1d", dotenv=dotenv),
        atr_trailing_multiplier=float(
            _read_env("ATR_TRAILING_MULTIPLIER", "2.5", dotenv=dotenv)
        ),
        atr_trail_activation_profit_r=float(
            _read_env("ATR_TRAIL_ACTIVATION_PROFIT_R", "0.5", dotenv=dotenv)
        ),
        atr_trail_min_bars=int(_read_env("ATR_TRAIL_MIN_BARS", "2", dotenv=dotenv)),
        exit_policy=_read_env("EXIT_POLICY", "fixed_tp_time_stop", dotenv=dotenv),
        fixed_take_profit_r=float(
            _read_env("FIXED_TAKE_PROFIT_R", "1.0", dotenv=dotenv)
        ),
        max_holding_bars=int(_read_env("MAX_HOLDING_BARS", "8", dotenv=dotenv)),
        live_start_leverage=int(_read_env("LIVE_START_LEVERAGE", "2", dotenv=dotenv)),
        system_leverage_cap=int(_read_env("SYSTEM_LEVERAGE_CAP", "10", dotenv=dotenv)),
        binance_futures_base_url=_read_env(
            "BINANCE_FUTURES_BASE_URL",
            "https://fapi.binance.com",
            dotenv=dotenv,
        ),
        binance_futures_ws_base_url=_read_env(
            "BINANCE_FUTURES_WS_BASE_URL",
            "wss://fstream.binance.com",
            dotenv=dotenv,
        ),
        binance_futures_testnet_base_url=_read_env(
            "BINANCE_FUTURES_TESTNET_BASE_URL",
            "https://demo-fapi.binance.com",
            dotenv=dotenv,
        ),
        binance_futures_testnet_ws_base_url=_read_env(
            "BINANCE_FUTURES_TESTNET_WS_BASE_URL",
            "wss://fstream.binancefuture.com",
            dotenv=dotenv,
        ),
        binance_api_key=_read_env("BINANCE_API_KEY", "", dotenv=dotenv),
        binance_api_secret=_read_env("BINANCE_API_SECRET", "", dotenv=dotenv),
        binance_testnet_api_key=_read_env_first(
            ["BINANCE_TESTNET_API_KEY", "BINANCE_API_KEY"],
            "",
            dotenv=dotenv,
        ),
        binance_testnet_api_secret=_read_env_first(
            ["BINANCE_TESTNET_API_SECRET", "BINANCE_API_SECRET"],
            "",
            dotenv=dotenv,
        ),
        local_model_id=_read_env(
            "LOCAL_MODEL_ID",
            "google/gemma-4-E2B-it",
            dotenv=dotenv,
        ),
        local_model_base=_read_env("LOCAL_MODEL_BASE", "google/gemma-4-E2B-it", dotenv=dotenv),
        local_model_path=_read_env(
            "LOCAL_MODEL_PATH",
            _default_local_model_path(),
            dotenv=dotenv,
        ),
        local_model_python=_read_env(
            "LOCAL_MODEL_PYTHON",
            _default_local_model_python(),
            dotenv=dotenv,
        ),
        local_model_endpoint=_read_env(
            "LOCAL_MODEL_ENDPOINT",
            "http://127.0.0.1:8080",
            dotenv=dotenv,
        ),
        local_model_mps_max_memory=_read_env(
            "LOCAL_MODEL_MPS_MAX_MEMORY",
            "",
            dotenv=dotenv,
        ),
        local_model_cpu_max_memory=_read_env(
            "LOCAL_MODEL_CPU_MAX_MEMORY",
            "",
            dotenv=dotenv,
        ),
        ai_gate_enabled=_read_env_bool("AI_GATE_ENABLED", False, dotenv=dotenv),
        ai_gate_fail_open=_read_env_bool("AI_GATE_FAIL_OPEN", False, dotenv=dotenv),
        ai_gate_min_setup_quality=float(
            _read_env("AI_GATE_MIN_SETUP_QUALITY", "0.55", dotenv=dotenv)
        ),
        ai_reduce_size_fraction=float(
            _read_env("AI_REDUCE_SIZE_FRACTION", "0.5", dotenv=dotenv)
        ),
        ai_request_timeout_seconds=float(
            _read_env("AI_REQUEST_TIMEOUT_SECONDS", "20.0", dotenv=dotenv)
        ),
        ai_max_tokens=int(_read_env("AI_MAX_TOKENS", "128", dotenv=dotenv)),
        ai_max_latency_ms=int(_read_env("AI_MAX_LATENCY_MS", "20000", dotenv=dotenv)),
        allow_long_entries=_read_env_bool("ALLOW_LONG_ENTRIES", False, dotenv=dotenv),
        allow_short_entries=_read_env_bool("ALLOW_SHORT_ENTRIES", True, dotenv=dotenv),
        max_long_funding_rate=float(
            _read_env("MAX_LONG_FUNDING_RATE", "-0.0001", dotenv=dotenv)
        ),
        min_short_funding_rate=float(
            _read_env("MIN_SHORT_FUNDING_RATE", "0.0001", dotenv=dotenv)
        ),
        min_long_taker_buy_ratio=float(
            _read_env("MIN_LONG_TAKER_BUY_RATIO", "0.57", dotenv=dotenv)
        ),
        max_short_taker_buy_ratio=float(
            _read_env("MAX_SHORT_TAKER_BUY_RATIO", "0.43", dotenv=dotenv)
        ),
        min_micro_long_ema_spread_pct=float(
            _read_env("MIN_MICRO_LONG_EMA_SPREAD_PCT", "0.0015", dotenv=dotenv)
        ),
        min_micro_short_ema_spread_pct=float(
            _read_env("MIN_MICRO_SHORT_EMA_SPREAD_PCT", "0.00115", dotenv=dotenv)
        ),
        min_higher_tf_ema_spread_pct=float(
            _read_env("MIN_HIGHER_TF_EMA_SPREAD_PCT", "0.001", dotenv=dotenv)
        ),
        min_volume_ratio_20=float(
            _read_env("MIN_VOLUME_RATIO_20", "1.05", dotenv=dotenv)
        ),
        max_daily_loss_r=float(_read_env("MAX_DAILY_LOSS_R", "3.0", dotenv=dotenv)),
        max_consecutive_losses=int(_read_env("MAX_CONSECUTIVE_LOSSES", "3", dotenv=dotenv)),
        cooldown_after_loss_minutes=int(
            _read_env("COOLDOWN_AFTER_LOSS_MINUTES", "30", dotenv=dotenv)
        ),
    )
    _validate_settings(settings)
    return settings


def _validate_settings(settings: Settings) -> None:
    if settings.strategy_mode not in {"single_profile", "best_pair_v1", "multi_symbol_priority_v1"}:
        raise ValueError(
            "STRATEGY_MODE must be one of "
            "['best_pair_v1', 'multi_symbol_priority_v1', 'single_profile']"
        )
    if settings.system_leverage_cap <= 0:
        raise ValueError("SYSTEM_LEVERAGE_CAP must be > 0")
    if settings.atr_trail_activation_profit_r < 0:
        raise ValueError("ATR_TRAIL_ACTIVATION_PROFIT_R must be >= 0")
    if settings.atr_trail_min_bars < 0:
        raise ValueError("ATR_TRAIL_MIN_BARS must be >= 0")
    if settings.exit_policy not in {"atr_trail", "fixed_tp_time_stop", "time_stop_only"}:
        raise ValueError("EXIT_POLICY must be one of ['atr_trail', 'fixed_tp_time_stop', 'time_stop_only']")
    if settings.fixed_take_profit_r <= 0:
        raise ValueError("FIXED_TAKE_PROFIT_R must be > 0")
    if settings.ai_max_latency_ms < 0:
        raise ValueError("AI_MAX_LATENCY_MS must be >= 0")
    if not settings.allow_long_entries and not settings.allow_short_entries:
        raise ValueError("at least one of ALLOW_LONG_ENTRIES or ALLOW_SHORT_ENTRIES must be true")
    if settings.min_long_taker_buy_ratio < 0 or settings.min_long_taker_buy_ratio > 1:
        raise ValueError("MIN_LONG_TAKER_BUY_RATIO must be between 0 and 1")
    if settings.max_short_taker_buy_ratio < 0 or settings.max_short_taker_buy_ratio > 1:
        raise ValueError("MAX_SHORT_TAKER_BUY_RATIO must be between 0 and 1")
    if settings.min_micro_long_ema_spread_pct < 0:
        raise ValueError("MIN_MICRO_LONG_EMA_SPREAD_PCT must be >= 0")
    if settings.min_micro_short_ema_spread_pct < 0:
        raise ValueError("MIN_MICRO_SHORT_EMA_SPREAD_PCT must be >= 0")
    if settings.min_higher_tf_ema_spread_pct < 0:
        raise ValueError("MIN_HIGHER_TF_EMA_SPREAD_PCT must be >= 0")
    if settings.min_volume_ratio_20 < 0:
        raise ValueError("MIN_VOLUME_RATIO_20 must be >= 0")
    if settings.max_daily_loss_r < 0:
        raise ValueError("MAX_DAILY_LOSS_R must be >= 0")
    if settings.max_consecutive_losses < 0:
        raise ValueError("MAX_CONSECUTIVE_LOSSES must be >= 0")
    if settings.cooldown_after_loss_minutes < 0:
        raise ValueError("COOLDOWN_AFTER_LOSS_MINUTES must be >= 0")
    validate_leverage(settings.live_start_leverage, settings)


def validate_leverage(leverage: int, settings: Settings | None = None) -> int:
    resolved = settings or load_settings()
    if leverage <= 0:
        raise ValueError("leverage must be > 0")
    if leverage > resolved.system_leverage_cap:
        raise ValueError(
            f"leverage must be <= system cap ({resolved.system_leverage_cap})"
        )
    return leverage


def required_layout(settings: Settings) -> list[Path]:
    return [
        settings.config_dir,
        settings.policy_dir / "approved_live",
        settings.policy_dir / "experiment",
        settings.policy_dir / "training_candidates",
        settings.policy_dir / "rollback",
        settings.data_dir / "raw",
        settings.data_dir / "features",
        settings.data_dir / "backtests",
        settings.data_dir / "runtime",
        settings.tests_dir,
    ]
