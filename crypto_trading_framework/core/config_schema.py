from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    exchange_id: str = "okx"
    symbols: List[str] = Field(default_factory=lambda: ["BTC/USDT:USDT"])
    yfinance_ticker: List[str] = Field(default_factory=lambda: ["BTC-USD"])
    yfinance_period: str = "10y"
    timeframes: List[str] = Field(default_factory=lambda: ["m15", "h1", "h4", "d1"])
    lookback: int = 2000
    fallback_enabled: bool = True


class OrderbookConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    enabled: bool = True
    limit: int = 100


class MetricToggleConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    enabled: bool = True


class MarketDataConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    orderbook: OrderbookConfig = Field(default_factory=OrderbookConfig)
    funding_rate: MetricToggleConfig = Field(default_factory=MetricToggleConfig)
    open_interest: MetricToggleConfig = Field(default_factory=MetricToggleConfig)


class TrainingScheduleConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    day: str = "sunday"
    time: str = "02:00"


class MarketHoursConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    enabled: bool = False
    timezone: str = "Asia/Jakarta"
    open_time: str = "08:00"
    close_time: str = "17:00"


class TradingConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    dry_run: bool = True
    auto_start: bool = True
    training_schedule: TrainingScheduleConfig = Field(default_factory=TrainingScheduleConfig)
    max_symbols: int = 10
    market_hours: MarketHoursConfig = Field(default_factory=MarketHoursConfig)


class MarketHoursConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    enabled: bool = False
    timezone: str = "Asia/Jakarta"
    open_time: str = "08:00"
    close_time: str = "17:00"


class IndicatorsConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    ema_periods: List[int] = Field(default_factory=lambda: [20, 50, 200])
    bb_period: int = 20
    bb_multiplier: float = 2.0
    rsi_period: int = 14
    stoch_k_period: int = 14
    stoch_d_period: int = 3
    stoch_smooth_k: int = 3
    atr_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9


class MLConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    scaler_type: str = "minmax"
    time_steps: int = 60
    forward_periods: int = 10
    test_size: float = 0.2
    feature_cols: List[str] = Field(default_factory=list)


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: str = "lstm"
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.3
    fc_hidden: int = 32
    epochs: int = 100
    learning_rate: float = 0.001
    optimizer: str = "adam"
    loss: str = "bce"
    class_weight_method: str = "balanced"
    checkpoint_dir: str = "models"
    save_best_only: bool = True


class EnsembleModelSubConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: str = "lstm"
    weight: float = 0.33


class EnsembleConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    enabled: bool = True
    models: List[Dict[str, Any]] = Field(default_factory=list)
    voting: str = "soft"
    stacking_meta_model: str = "logistic"


class TargetConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: str = "binary"
    forward_periods: int = 10
    atr_multiplier: float = 1.5
    regime_lookback: int = 20
    regime_threshold: float = 0.5


class MultiTimeframeConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    enabled: bool = False
    primary_timeframe: str = "m15"
    auxiliary_timeframes: List[str] = Field(default_factory=lambda: ["m30", "h1", "h4", "d1"])


class HyperparameterTuningConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    enabled: bool = True
    n_trials: int = 10
    param_grid: Dict[str, Any] = Field(default_factory=dict)


class WalkForwardConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    enabled: bool = False
    n_splits: int = 5
    train_size: int | None = None
    test_size: int = 50
    embargo: int = 0
    min_train_size: int = 100
    expanding: bool = True


class BacktestConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    tp_pct: float = 0.03
    sl_pct: float = 0.015
    initial_capital: float = 10000.0
    position_size_method: str = "atr"
    atr_multiplier: float = 1.0
    max_risk_per_trade: float = 0.02
    trailing_stop_enabled: bool = True
    trailing_stop_activation_pct: float = 0.015
    trailing_stop_distance_pct: float = 0.01
    max_drawdown_pct: float = 0.20
    transaction_fee_pct: float = 0.001
    slippage_pct: float = 0.0005
    forward_periods: int = 20


class SignalConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    threshold: float = 0.55
    tp_pct: float = 0.03
    sl_pct: float = 0.015
    min_adx: float = 25.0
    require_volume_spike: bool = False
    min_confirmations: int = 2
    use_rule_fallback: bool = True
    min_confluences: int = 3


class LoggingConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    log_file: str = "logs/main.log"
    log_level: str = "INFO"
    format: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


class PathsConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    data_dir: str = "data"
    models_dir: str = "models"
    logs_dir: str = "logs"
    backtest_dir: str = "backtests"
    config_file: str = "config.yaml"


class ValidationConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    enabled: bool = True
    max_price_change_pct: float = 0.5
    allow_gaps: bool = False
    min_volume: float = 0.0


class TelegramAlertConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    enabled: bool = False
    bot_token: str = ""
    chat_id: str = ""


class DiscordAlertConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    enabled: bool = False
    webhook_url: str = ""


class EmailAlertConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    enabled: bool = False
    smtp_host: str = ""
    smtp_port: int = 465
    username: str = ""
    password: str = ""
    from_addr: str = ""
    to_addrs: list[str] = Field(default_factory=list)


class AlertingConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    enabled: bool = True
    min_level: str = "WARNING"
    telegram: TelegramAlertConfig = Field(default_factory=TelegramAlertConfig)
    discord: DiscordAlertConfig = Field(default_factory=DiscordAlertConfig)
    email: EmailAlertConfig = Field(default_factory=EmailAlertConfig)


class ApiConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = 8000
    api_key: str = ""
    allowed_origins: list[str] = Field(default_factory=lambda: ["*"])


class WebSocketConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    enabled: bool = False
    buffer_size: int = 100
    buffer_interval: float = 5.0


class FeatureStoreConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    enabled: bool = True
    cache_ttl: int = 60
    compute_on_ingest: bool = True


class ModelRegistryConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    enabled: bool = True
    registry_dir: str = "models/registry"
    active_version: str = "latest"
    canary: dict[str, Any] = Field(default_factory=dict)


class TaskQueueConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    enabled: bool = True
    max_concurrency: int = 4
    timeout: float = 120.0
    celery: "CeleryConfig" = Field(default_factory=lambda: CeleryConfig())


class CeleryConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    broker_url: str = "redis://localhost:6379/0"
    result_backend: str = "redis://localhost:6379/1"
    task_prefix: str = "inference"


class DriftDetectionConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    enabled: bool = True
    prediction_window: int = 100
    feature_window: int = 100
    ks_threshold: float = 0.3
    feature_z_threshold: float = 3.0
    alert_cooldown: int = 10


class RiskManagementConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    atr_multiplier_sl: float = 1.5
    atr_multiplier_tp: float = 2.0
    trailing_stop_atr_multiple: float = 1.0
    kelly_enabled: bool = True
    kelly_fraction: float = 0.5
    max_risk_per_trade: float = 0.05


class TelegramGroupConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    group_id: str = ""
    name: str = ""
    tier: str = "free"
    asset_type: str = "crypto"
    allowed_symbols: list[str] = Field(default_factory=list)
    max_signals_per_hour: int = 5
    max_messages_per_minute: int = 10
    cooldown_minutes_per_symbol: int = 15
    burst_window_minutes: int = 5
    burst_max_count: int = 3


class TelegramSignalConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    include_disclaimer: bool = True
    disclaimer_text: str = "⚠ Ini adalah analisis teknikal, bukan nasihat investasi. Gunakan dengan risiko Anda sendiri."
    timezone: str = "Asia/Jakarta"
    timestamp_format: str = "%Y-%m-%d %H:%M:%S WIB"


class TelegramBotConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    enabled: bool = False
    bot_token: str = ""
    polling_interval: float = 1.0
    max_retries: int = 3
    retry_delay_seconds: int = 5
    groups: list[TelegramGroupConfig] = Field(default_factory=list)
    signal: TelegramSignalConfig = Field(default_factory=TelegramSignalConfig)


class ContinuousLearningConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    enabled: bool = True
    evaluation_interval_hours: float = 2.0
    retrain_interval_hours: float = 24.0
    lookback_hours: float = 6.0
    min_golden_samples: int = 50
    max_memory_patterns: int = 5000
    recency_decay_alpha: float = 0.95
    retrain_on_accuracy_drop: bool = True
    accuracy_drop_threshold: float = 0.05


class DatabaseConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    url: str = ""
    pool_size: int = 5
    max_overflow: int = 10
    pool_recycle: int = 1800
    echo: bool = False


class PortfolioRiskConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    enabled: bool = True
    max_portfolio_exposure: float = 0.5
    max_correlation_exposure: float = 0.3
    max_portfolio_drawdown_pct: float = 0.25
    lookback_correlation: int = 100
    risk_per_trade: float = 0.02
    rebalance_threshold: float = 0.1


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    data: DataConfig = Field(default_factory=DataConfig)
    market_data: MarketDataConfig = Field(default_factory=MarketDataConfig)
    trading: TradingConfig = Field(default_factory=TradingConfig)
    indicators: IndicatorsConfig = Field(default_factory=IndicatorsConfig)
    ml: MLConfig = Field(default_factory=MLConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    ensemble: EnsembleConfig = Field(default_factory=EnsembleConfig)
    target: TargetConfig = Field(default_factory=TargetConfig)
    multi_timeframe: MultiTimeframeConfig = Field(default_factory=MultiTimeframeConfig)
    hyperparameter_tuning: HyperparameterTuningConfig = Field(default_factory=HyperparameterTuningConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    walk_forward: WalkForwardConfig = Field(default_factory=WalkForwardConfig)
    signal: SignalConfig = Field(default_factory=SignalConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    alerting: AlertingConfig = Field(default_factory=AlertingConfig)
    api: ApiConfig = Field(default_factory=ApiConfig)
    websocket: WebSocketConfig = Field(default_factory=WebSocketConfig)
    feature_store: FeatureStoreConfig = Field(default_factory=FeatureStoreConfig)
    drift_detection: DriftDetectionConfig = Field(default_factory=DriftDetectionConfig)
    model_registry: ModelRegistryConfig = Field(default_factory=ModelRegistryConfig)
    task_queue: TaskQueueConfig = Field(default_factory=TaskQueueConfig)
    portfolio_risk: PortfolioRiskConfig = Field(default_factory=PortfolioRiskConfig)
    continuous_learning: ContinuousLearningConfig = Field(default_factory=ContinuousLearningConfig)
    risk_management: RiskManagementConfig = Field(default_factory=RiskManagementConfig)
    telegram_bot: TelegramBotConfig = Field(default_factory=TelegramBotConfig)


def validate_config(raw_config: dict) -> dict:
    """Memvalidasi dan mengembalikan dictionary konfigurasi yang terstruktur."""
    validated = AppConfig(**raw_config)
    return validated.model_dump()
