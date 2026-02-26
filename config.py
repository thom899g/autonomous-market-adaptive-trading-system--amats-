"""
Configuration and environment management for AMATS.
Centralizes all configurable parameters and environment variables.
"""
import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

@dataclass
class DatabaseConfig:
    """Firebase configuration"""
    firebase_credentials_path: str = os.getenv("FIREBASE_CREDENTIALS_PATH", "./firebase-credentials.json")
    firestore_collection: str = os.getenv("FIRESTORE_COLLECTION", "amats_trading")
    trade_history_subcollection: str = os.getenv("TRADE_HISTORY_SUBCOLLECTION", "trade_history")
    strategy_state_subcollection: str = os.getenv("STRATEGY_STATE_SUBCOLLECTION", "strategy_states")

@dataclass
class ExchangeConfig:
    """Exchange API configuration"""
    exchange_id: str = os.getenv("EXCHANGE_ID", "binance")
    api_key: Optional[str] = os.getenv("EXCHANGE_API_KEY")
    api_secret: Optional[str] = os.getenv("EXCHANGE_API_SECRET")
    sandbox_mode: bool = os.getenv("SANDBOX_MODE", "True").lower() == "true"
    rate_limit: int = int(os.getenv("RATE_LIMIT", "1000"))

@dataclass
class TradingConfig:
    """Trading parameters"""
    initial_capital: float = float(os.getenv("INITIAL_CAPITAL", "10000.0"))
    max_position_size: float = float(os.getenv("MAX_POSITION_SIZE", "0.1"))
    stop_loss_pct: float = float(os.getenv("STOP_LOSS_PCT", "0.02"))
    take_profit_pct: float = float(os.getenv("TAKE_PROFIT_PCT", "0.04"))
    max_daily_trades: int = int(os.getenv("MAX_DAILY_TRADES", "10"))

@dataclass
class MLConfig:
    """Machine learning configuration"""
    model_save_path: str = os.getenv("MODEL_SAVE_PATH", "./models")
    training_lookback_days: int = int(os.getenv("TRAINING_LOOKBACK_DAYS", "365"))
    prediction_horizon: int = int(os.getenv("PREDICTION_HORIZON", "24"))
    feature_window_size: int = int(os.getenv("FEATURE_WINDOW_SIZE", "50"))
    retrain_frequency_hours: int = int(os.getenv("RETRAIN_FREQUENCY_HOURS", "24"))

@dataclass
class LoggingConfig:
    """Logging configuration"""
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_file: str = os.getenv("LOG_FILE", "./logs/amats.log")
    max_log_size_mb: int = int(os.getenv("MAX_LOG_SIZE_MB", "100"))
    backup_count: int = int(os.getenv("LOG_BACKUP_COUNT", "5"))

class ConfigManager:
    """Manages configuration validation and access"""
    
    def __init__(self):
        self.db = DatabaseConfig()
        self.exchange = ExchangeConfig()
        self.trading = TradingConfig()
        self.ml = MLConfig()
        self.logging = LoggingConfig()
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validate critical configuration parameters"""
        if not self.exchange.api_key or not self.exchange.api_secret:
            logging.warning("Exchange API credentials not found. Sandbox mode will be used.")
            self.exchange.sandbox_mode = True
        
        if not os.path.exists(self.db.firebase_credentials_path):
            raise FileNotFoundError(
                f"Firebase credentials file not found at {self.db.firebase_credentials_path}"
            )
        
        # Validate trading parameters
        if self.trading.max_position_size <= 0 or self.trading.max_position_size > 1:
            raise ValueError("max_position_size must be between 0 and 1")
        
        if self.trading.stop_loss_pct <= 0:
            raise ValueError("stop_loss_pct must be positive")
    
    def get_all_config(self) -> dict:
        """Return all configuration as dictionary"""
        return {
            "database": self.db.__dict__,
            "exchange": self.exchange.__dict__,
            "trading": self.trading.__dict__,
            "ml": self.ml.__dict__,
            "logging": self.logging.__dict__
        }

# Global configuration instance
config = ConfigManager()