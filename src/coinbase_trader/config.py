from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec

from coinbase_trader.models import TraderMode


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    openai_api_key: SecretStr
    openai_model: str = "gpt-4.1-mini"

    coinbase_api_key: str | None = None
    coinbase_api_secret: SecretStr | None = None
    coinbase_api_key_file: Path | None = None
    cdp_api_key_name: str | None = None
    cdp_api_key_private_key: SecretStr | None = None
    coinbase_use_sandbox: bool = False

    trader_mode: TraderMode = TraderMode.PAPER
    trader_product_id: str = "BTC-USD"
    trader_product_ids_raw: str = Field(default="BTC-USD", validation_alias="TRADER_PRODUCT_IDS")
    trader_candle_granularity: str = "ONE_HOUR"
    trader_lookback_candles: int = Field(default=48, ge=24, le=300)
    trader_loop_seconds: int = Field(default=0, ge=0, le=86400)
    trader_max_trades_per_cycle: int = Field(default=2, ge=1, le=20)
    trader_aggressive_mode: bool = False
    trader_aggressive_size_multiplier: float = Field(default=1.40, ge=1.0, le=5.0)
    trader_aggressive_hold_to_trade_edge_bps: Decimal = Field(default=Decimal("180"), ge=Decimal("0"))
    trader_probe_trade_on_all_hold: bool = False
    trader_probe_trade_notional: Decimal = Field(default=Decimal("2"), ge=Decimal("0.5"))
    trader_stop_loss_pct: Decimal = Field(default=Decimal("0.03"), ge=Decimal("0"), le=Decimal("0.50"))
    trader_paper_starting_cash: Decimal = Field(default=Decimal("100"), gt=Decimal("0"))

    trader_max_notional_per_trade: Decimal = Field(default=Decimal("15"), gt=Decimal("0"))
    trader_max_position_notional: Decimal = Field(default=Decimal("40"), gt=Decimal("0"))
    trader_max_daily_loss: Decimal = Field(default=Decimal("10"), gt=Decimal("0"))
    trader_max_drawdown_pct: float = Field(default=0.12, gt=0.0, le=1.0)
    trader_max_slippage_bps: Decimal = Field(default=Decimal("35"), ge=Decimal("0"))
    trader_min_confidence: float = Field(default=0.62, ge=0.0, le=1.0)
    trader_fee_bps: Decimal = Field(default=Decimal("60"), ge=Decimal("0"))
    trader_min_expected_edge_bps: Decimal = Field(default=Decimal("20"), ge=Decimal("0"))
    trader_max_trades_per_day: int = Field(default=8, ge=1, le=200)
    trader_min_minutes_between_trades: int = Field(default=30, ge=0, le=1440)
    trader_volatility_target: float = Field(default=0.02, gt=0.0, le=1.0)
    trader_volatility_hard_limit: float = Field(default=0.08, gt=0.0, le=2.0)
    trader_feature_correlation_threshold: float = Field(default=0.60, gt=0.0, le=1.0)
    trader_api_timeout_seconds: int = Field(default=20, ge=5, le=120)
    trader_marketdata_retries: int = Field(default=2, ge=0, le=10)

    trader_live_trading_enabled: bool = False
    trader_state_dir: Path = Path(".agent_state")

    @property
    def state_db_path(self) -> Path:
        return self.trader_state_dir / "trader.db"

    @property
    def effective_product_ids(self) -> list[str]:
        raw = self.trader_product_ids_raw.strip()
        parsed = [product.strip().upper() for product in raw.split(",") if product.strip()]
        if parsed:
            return parsed
        return [self.trader_product_id.strip().upper()]

    @property
    def effective_granularity(self) -> str:
        aliases = {
            "60": "ONE_MINUTE",
            "300": "FIVE_MINUTE",
            "900": "FIFTEEN_MINUTE",
            "1800": "THIRTY_MINUTE",
            "3600": "ONE_HOUR",
            "7200": "TWO_HOUR",
            "21600": "SIX_HOUR",
            "86400": "ONE_DAY",
            "ONE_MINUTE": "ONE_MINUTE",
            "FIVE_MINUTE": "FIVE_MINUTE",
            "FIFTEEN_MINUTE": "FIFTEEN_MINUTE",
            "THIRTY_MINUTE": "THIRTY_MINUTE",
            "ONE_HOUR": "ONE_HOUR",
            "TWO_HOUR": "TWO_HOUR",
            "SIX_HOUR": "SIX_HOUR",
            "ONE_DAY": "ONE_DAY",
        }
        key = self.trader_candle_granularity.strip().upper()
        return aliases.get(key, self.trader_candle_granularity.strip().upper())

    @property
    def effective_coinbase_api_key(self) -> str | None:
        value = self.coinbase_api_key or self.cdp_api_key_name
        if value is None:
            value = self._extract_from_key_file(["name", "key_name", "apiKeyName", "api_key_name"])
        if value is None:
            return None
        cleaned = value.strip().strip('"').strip("'")
        return cleaned or None

    @property
    def effective_coinbase_api_secret(self) -> SecretStr | None:
        value = self.coinbase_api_secret or self.cdp_api_key_private_key
        if value is None:
            extracted = self._extract_from_key_file(["privateKey", "private_key", "key_secret", "apiSecret"])
            value = SecretStr(extracted) if extracted is not None else None
        if value is None:
            return None
        raw = value.get_secret_value().strip().strip('"').strip("'")
        if raw.startswith("{") and raw.endswith("}"):
            try:
                blob = json.loads(raw)
                extracted = self._extract_from_json_obj(
                    blob,
                    ["privateKey", "private_key", "key_secret", "apiSecret"],
                )
                if extracted is not None:
                    raw = extracted
            except json.JSONDecodeError:
                pass
        cleaned = raw
        cleaned = cleaned.replace("\\n", "\n")
        return SecretStr(cleaned)

    @property
    def effective_openai_api_key(self) -> str:
        cleaned = self.openai_api_key.get_secret_value().strip().strip('"').strip("'")
        if not cleaned:
            raise ValueError("OPENAI_API_KEY is empty after trimming whitespace.")
        return cleaned

    def validate_live_coinbase_credentials(self) -> tuple[str, str]:
        key = self.effective_coinbase_api_key
        secret = self.effective_coinbase_api_secret
        if key is None or secret is None:
            raise ValueError(
                "Live mode requires Coinbase credentials. Use COINBASE_API_KEY and COINBASE_API_SECRET "
                "(or CDP_API_KEY_NAME and CDP_API_KEY_PRIVATE_KEY)."
            )
        key_value = key.strip()
        secret_value = secret.get_secret_value().strip()
        if not key_value.startswith("organizations/") or "/apiKeys/" not in key_value:
            raise ValueError(
                "COINBASE_API_KEY format is invalid. Expected: organizations/{org_id}/apiKeys/{key_id}. "
                "Create a Secret API Key in CDP and copy the full key name."
            )
        if "BEGIN" not in secret_value or "PRIVATE KEY" not in secret_value:
            raise ValueError(
                "COINBASE_API_SECRET is not a PEM private key. It must include BEGIN/END PRIVATE KEY lines "
                "with preserved newlines (or escaped \\n)."
            )
        try:
            parsed = serialization.load_pem_private_key(secret_value.encode("utf-8"), password=None)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(
                "COINBASE_API_SECRET could not be parsed as PEM. Verify you copied the private key exactly from "
                "CDP (Secret API Keys) and did not paste key id/UUID or base64 secret fields."
            ) from exc
        if not isinstance(parsed, ec.EllipticCurvePrivateKey):
            raise ValueError("Unsupported key type. Coinbase App/Advanced Trade SDK requires ECDSA (ES256) keys.")
        return key_value, secret_value

    def _extract_from_key_file(self, key_candidates: list[str]) -> str | None:
        if self.coinbase_api_key_file is None:
            return None
        path = self.coinbase_api_key_file.expanduser()
        if not path.exists():
            return None
        try:
            blob = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return None
        return self._extract_from_json_obj(blob, key_candidates)

    @staticmethod
    def _extract_from_json_obj(blob: dict[str, object], key_candidates: list[str]) -> str | None:
        for candidate in key_candidates:
            value = blob.get(candidate)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

