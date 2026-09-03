from __future__ import annotations

import argparse
import json
from decimal import Decimal
from pathlib import Path

from pydantic import SecretStr

from coinbase_trader.agent import TradingAgent
from coinbase_trader.config import Settings
from coinbase_trader.logging_config import configure_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Coinbase AI trader")
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("run-once", help="Fetch data, ask the model, and process one trading step.")
    subcommands.add_parser("loop", help="Run continuously using TRADER_LOOP_SECONDS.")
    subcommands.add_parser("show-config", help="Render the active, non-secret configuration.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    configure_logging()
    settings = Settings()

    if args.command == "show-config":
        payload = settings.model_dump(exclude={"openai_api_key", "coinbase_api_secret", "cdp_api_key_private_key"})
        payload["state_db_path"] = str(settings.state_db_path)
        payload["trader_product_ids"] = settings.effective_product_ids
        payload["trader_candle_granularity"] = settings.effective_granularity
        print(json.dumps(_stringify_decimals(payload), indent=2, sort_keys=True))
        return

    agent = TradingAgent(settings)
    if args.command == "run-once":
        result = agent.run_once()
        print(result.model_dump_json(indent=2))
        return
    if args.command == "loop":
        agent.loop()
        return
    raise ValueError(f"Unsupported command: {args.command}")


def _stringify_decimals(value: object) -> object:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, SecretStr):
        return "********"
    if isinstance(value, dict):
        return {key: _stringify_decimals(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_stringify_decimals(item) for item in value]
    return value


if __name__ == "__main__":
    main()

