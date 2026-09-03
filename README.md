# Coinbase Trader Agent

This project is a simulation-first AI trading agent for Coinbase Advanced Trade. It is built to evaluate an agent on a small bankroll with strict risk controls before any live trading is possible.

It does not promise returns. A goal like turning `$100` into `$400` in two weeks is highly speculative and statistically unlikely without taking substantial risk. The code defaults to `paper` mode and requires an explicit live-trading enable flag before it can submit orders.

## Why this design

- Coinbase AgentKit examples are useful for agent orchestration patterns, but exchange execution belongs on Coinbase Advanced Trade.
- The agent separates responsibilities: market data, AI policy generation, hard risk checks, execution, and state persistence.
- The AI never places orders directly. It proposes a structured action which must pass deterministic validation.

## Features

- Typed settings via environment variables
- Public market-data ingestion from Coinbase Advanced Trade
- Multi-asset universe support (`TRADER_PRODUCT_IDS`) for rotating across Coinbase pairs
- AI decision engine using the OpenAI Responses API
- Feature-engineering pipeline with de-correlation filtering inspired by DRL crypto research
- Multi-objective decision shaping (return potential, downside risk proxy, and trade-count penalty)
- Deterministic risk controls for position sizing, slippage guardrails, drawdown, and daily loss
- Cost-aware edge gating (spread + fees + minimum edge floor) and anti-overtrading cooldown limits
- SQLite-backed paper broker for repeatable evaluation
- Optional live broker integration through Coinbase Advanced Trade
- CLI for one-shot or looped execution
- Unit tests around risk enforcement and AI-response parsing

## Quickstart

1. Create a virtual environment and install dependencies.
2. Copy `.env.example` to `.env` and fill in your keys.
3. Run in paper mode first:

```powershell
uv sync
uv run coinbase-trader run-once
```

4. Review the SQLite state in `.agent_state/trader.db`.

## Core commands

```powershell
uv run coinbase-trader run-once
uv run coinbase-trader loop
uv run pytest
```

## Environment variables

- `TRADER_MODE`: `paper` or `live`
- `COINBASE_API_KEY`: must be CDP key name format `organizations/{org_id}/apiKeys/{key_id}`
- `COINBASE_API_SECRET`: must be ECDSA PEM private key (BEGIN/END PRIVATE KEY with preserved newlines)
- `COINBASE_API_KEY_FILE`: optional path to downloaded CDP key JSON file containing `name` and `privateKey`
- `TRADER_PRODUCT_ID`: Market pair such as `BTC-USD`
- `TRADER_PRODUCT_IDS`: Comma-separated universe, such as `BTC-USD,ETH-USD,SOL-USD`
- `TRADER_MAX_TRADES_PER_CYCLE`: Maximum approved trades in one `run-once` cycle
- `TRADER_AGGRESSIVE_MODE`: Enables hold-to-trade conversion and larger sizing multipliers
- `TRADER_AGGRESSIVE_SIZE_MULTIPLIER`: Size multiplier used in aggressive mode
- `TRADER_AGGRESSIVE_HOLD_TO_TRADE_EDGE_BPS`: Edge threshold that can convert hold into a directional trade
- `TRADER_PAPER_STARTING_CASH`: Starting bankroll for the simulator
- `TRADER_MAX_NOTIONAL_PER_TRADE`: Max USD notional for any single trade
- `TRADER_MAX_POSITION_NOTIONAL`: Max total position size
- `TRADER_MAX_DAILY_LOSS`: Daily stop-loss in quote currency
- `TRADER_STOP_LOSS_PCT`: Per-position forced sell threshold, as a decimal fraction; `0.03` means sell when price is 3% below average entry
- `TRADER_MAX_DRAWDOWN_PCT`: Max peak-to-trough drawdown before the agent stands down
- `TRADER_MIN_CONFIDENCE`: Minimum AI confidence required to trade
- `TRADER_FEE_BPS`: Fee model used for paper mode and edge checks
- `TRADER_MIN_EXPECTED_EDGE_BPS`: Margin above estimated costs required to allow trades
- `TRADER_MAX_TRADES_PER_DAY`: Hard cap on daily filled trades
- `TRADER_MIN_MINUTES_BETWEEN_TRADES`: Cooldown after any filled trade
- `TRADER_LIVE_TRADING_ENABLED`: Must be `true` before live orders are allowed

## Suggested evaluation process

1. Run the agent in paper mode for at least several days.
2. Inspect the trade ledger and decision log.
3. Tighten limits if turnover or concentration is too high.
4. Only then consider extremely small live sizing.

## References

- Coinbase Advanced Trade API: <https://docs.cdp.coinbase.com/api-reference/advanced-trade-api/rest-api/orders/create-order>
- Coinbase Advanced Python SDK: <https://coinbase.github.io/coinbase-advanced-py/>
- Coinbase AgentKit examples: <https://github.com/coinbase/agentkit/tree/main/python/examples>
- Deep Reinforcement Learning for Cryptocurrency Trading: Practical Approach to Address Backtest Overfitting: <https://arxiv.org/abs/2209.05559>
- A profitable trading algorithm for cryptocurrencies using a Neural Network model: <https://www.sciencedirect.com/science/article/pii/S0957417423023084>
- Optimization of Cryptocurrency Algorithmic Trading Strategies Using the Decomposition Approach: <https://www.mdpi.com/2504-2289/7/4/174>
- Gaia + Coinbase AgentKit tutorial: <https://docs.gaianet.ai/tutorial/coinbase/>

