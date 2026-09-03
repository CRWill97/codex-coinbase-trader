from __future__ import annotations

import logging
import time
from decimal import Decimal

from coinbase_trader.ai import OpenAIDecisionEngine
from coinbase_trader.config import Settings
from coinbase_trader.execution import Broker, LiveBroker, PaperBroker
from coinbase_trader.market_data import MarketDataClient
from coinbase_trader.models import ExecutionReport, PortfolioState, ProductSnapshot, TradeAction, TraderMode, TradingDecision
from coinbase_trader.risk import RiskLimits, RiskManager
from coinbase_trader.state import StateStore

logger = logging.getLogger(__name__)


class TradingAgent:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._product_ids = settings.effective_product_ids
        effective_min_confidence = (
            settings.trader_min_confidence
            if not settings.trader_aggressive_mode
            else max(0.35, settings.trader_min_confidence * 0.75)
        )
        effective_min_edge = (
            settings.trader_min_expected_edge_bps
            if not settings.trader_aggressive_mode
            else (settings.trader_min_expected_edge_bps * Decimal("0.70")).quantize(Decimal("0.01"))
        )
        effective_cooldown = 0 if settings.trader_aggressive_mode else settings.trader_min_minutes_between_trades
        self._market_data = {
            product_id: MarketDataClient(
                product_id=product_id,
                granularity=settings.effective_granularity,
                lookback_candles=settings.trader_lookback_candles,
                timeout_seconds=settings.trader_api_timeout_seconds,
                retries=settings.trader_marketdata_retries,
            )
            for product_id in self._product_ids
        }
        self._store = StateStore(
            db_path=settings.state_db_path,
            starting_cash=settings.trader_paper_starting_cash,
            bootstrap_product_id=self._product_ids[0],
        )
        self._risk_manager = RiskManager(
            RiskLimits(
                max_notional_per_trade=settings.trader_max_notional_per_trade,
                max_position_notional=settings.trader_max_position_notional,
                max_daily_loss=settings.trader_max_daily_loss,
                max_drawdown_pct=settings.trader_max_drawdown_pct,
                max_slippage_bps=settings.trader_max_slippage_bps,
                min_confidence=effective_min_confidence,
                fee_bps=settings.trader_fee_bps,
                min_expected_edge_bps=effective_min_edge,
                max_trades_per_day=settings.trader_max_trades_per_day,
                min_minutes_between_trades=effective_cooldown,
                volatility_hard_limit=settings.trader_volatility_hard_limit,
            )
        )
        self._decision_engine = OpenAIDecisionEngine(
            api_key=settings.effective_openai_api_key,
            model=settings.openai_model,
            max_notional_per_trade=settings.trader_max_notional_per_trade,
            volatility_target=settings.trader_volatility_target,
            feature_correlation_threshold=settings.trader_feature_correlation_threshold,
            aggressive_mode=settings.trader_aggressive_mode,
            aggressive_size_multiplier=settings.trader_aggressive_size_multiplier,
            aggressive_hold_to_trade_edge_bps=settings.trader_aggressive_hold_to_trade_edge_bps,
        )
        self._broker = self._build_broker()

    def run_once(self) -> ExecutionReport:
        snapshots: list[ProductSnapshot] = []
        for product_id, client in self._market_data.items():
            try:
                snapshots.append(client.fetch_snapshot())
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to fetch snapshot for %s: %s", product_id, exc)
        if not snapshots:
            return ExecutionReport(
                product_id=self._product_ids[0],
                status="noop",
                side="hold",
                reason="No market snapshots available in this cycle.",
            )
        prices = {snapshot.product_id: snapshot.price for snapshot in snapshots}
        candidates: list[tuple[Decimal, ProductSnapshot, TradingDecision]] = []
        for snapshot in snapshots:
            portfolio = self._store.load_portfolio(product_id=snapshot.product_id, last_prices=prices)
            stop_loss_decision = self._build_stop_loss_decision(snapshot, portfolio)
            if stop_loss_decision is not None:
                candidates.append((Decimal("9999999"), snapshot, stop_loss_decision))
                continue
            decision = self._decision_engine.decide(snapshot, portfolio, self._risk_manager.context())
            score = (decision.expected_edge_bps * Decimal(str(max(decision.confidence, 0.01)))).quantize(Decimal("0.01"))
            candidates.append((score, snapshot, decision))
        candidates = self._inject_probe_trade_if_all_hold(candidates, prices)

        executions: list[ExecutionReport] = []
        for _, snapshot, decision in sorted(candidates, key=lambda item: item[0], reverse=True):
            if len(executions) >= self._settings.trader_max_trades_per_cycle:
                break
            portfolio = self._store.load_portfolio(product_id=snapshot.product_id, last_prices=prices)
            if decision.action == TradeAction.SELL and portfolio.position.is_flat:
                decision = decision.model_copy(
                    update={
                        "action": TradeAction.HOLD,
                        "size_quote": Decimal("0"),
                        "thesis": "Converted sell to hold because no position is open.",
                        "risk_notes": decision.risk_notes + ["No position to sell."],
                    }
                )
            if decision.action == TradeAction.HOLD:
                hold_report = ExecutionReport(
                    product_id=snapshot.product_id,
                    status="noop",
                    side=decision.action,
                    reason="Model chose hold; skipping execution slot.",
                )
                self._store.record_execution(decision, hold_report)
                logger.info(
                    "No trade for %s: hold decision (confidence=%.2f directional_edge=%s bps expected_edge=%s bps).",
                    snapshot.product_id,
                    decision.confidence,
                    decision.directional_edge_bps,
                    decision.expected_edge_bps,
                )
                continue
            activity = self._store.trade_activity()
            allowed, reason = self._risk_manager.validate(decision, portfolio, snapshot, activity)
            if not allowed:
                logger.warning("Decision rejected for %s: %s", snapshot.product_id, reason)
                rejected = ExecutionReport(
                    product_id=snapshot.product_id,
                    status="rejected",
                    side=decision.action,
                    reason=reason,
                )
                self._store.record_execution(decision, rejected)
                continue
            try:
                report = self._broker.execute(decision, portfolio, snapshot)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Broker execution failed for %s: %s", snapshot.product_id, exc)
                rejected = ExecutionReport(
                    product_id=snapshot.product_id,
                    status="rejected",
                    side=decision.action,
                    reason=f"Broker execution error: {exc}",
                )
                self._store.record_execution(decision, rejected)
                continue
            if report.status == "filled":
                executions.append(report)
                logger.info("Executed %s trade on %s status=%s", decision.action.value, snapshot.product_id, report.status)
            else:
                logger.info("No fill on %s action=%s status=%s", snapshot.product_id, decision.action.value, report.status)

        if executions:
            return executions[0]
        return ExecutionReport(
            product_id=self._product_ids[0],
            status="noop",
            side=TradeAction.HOLD,
            reason="No trade passed risk checks in this cycle.",
        )

    def _build_stop_loss_decision(
        self,
        snapshot: ProductSnapshot,
        portfolio: PortfolioState,
    ) -> TradingDecision | None:
        stop_loss_pct = self._settings.trader_stop_loss_pct
        if stop_loss_pct <= 0:
            return None
        position = portfolio.position
        if position.is_flat or position.average_entry_price <= 0:
            return None
        trigger_price = position.average_entry_price * (Decimal("1") - stop_loss_pct)
        if snapshot.price > trigger_price:
            return None
        sell_notional = position.quantity * snapshot.best_bid
        logger.warning(
            "Stop loss triggered for %s: price=%s entry=%s trigger=%s size_quote=%s.",
            snapshot.product_id,
            snapshot.price,
            position.average_entry_price,
            trigger_price,
            sell_notional,
        )
        return TradingDecision(
            product_id=snapshot.product_id,
            action=TradeAction.SELL,
            confidence=1.0,
            size_quote=sell_notional,
            thesis=f"Deterministic stop loss triggered at {stop_loss_pct:.2%} below average entry.",
            risk_notes=[
                "Forced exit: bypasses normal entry filters to reduce downside.",
                f"Entry={position.average_entry_price}; trigger={trigger_price}; mark={snapshot.price}.",
            ],
            stop_loss_pct=float(stop_loss_pct),
            expected_edge_bps=Decimal("999999"),
            directional_edge_bps=Decimal("-999999"),
            force_exit=True,
        )

    def loop(self) -> None:
        if self._settings.trader_loop_seconds <= 0:
            raise ValueError("TRADER_LOOP_SECONDS must be positive for loop mode.")
        while True:
            self.run_once()
            time.sleep(self._settings.trader_loop_seconds)

    def _inject_probe_trade_if_all_hold(
        self,
        candidates: list[tuple[Decimal, ProductSnapshot, TradingDecision]],
        prices: dict[str, Decimal],
    ) -> list[tuple[Decimal, ProductSnapshot, TradingDecision]]:
        if not self._settings.trader_probe_trade_on_all_hold:
            return candidates
        if not candidates:
            return candidates
        if any(decision.action != TradeAction.HOLD for _, _, decision in candidates):
            return candidates

        best_score, best_snapshot, best_decision = max(
            candidates,
            key=lambda item: abs(item[2].directional_edge_bps),
        )
        portfolio = self._store.load_portfolio(product_id=best_snapshot.product_id, last_prices=prices)
        probe_action = TradeAction.BUY
        if not portfolio.position.is_flat and best_decision.directional_edge_bps < 0:
            probe_action = TradeAction.SELL
        probe_notional = min(self._settings.trader_probe_trade_notional, self._settings.trader_max_notional_per_trade)
        probe_decision = best_decision.model_copy(
            update={
                "action": probe_action,
                "probe_trade": True,
                "confidence": max(best_decision.confidence, 0.55),
                "size_quote": probe_notional,
                "expected_edge_bps": max(best_decision.expected_edge_bps, Decimal("1")),
                "thesis": "Probe trade injected because all symbols were hold.",
                "risk_notes": best_decision.risk_notes + ["Probe trade mode: forced small trade to gather execution feedback."],
            }
        )
        logger.warning(
            "Probe trade injected on %s action=%s size=%s because all decisions were hold.",
            best_snapshot.product_id,
            probe_action.value,
            probe_notional,
        )
        updated_candidates = [item for item in candidates if item[1].product_id != best_snapshot.product_id]
        updated_candidates.append((Decimal("999999"), best_snapshot, probe_decision))
        return updated_candidates

    def _build_broker(self) -> Broker:
        if self._settings.trader_mode == TraderMode.PAPER:
            return PaperBroker(self._store, fee_bps=self._settings.trader_fee_bps)
        if not self._settings.trader_live_trading_enabled:
            raise ValueError("Refusing live broker initialization because TRADER_LIVE_TRADING_ENABLED is false.")
        key, secret = self._settings.validate_live_coinbase_credentials()
        return LiveBroker(
            api_key=key,
            api_secret=secret,
            use_sandbox=self._settings.coinbase_use_sandbox,
        )

