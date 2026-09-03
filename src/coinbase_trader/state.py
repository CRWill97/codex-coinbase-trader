from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Iterator

from coinbase_trader.models import ExecutionReport, PortfolioState, Position, TradeAction, TradingDecision
from coinbase_trader.risk import TradeActivity


class StateStore:
    def __init__(self, db_path: Path, starting_cash: Decimal, bootstrap_product_id: str) -> None:
        self._db_path = db_path
        self._starting_cash = starting_cash
        self._bootstrap_product_id = bootstrap_product_id
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self._db_path)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS account (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    cash TEXT NOT NULL,
                    realized_pnl TEXT NOT NULL,
                    peak_equity TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS positions (
                    product_id TEXT PRIMARY KEY,
                    quantity TEXT NOT NULL,
                    average_entry_price TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS trade_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    product_id TEXT NOT NULL,
                    side TEXT NOT NULL,
                    status TEXT NOT NULL,
                    filled_quote TEXT NOT NULL,
                    filled_base TEXT NOT NULL,
                    average_price TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    external_order_id TEXT,
                    decision_json TEXT NOT NULL
                )
                """
            )
            self._ensure_trade_log_product_id(conn)
            self._seed_account(conn)

    def _ensure_trade_log_product_id(self, conn: sqlite3.Connection) -> None:
        columns = conn.execute("PRAGMA table_info(trade_log)").fetchall()
        if not columns:
            return
        names = {row["name"] for row in columns}
        if "product_id" not in names:
            conn.execute(
                """
                ALTER TABLE trade_log
                ADD COLUMN product_id TEXT NOT NULL DEFAULT 'BTC-USD'
                """
            )

    def _seed_account(self, conn: sqlite3.Connection) -> None:
        account = conn.execute("SELECT 1 FROM account WHERE id = 1").fetchone()
        if account is not None:
            return
        legacy_table = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='portfolio'"
        ).fetchone()
        legacy = None
        if legacy_table is not None:
            legacy = conn.execute(
                """
                SELECT cash, quantity, average_entry_price, realized_pnl, peak_equity
                FROM portfolio
                WHERE id = 1
                """
            ).fetchone()
        if legacy is None:
            cash = self._starting_cash
            realized = Decimal("0")
            peak = self._starting_cash
            quantity = Decimal("0")
            avg = Decimal("0")
        else:
            cash = Decimal(legacy["cash"])
            realized = Decimal(legacy["realized_pnl"])
            peak = Decimal(legacy["peak_equity"])
            quantity = Decimal(legacy["quantity"])
            avg = Decimal(legacy["average_entry_price"])

        now = datetime.now(UTC).isoformat()
        conn.execute(
            """
            INSERT INTO account (id, cash, realized_pnl, peak_equity, updated_at)
            VALUES (1, ?, ?, ?, ?)
            """,
            (str(cash), str(realized), str(peak), now),
        )
        if quantity > 0:
            conn.execute(
                """
                INSERT OR REPLACE INTO positions (product_id, quantity, average_entry_price, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                (self._bootstrap_product_id, str(quantity), str(avg), now),
            )

    def load_portfolio(self, product_id: str, last_prices: dict[str, Decimal]) -> PortfolioState:
        with self._connect() as conn:
            account = conn.execute("SELECT * FROM account WHERE id = 1").fetchone()
            if account is None:
                raise RuntimeError("Account state is missing.")
            position = conn.execute(
                "SELECT quantity, average_entry_price FROM positions WHERE product_id = ?",
                (product_id,),
            ).fetchone()
            quantity = Decimal(position["quantity"]) if position is not None else Decimal("0")
            average_entry_price = Decimal(position["average_entry_price"]) if position is not None else Decimal("0")
            cash = Decimal(account["cash"])
            realized_pnl = Decimal(account["realized_pnl"])
            peak_equity = Decimal(account["peak_equity"])

            equity = cash + self._positions_market_value(conn, last_prices)
            daily_realized_pnl = self._calculate_daily_realized_pnl(conn, datetime.now(UTC).date())
            if equity > peak_equity:
                peak_equity = equity
                conn.execute(
                    "UPDATE account SET peak_equity = ?, updated_at = ? WHERE id = 1",
                    (str(peak_equity), datetime.now(UTC).isoformat()),
                )

            return PortfolioState(
                cash=cash,
                position=Position(product_id=product_id, quantity=quantity, average_entry_price=average_entry_price),
                realized_pnl=realized_pnl,
                equity=equity,
                peak_equity=peak_equity,
                daily_realized_pnl=daily_realized_pnl,
                last_price=last_prices.get(product_id, Decimal("0")),
            )

    def _positions_market_value(self, conn: sqlite3.Connection, last_prices: dict[str, Decimal]) -> Decimal:
        positions = conn.execute("SELECT product_id, quantity, average_entry_price FROM positions").fetchall()
        total = Decimal("0")
        for row in positions:
            quantity = Decimal(row["quantity"])
            if quantity <= 0:
                continue
            mark = last_prices.get(row["product_id"], Decimal(row["average_entry_price"]))
            total += quantity * mark
        return total

    def record_execution(self, decision: TradingDecision, report: ExecutionReport) -> None:
        with self._connect() as conn:
            account = conn.execute("SELECT * FROM account WHERE id = 1").fetchone()
            if account is None:
                raise RuntimeError("Account state is missing.")
            cash = Decimal(account["cash"])
            realized_pnl = Decimal(account["realized_pnl"])

            position = conn.execute(
                "SELECT quantity, average_entry_price FROM positions WHERE product_id = ?",
                (decision.product_id,),
            ).fetchone()
            quantity = Decimal(position["quantity"]) if position is not None else Decimal("0")
            average_entry_price = Decimal(position["average_entry_price"]) if position is not None else Decimal("0")

            if report.status == "filled":
                if report.side == TradeAction.BUY:
                    new_quantity = quantity + report.filled_base
                    new_cost = (quantity * average_entry_price) + report.filled_quote
                    average_entry_price = Decimal("0") if new_quantity == 0 else new_cost / new_quantity
                    quantity = new_quantity
                    cash -= report.filled_quote
                elif report.side == TradeAction.SELL:
                    quantity -= report.filled_base
                    cash += report.filled_quote
                    realized_pnl += report.filled_quote - (average_entry_price * report.filled_base)
                    if quantity <= 0:
                        quantity = Decimal("0")
                        average_entry_price = Decimal("0")

                conn.execute(
                    """
                    INSERT OR REPLACE INTO positions (product_id, quantity, average_entry_price, updated_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        decision.product_id,
                        str(quantity),
                        str(average_entry_price),
                        datetime.now(UTC).isoformat(),
                    ),
                )
                conn.execute(
                    """
                    UPDATE account
                    SET cash = ?, realized_pnl = ?, updated_at = ?
                    WHERE id = 1
                    """,
                    (str(cash), str(realized_pnl), datetime.now(UTC).isoformat()),
                )

            conn.execute(
                """
                INSERT INTO trade_log
                (created_at, product_id, side, status, filled_quote, filled_base, average_price, reason, external_order_id, decision_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now(UTC).isoformat(),
                    decision.product_id,
                    report.side.value,
                    report.status,
                    str(report.filled_quote),
                    str(report.filled_base),
                    str(report.average_price),
                    report.reason,
                    report.external_order_id,
                    json.dumps(decision.model_dump(mode="json")),
                ),
            )

    def trade_activity(self) -> TradeActivity:
        now = datetime.now(UTC)
        today = now.date().isoformat()
        with self._connect() as conn:
            today_count = conn.execute(
                """
                SELECT COUNT(*) AS count
                FROM trade_log
                WHERE status = 'filled'
                  AND side IN ('buy', 'sell')
                  AND DATE(created_at) = ?
                """,
                (today,),
            ).fetchone()
            last_fill = conn.execute(
                """
                SELECT created_at
                FROM trade_log
                WHERE status = 'filled'
                  AND side IN ('buy', 'sell')
                ORDER BY created_at DESC
                LIMIT 1
                """
            ).fetchone()
        count = int(today_count["count"]) if today_count is not None else 0
        minutes_since_last_fill: int | None = None
        if last_fill is not None:
            last_dt = datetime.fromisoformat(last_fill["created_at"])
            minutes_since_last_fill = int((now - last_dt).total_seconds() // 60)
        return TradeActivity(
            filled_trades_today=count,
            minutes_since_last_fill=minutes_since_last_fill,
        )

    def _calculate_daily_realized_pnl(self, conn: sqlite3.Connection, current_date: date) -> Decimal:
        rows = conn.execute(
            """
            SELECT created_at, product_id, side, filled_quote, filled_base, average_price
            FROM trade_log
            WHERE status = 'filled'
            ORDER BY created_at ASC
            """
        ).fetchall()
        if not rows:
            return Decimal("0")

        quantity_by_product: dict[str, Decimal] = {}
        cost_by_product: dict[str, Decimal] = {}
        daily_pnl = Decimal("0")
        for row in rows:
            product_id = row["product_id"]
            side = TradeAction(row["side"])
            filled_quote = Decimal(row["filled_quote"])
            filled_base = Decimal(row["filled_base"])
            quantity = quantity_by_product.get(product_id, Decimal("0"))
            cost_basis = cost_by_product.get(product_id, Decimal("0"))
            if side == TradeAction.BUY:
                quantity_by_product[product_id] = quantity + filled_base
                cost_by_product[product_id] = cost_basis + filled_quote
                continue
            if side == TradeAction.SELL and quantity > 0:
                average_cost = cost_basis / quantity
                realized_trade = filled_quote - (average_cost * filled_base)
                quantity -= filled_base
                cost_basis -= average_cost * filled_base
                if quantity <= 0:
                    quantity = Decimal("0")
                    cost_basis = Decimal("0")
                quantity_by_product[product_id] = quantity
                cost_by_product[product_id] = cost_basis
                trade_date = datetime.fromisoformat(row["created_at"]).date()
                if trade_date == current_date:
                    daily_pnl += realized_trade
        return daily_pnl

