import sqlite3
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List

class StateManager:
    """
    Manages the persistent state of the trading bot using an SQLite database.
    This includes tracking open positions and order history.
    """

    def __init__(self, db_path: str = "trader_state.db"):
        """
        Initializes the StateManager and connects to the database.
        :param db_path: Path to the SQLite database file.
        """
        try:
            self.conn = sqlite3.connect(db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
            self.cursor = self.conn.cursor()
            self._create_tables()
            
            # Initialize TradeTracker for enhanced monitoring
            from trader.trade_tracker import TradeTracker
            self.trade_tracker = TradeTracker(db_path=db_path)
            
            logging.info(f"StateManager initialized with database: {db_path}")
        except sqlite3.Error as e:
            logging.error(f"Database connection failed: {e}", exc_info=True)
            raise

    def _create_tables(self):
        """
        Creates the necessary tables if they don't exist.
        """
        try:
            # Position table: one row per symbol
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS positions (
                    symbol TEXT PRIMARY KEY,
                    quantity REAL NOT NULL,
                    entry_price REAL NOT NULL,
                    last_update TEXT NOT NULL
                )
            """)

            # Orders table: logs every order placed
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS orders (
                    order_id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL, -- 'buy' or 'sell'
                    quantity REAL NOT NULL,
                    price REAL, -- Entry price for market orders, limit price for limit orders
                    status TEXT NOT NULL, -- e.g., 'open', 'filled', 'cancelled', 'failed'
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            self.conn.commit()
        except sqlite3.Error as e:
            logging.error(f"Failed to create tables: {e}", exc_info=True)
            self.conn.rollback()
            raise

    def update_position(self, symbol: str, quantity: float, entry_price: float):
        """
        Updates or creates a position for a given symbol.
        If quantity is 0, the position is removed.
        """
        now_utc = datetime.utcnow().isoformat()
        try:
            if quantity > 0:
                self.cursor.execute("""
                    INSERT INTO positions (symbol, quantity, entry_price, last_update)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(symbol) DO UPDATE SET
                    quantity = excluded.quantity,
                    entry_price = excluded.entry_price,
                    last_update = excluded.last_update;
                """, (symbol, quantity, entry_price, now_utc))
                logging.info(f"Position updated: {symbol}, Quantity: {quantity}, Entry Price: {entry_price}")
            else:
                self.cursor.execute("DELETE FROM positions WHERE symbol = ?", (symbol,))
                logging.info(f"Position closed and removed for symbol: {symbol}")
            self.conn.commit()
        except sqlite3.Error as e:
            logging.error(f"Failed to update position for {symbol}: {e}", exc_info=True)
            self.conn.rollback()

    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves the current position for a given symbol.
        """
        try:
            self.cursor.execute("SELECT * FROM positions WHERE symbol = ?", (symbol,))
            row = self.cursor.fetchone()
            return dict(row) if row else None
        except sqlite3.Error as e:
            logging.error(f"Failed to retrieve position for {symbol}: {e}", exc_info=True)
            return None

    def upsert_order(self, order_data: Dict[str, Any]):
        """
        Inserts or updates an order in the database.
        """
        now_utc = datetime.utcnow().isoformat()
        try:
            self.cursor.execute("""
                INSERT INTO orders (order_id, symbol, side, quantity, price, status, created_at, updated_at)
                VALUES (:order_id, :symbol, :side, :quantity, :price, :status, :created_at, :updated_at)
                ON CONFLICT(order_id) DO UPDATE SET
                status = excluded.status,
                updated_at = excluded.updated_at;
            """, {
                "order_id": order_data.get('order_id'),
                "symbol": order_data.get('symbol'),
                "side": order_data.get('side'),
                "quantity": order_data.get('quantity'),
                "price": order_data.get('price'),
                "status": order_data.get('status'),
                "created_at": order_data.get('created_at', now_utc),
                "updated_at": now_utc
            })
            self.conn.commit()
            logging.info(f"Order {order_data.get('order_id')} upserted with status {order_data.get('status')}")
        except sqlite3.Error as e:
            logging.error(f"Failed to upsert order {order_data.get('order_id')}: {e}", exc_info=True)
            self.conn.rollback()

    def get_open_orders(self, symbol: str) -> List[Dict[str, Any]]:
        """
        Retrieves all open orders for a given symbol.
        """
        try:
            self.cursor.execute("SELECT * FROM orders WHERE symbol = ? AND status = 'open'", (symbol,))
            rows = self.cursor.fetchall()
            return [dict(row) for row in rows]
        except sqlite3.Error as e:
            logging.error(f"Failed to get open orders for {symbol}: {e}", exc_info=True)
            return []

    def close(self):
        """
        Closes the database connection.
        """
        if self.conn:
            self.conn.close()
            logging.info("Database connection closed.")
