import logging
from datetime import datetime, timedelta
import sqlite3
import os


class RiskManager:
    """
    A comprehensive risk management module to check trade orders against predefined rules.
    """
    def __init__(self, balance, config, db_path="trader_state.db"):
        """
        Initializes the RiskManager.

        :param balance: The current total account balance.
        :param config: A dictionary containing risk management settings.
                       Expected keys: 'position_sizing', 'risk_management'.
        :param db_path: Path to the database for storing trade records
        """
        self.balance = balance
        self.config = config
        self.db_path = db_path
        
        # Position sizing configuration
        self.position_sizing_strategy = config.get('position_sizing', {}).get('strategy', 'fixed')
        self.fixed_quantity = config.get('position_sizing', {}).get('fixed_quantity', 0.001)
        self.risk_per_trade = config.get('position_sizing', {}).get('risk_per_trade', 0.01)
        
        # Risk management configuration
        risk_config = config.get('risk_management', {})
        self.daily_loss_limit = risk_config.get('daily_loss_limit', 0.05)  # 5% daily loss limit
        self.weekly_loss_limit = risk_config.get('weekly_loss_limit', 0.10)  # 10% weekly loss limit
        self.max_consecutive_losses = risk_config.get('max_consecutive_losses', 3)
        self.max_position_percentage = risk_config.get('max_position_percentage', 0.20)  # 20% max position size
        
        # Initialize database connection to track trades and P&L
        self._init_db()
        
        logging.info("RiskManager initialized.")
        logging.info(f"Position sizing strategy: {self.position_sizing_strategy}")
        logging.info(f"Daily loss limit: {self.daily_loss_limit*100:.2f}%")
        logging.info(f"Weekly loss limit: {self.weekly_loss_limit*100:.2f}%")
        logging.info(f"Max consecutive losses: {self.max_consecutive_losses}")
        logging.info(f"Max position percentage: {self.max_position_percentage*100:.2f}%")

    def _init_db(self):
        """Initialize the database and create necessary tables if they don't exist."""
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row  # Enable column access by name
            self.cursor = self.conn.cursor()
            
            # Create table for tracking trades and their outcomes
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS trade_outcomes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_id TEXT,
                    symbol TEXT NOT NULL,
                    entry_time TEXT NOT NULL,
                    exit_time TEXT,
                    entry_price REAL NOT NULL,
                    exit_price REAL,
                    quantity REAL NOT NULL,
                    pnl REAL DEFAULT 0,
                    pnl_pct REAL DEFAULT 0,
                    is_profit INTEGER DEFAULT 0
                )
            """)
            self.conn.commit()
            logging.info("Trade outcomes table initialized.")
        except sqlite3.Error as e:
            logging.error(f"Failed to initialize database: {e}", exc_info=True)
            raise

    def record_trade_outcome(self, trade_id, symbol, entry_time, exit_time, entry_price, exit_price, quantity):
        """
        Record the outcome of a completed trade.
        
        :param trade_id: Unique identifier for the trade
        :param symbol: Trading symbol
        :param entry_time: Time when the trade was entered
        :param exit_time: Time when the trade was exited
        :param entry_price: Entry price
        :param exit_price: Exit price
        :param quantity: Quantity traded
        """
        try:
            # Calculate P&L
            pnl = (exit_price - entry_price) * quantity
            pnl_pct = (exit_price - entry_price) / entry_price if entry_price != 0 else 0
            is_profit = 1 if pnl > 0 else 0

            self.cursor.execute("""
                INSERT INTO trade_outcomes (trade_id, symbol, entry_time, exit_time, entry_price, 
                                          exit_price, quantity, pnl, pnl_pct, is_profit)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (trade_id, symbol, entry_time, exit_time, entry_price, exit_price, quantity, pnl, pnl_pct, is_profit))
            self.conn.commit()
            logging.info(f"Recorded trade outcome: P&L = {pnl:.4f} ({pnl_pct*100:.2f}%)")
        except sqlite3.Error as e:
            logging.error(f"Failed to record trade outcome: {e}", exc_info=True)

    def _get_period_pnl(self, period_hours):
        """
        Get P&L for the specified period in hours.
        
        :param period_hours: Number of hours to look back
        :return: Total P&L in the period
        """
        try:
            since_time = (datetime.utcnow() - timedelta(hours=period_hours)).isoformat()
            self.cursor.execute("""
                SELECT SUM(pnl) as total_pnl 
                FROM trade_outcomes 
                WHERE entry_time >= ?
            """, (since_time,))
            result = self.cursor.fetchone()
            return result['total_pnl'] or 0.0
        except sqlite3.Error as e:
            logging.error(f"Failed to get P&L for period: {e}", exc_info=True)
            return 0.0

    def _get_daily_pnl(self):
        """Get P&L for the current day."""
        return self._get_period_pnl(24)

    def _get_weekly_pnl(self):
        """Get P&L for the current week (7 days)."""
        return self._get_period_pnl(24 * 7)

    def _get_consecutive_losses_count(self):
        """
        Get the count of consecutive losing trades.
        
        :return: Number of consecutive losses
        """
        try:
            # Get recent trades ordered by time (most recent first)
            self.cursor.execute("""
                SELECT is_profit 
                FROM trade_outcomes 
                ORDER BY entry_time DESC
            """)
            trades = self.cursor.fetchall()
            
            consecutive_losses = 0
            for trade in trades:
                if trade['is_profit'] == 0:  # Loss
                    consecutive_losses += 1
                else:  # Profit
                    break
            
            return consecutive_losses
        except sqlite3.Error as e:
            logging.error(f"Failed to get consecutive losses count: {e}", exc_info=True)
            return 0

    def _get_current_position_value(self, current_price):
        """
        Get the current value of open positions relative to account balance.
        
        :param current_price: Current market price for the asset
        :return: Position value as a percentage of account balance
        """
        # This would need integration with StateManager to get actual position
        # For now, we'll return a placeholder - in a real implementation, 
        # this would check the actual position from StateManager
        return 0.0  # Placeholder implementation

    def calculate_order_size(self, price):
        """
        Calculates the quantity for an order based on the position sizing strategy.

        :param price: The current price of the asset.
        :return: The calculated quantity for the trade.
        """
        if self.position_sizing_strategy == 'fixed':
            return self.fixed_quantity
        elif self.position_sizing_strategy == 'fractional':
            # Fractional sizing based on risk percentage
            # This is a simplified example. A real implementation would also consider stop-loss distance.
            return (self.balance * self.risk_per_trade) / price
        else:
            logging.warning(f"Unknown position sizing strategy: {self.position_sizing_strategy}. Defaulting to fixed size.")
            return self.fixed_quantity

    def assess_trade(self, proposed_quantity, current_price=None, symbol="BTC-USDT"):
        """
        Assess a proposed trade against all risk management rules.
        
        :param proposed_quantity: The quantity of the asset to be traded
        :param current_price: Current market price (for position size calculation)
        :param symbol: Trading symbol
        :return: tuple(bool, str) - (is_approved, reason)
        """
        # Check 1: Basic validation - trade size is positive
        if proposed_quantity <= 0:
            reason = f"Trade rejected: Proposed quantity ({proposed_quantity}) is not positive."
            logging.warning(reason)
            return False, reason

        # Check 2: Daily loss limit
        daily_pnl = self._get_daily_pnl()
        daily_loss_pct = abs(daily_pnl) / self.balance if self.balance > 0 else 0
        if daily_pnl < 0 and daily_loss_pct >= self.daily_loss_limit:
            reason = f"Trade rejected: Daily loss limit exceeded. Today's loss: {daily_loss_pct*100:.2f}% (limit: {self.daily_loss_limit*100:.2f}%)"
            logging.warning(reason)
            return False, reason

        # Check 3: Weekly loss limit
        weekly_pnl = self._get_weekly_pnl()
        weekly_loss_pct = abs(weekly_pnl) / self.balance if self.balance > 0 else 0
        if weekly_pnl < 0 and weekly_loss_pct >= self.weekly_loss_limit:
            reason = f"Trade rejected: Weekly loss limit exceeded. This week's loss: {weekly_loss_pct*100:.2f}% (limit: {self.weekly_loss_limit*100:.2f}%)"
            logging.warning(reason)
            return False, reason

        # Check 4: Consecutive losses limit
        consecutive_losses = self._get_consecutive_losses_count()
        if consecutive_losses >= self.max_consecutive_losses:
            reason = f"Trade rejected: Max consecutive losses reached ({consecutive_losses}/{self.max_consecutive_losses})"
            logging.warning(reason)
            return False, reason

        # Check 5: Maximum position size relative to account balance
        if current_price and proposed_quantity > 0:
            proposed_position_value = proposed_quantity * current_price
            position_pct = proposed_position_value / self.balance if self.balance > 0 else 0
            
            if position_pct > self.max_position_percentage:
                reason = f"Trade rejected: Position size would be {position_pct*100:.2f}% of account (limit: {self.max_position_percentage*100:.2f}%)"
                logging.warning(reason)
                return False, reason

        return True, f"Trade approved. Daily P&L: {self._get_daily_pnl():.4f}, Weekly P&L: {self._get_weekly_pnl():.4f}, Consecutive losses: {self._get_consecutive_losses_count()}"

    def calculate_order_size_with_volatility_adjustment(self, price, market_data=None):
        """
        Calculate order size with adjustment based on market volatility.
        
        :param price: Current price of the asset
        :param market_data: DataFrame with recent market data for volatility calculation
        :return: Adjusted quantity for the trade
        """
        # First calculate base order size
        base_size = self.calculate_order_size(price)
        
        if market_data is None or len(market_data) < 2:
            # If no market data provided or insufficient data, return base size
            return base_size
        
        try:
            # Calculate volatility as standard deviation of returns
            returns = market_data['close'].pct_change().dropna()
            if len(returns) < 2:
                return base_size
                
            volatility = returns.std()
            
            # Set up volatility thresholds
            low_volatility_threshold = 0.02  # 2% daily volatility
            high_volatility_threshold = 0.05  # 5% daily volatility
            
            # Adjust position size based on volatility
            if volatility > high_volatility_threshold:
                # High volatility: reduce position size by 50%
                adjusted_size = base_size * 0.5
                logging.info(f"High volatility detected ({volatility*100:.2f}%), reducing position size to {adjusted_size}")
            elif volatility > low_volatility_threshold:
                # Medium volatility: reduce position size by 25%
                adjusted_size = base_size * 0.75
                logging.info(f"Medium volatility detected ({volatility*100:.2f}%), reducing position size to {adjusted_size}")
            else:
                # Low volatility: use normal position size
                adjusted_size = base_size
                logging.info(f"Low volatility detected ({volatility*100:.2f}%), using normal position size {adjusted_size}")
                
            return adjusted_size
            
        except Exception as e:
            logging.warning(f"Error calculating volatility-adjusted position size: {e}. Using base position size.")
            return base_size


    def calculate_sl_tp(self, entry_price, side, volatility=None, atr_multiple=2.0):
        """
        Calculate Stop Loss and Take Profit prices.
        
        :param entry_price: The entry price of the trade
        :param side: 'buy' or 'sell'
        :param volatility: Current volatility (e.g., ATR). If None, uses fixed percentage.
        :param atr_multiple: Multiplier for ATR-based SL/TP
        :return: tuple(stop_loss_price, take_profit_price)
        """
        # Default fixed percentage if volatility is not provided
        sl_pct = 0.02 # 2% stop loss
        tp_pct = 0.04 # 4% take profit (1:2 risk-reward)
        
        if volatility:
            # Dynamic SL based on volatility
            sl_distance = volatility * atr_multiple
            tp_distance = sl_distance * 2 # 1:2 risk-reward
            
            if side == 'buy':
                stop_loss = entry_price - sl_distance
                take_profit = entry_price + tp_distance
            else:
                stop_loss = entry_price + sl_distance
                take_profit = entry_price - tp_distance
        else:
            # Fixed percentage fallback
            if side == 'buy':
                stop_loss = entry_price * (1 - sl_pct)
                take_profit = entry_price * (1 + tp_pct)
            else:
                stop_loss = entry_price * (1 + sl_pct)
                take_profit = entry_price * (1 - tp_pct)
                
        return stop_loss, take_profit
