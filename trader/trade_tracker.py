"""
交易跟踪器模块，用于实时跟踪交易结果和风险指标
"""
import sqlite3
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd


class TradeTracker:
    """
    交易跟踪器，用于实时跟踪交易结果和风险指标
    """
    
    def __init__(self, db_path: str = "trader_state.db"):
        self.db_path = db_path
        self._init_db()
        
    def _init_db(self):
        """初始化数据库表"""
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.execute("PRAGMA journal_mode=WAL;")
            self.conn.row_factory = sqlite3.Row
            self.cursor = self.conn.cursor()
            
            # 创建交易跟踪表
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS trade_tracking (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_id TEXT UNIQUE,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    entry_time TEXT NOT NULL,
                    exit_time TEXT,
                    entry_price REAL NOT NULL,
                    exit_price REAL,
                    quantity REAL NOT NULL,
                    pnl REAL DEFAULT 0,
                    pnl_pct REAL DEFAULT 0,
                    is_profit INTEGER DEFAULT 0,
                    drawdown REAL DEFAULT 0,
                    risk_level TEXT DEFAULT 'LOW',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # 创建风险事件记录表
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS risk_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    event_description TEXT,
                    severity_level TEXT NOT NULL, -- LOW, MEDIUM, HIGH, CRITICAL
                    affected_trades TEXT, -- JSON format for multiple trades
                    detected_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    resolved_at TEXT,
                    is_resolved INTEGER DEFAULT 0
                )
            """)
            
            self.conn.commit()
            logging.info("TradeTracker database initialized.")
        except sqlite3.Error as e:
            logging.error(f"Failed to initialize TradeTracker database: {e}", exc_info=True)
            raise

    def record_trade_entry(self, trade_id: str, symbol: str, side: str, 
                          entry_price: float, quantity: float):
        """记录交易进入"""
        try:
            current_time = datetime.utcnow().isoformat()
            self.cursor.execute("""
                INSERT OR REPLACE INTO trade_tracking 
                (trade_id, symbol, side, entry_time, entry_price, quantity)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (trade_id, symbol, side, current_time, entry_price, quantity))
            
            self.conn.commit()
            logging.info(f"Recorded trade entry: {trade_id} for {symbol}")
        except sqlite3.Error as e:
            logging.error(f"Failed to record trade entry: {e}", exc_info=True)

    def record_trade_exit(self, trade_id: str, exit_price: float):
        """记录交易退出并计算盈亏"""
        try:
            current_time = datetime.utcnow().isoformat()
            
            # 获取交易 entry 信息
            self.cursor.execute("""
                SELECT entry_price, quantity, symbol, side FROM trade_tracking 
                WHERE trade_id = ?
            """, (trade_id,))
            
            row = self.cursor.fetchone()
            if not row:
                logging.error(f"Trade with ID {trade_id} not found in tracking table")
                return
            
            entry_price = row['entry_price']
            quantity = row['quantity']
            
            # 计算盈亏
            pnl = (exit_price - entry_price) * quantity
            pnl_pct = (exit_price - entry_price) / entry_price if entry_price != 0 else 0
            is_profit = 1 if pnl > 0 else 0
            
            # 更新交易记录
            self.cursor.execute("""
                UPDATE trade_tracking 
                SET exit_time = ?, exit_price = ?, pnl = ?, pnl_pct = ?, is_profit = ?
                WHERE trade_id = ?
            """, (current_time, exit_price, pnl, pnl_pct, is_profit, trade_id))
            
            self.conn.commit()
            logging.info(f"Recorded trade exit: {trade_id}, P&L: {pnl:.4f}")
            
            # 检查是否需要记录风险事件
            self._check_and_record_risk_events(trade_id, pnl, pnl_pct)
            
        except sqlite3.Error as e:
            logging.error(f"Failed to record trade exit: {e}", exc_info=True)

    def _check_and_record_risk_events(self, trade_id: str, pnl: float, pnl_pct: float):
        """检查并记录风险事件"""
        try:
            # 检查单笔交易亏损是否过大
            if pnl_pct < -0.03:  # 3% 亏损
                severity = 'HIGH' if pnl_pct < -0.05 else 'MEDIUM'  # 5% 亏损为高风险
                self._log_risk_event(
                    'HIGH_LOSS_TRADE',
                    f'Trade {trade_id} resulted in significant loss: {pnl_pct*100:.2f}%',
                    severity,
                    [trade_id]
                )
                
            # 检查连续亏损
            consecutive_losses = self._get_consecutive_losses_count()
            if consecutive_losses >= 3:
                self._log_risk_event(
                    'CONSECUTIVE_LOSSES',
                    f'Detected {consecutive_losses} consecutive losses',
                    'HIGH' if consecutive_losses >= 5 else 'MEDIUM',
                    []
                )
                
        except Exception as e:
            logging.error(f"Failed to check risk events: {e}", exc_info=True)

    def _log_risk_event(self, event_type: str, description: str, 
                       severity: str, affected_trades: List[str]):
        """记录风险事件"""
        try:
            affected_trades_str = ','.join(affected_trades) if affected_trades else ''
            self.cursor.execute("""
                INSERT INTO risk_events 
                (event_type, event_description, severity_level, affected_trades)
                VALUES (?, ?, ?, ?)
            """, (event_type, description, severity, affected_trades_str))
            
            self.conn.commit()
            logging.warning(f"Risk event logged: {event_type} - {description}")
        except sqlite3.Error as e:
            logging.error(f"Failed to log risk event: {e}", exc_info=True)

    def get_daily_pnl(self) -> float:
        """获取当天盈亏"""
        try:
            today_start = datetime.utcnow().date().isoformat()
            self.cursor.execute("""
                SELECT SUM(pnl) as daily_pnl 
                FROM trade_tracking 
                WHERE DATE(entry_time) = ? AND exit_time IS NOT NULL
            """, (today_start,))
            
            result = self.cursor.fetchone()
            return result['daily_pnl'] or 0.0
        except sqlite3.Error as e:
            logging.error(f"Failed to get daily P&L: {e}", exc_info=True)
            return 0.0

    def get_consecutive_losses_count(self) -> int:
        """获取连续亏损次数"""
        return self._get_consecutive_losses_count()
    
    def _get_consecutive_losses_count(self) -> int:
        """获取连续亏损次数的内部实现"""
        try:
            # 获取最近的交易，按时间倒序排列
            self.cursor.execute("""
                SELECT is_profit 
                FROM trade_tracking 
                WHERE exit_time IS NOT NULL
                ORDER BY entry_time DESC
            """)
            
            trades = self.cursor.fetchall()
            consecutive_losses = 0
            
            for trade in trades:
                if trade['is_profit'] == 0:  # 亏损
                    consecutive_losses += 1
                else:  # 盈利
                    break
                    
            return consecutive_losses
        except sqlite3.Error as e:
            logging.error(f"Failed to get consecutive losses count: {e}", exc_info=True)
            return 0

    def get_recent_performance(self, days: int = 7) -> Dict:
        """获取近期表现统计"""
        try:
            since_time = (datetime.utcnow() - timedelta(days=days)).isoformat()
            self.cursor.execute("""
                SELECT 
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN is_profit = 1 THEN 1 ELSE 0 END) as winning_trades,
                    SUM(pnl) as total_pnl,
                    AVG(pnl) as avg_pnl,
                    MIN(pnl) as min_pnl,
                    MAX(pnl) as max_pnl,
                    AVG(pnl_pct) as avg_pnl_pct
                FROM trade_tracking 
                WHERE entry_time >= ? AND exit_time IS NOT NULL
            """, (since_time,))
            
            result = self.cursor.fetchone()
            if result:
                total_trades = result['total_trades']
                win_rate = result['winning_trades'] / total_trades if total_trades > 0 else 0
                return {
                    'total_trades': total_trades,
                    'winning_trades': result['winning_trades'],
                    'win_rate': win_rate,
                    'total_pnl': result['total_pnl'] or 0.0,
                    'avg_pnl': result['avg_pnl'] or 0.0,
                    'min_pnl': result['min_pnl'] or 0.0,
                    'max_pnl': result['max_pnl'] or 0.0,
                    'avg_pnl_pct': result['avg_pnl_pct'] or 0.0
                }
            else:
                return {
                    'total_trades': 0,
                    'winning_trades': 0,
                    'win_rate': 0.0,
                    'total_pnl': 0.0,
                    'avg_pnl': 0.0,
                    'min_pnl': 0.0,
                    'max_pnl': 0.0,
                    'avg_pnl_pct': 0.0
                }
        except sqlite3.Error as e:
            logging.error(f"Failed to get recent performance: {e}", exc_info=True)
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'win_rate': 0.0,
                'total_pnl': 0.0,
                'avg_pnl': 0.0,
                'min_pnl': 0.0,
                'max_pnl': 0.0,
                'avg_pnl_pct': 0.0
            }

    def get_unresolved_risk_events(self) -> List[Dict]:
        """获取未解决的风险事件"""
        try:
            self.cursor.execute("""
                SELECT * FROM risk_events 
                WHERE is_resolved = 0 
                ORDER BY detected_at DESC
            """)
            
            rows = self.cursor.fetchall()
            return [dict(row) for row in rows]
        except sqlite3.Error as e:
            logging.error(f"Failed to get unresolved risk events: {e}", exc_info=True)
            return []

    def mark_risk_event_resolved(self, event_id: int):
        """标记风险事件为已解决"""
        try:
            resolved_at = datetime.utcnow().isoformat()
            self.cursor.execute("""
                UPDATE risk_events 
                SET is_resolved = 1, resolved_at = ? 
                WHERE id = ?
            """, (resolved_at, event_id))
            
            self.conn.commit()
            logging.info(f"Risk event {event_id} marked as resolved")
        except sqlite3.Error as e:
            logging.error(f"Failed to mark risk event as resolved: {e}", exc_info=True)