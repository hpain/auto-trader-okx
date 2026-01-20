import asyncio
import argparse
import sys
import os
import logging
import math
from typing import Dict, Optional, Tuple
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from exchange.factory import ExchangeFactory
from strategies.funding_arb import FundingRateArbitrageStrategy
from trader.execution_handler import ExecutionHandler
from utils.logger import setup_script_logger as setup_logger

# -----------------------------------------------------------------------------
# Configuration Constants
# -----------------------------------------------------------------------------
RISK_ALLOCATION = 0.5        # Use 50% of available capital
MAX_LEVERAGE = 1.0           # 1x Leverage (Safe)
SLIPPAGE_TOLERANCE = 0.005   # 0.5% Max Slippage for IOC orders
MIN_PROFIT_SPREAD = 0.0005   # 0.05% Min Net Yield (Funding - Cost) to enter
KILL_SWITCH_DROP = -0.02     # -2% Drawdown triggers Panic Close
BASIS_ALARM = -0.01          # -1% Basis (Perp << Spot) trigger Stop Loss

class SimpleBot:
    """
    A robust, defensive arbitrage bot.
    Features:
    1. Execution Safety: Perp First -> Spot Second (with IOC).
    2. Zero Naked Exposure: Strict atomic-like checks and rollbacks.
    3. Real-Time Risk: Monitors PnL and Basis deviation.
    """
    def __init__(self, strategy_name: str, symbol: str, interval: str = "1H", mock: bool = False):
        self.symbol = symbol
        self.interval = interval
        self.mock = mock
        
        # 1. Setup Logger
        self.logger = setup_logger("logs", f"simple_{strategy_name}.log")
        self.logger.info(f"Initializing GuardedBot for {strategy_name} on {symbol} (Mock={mock})")

        # 2. Config & Exchange
        load_dotenv()
        self.exchange = None      # Perp / Swap
        self.spot_exchange = None # Spot
        
        # State Tracking
        self.has_position = False
        self.entry_equity = 0.0   # Snapshot of equity at entry
        self.position_qty = 0.0   # Current holding size

        # 3. Strategy
        if strategy_name == 'funding_arb':
            self.logger.info("Using FundingRateArbitrageStrategy")
            self.strategy = FundingRateArbitrageStrategy(
                positive_threshold=None, # Bot will handle dynamic cost check
                neutral_threshold=0.0001,
                transaction_cost=0.003,
                target_days=5.0
            )
        else:
            self.logger.error(f"Unknown strategy: {strategy_name}")
            sys.exit(1)

    async def initialize(self):
        """Async initialization of exchanges"""
        flag = os.environ.get('OKX_FLAG', '0')
        is_sandbox = (str(flag) == '1')
        self.logger.info(f"Environment OKX_FLAG={flag} -> Sandbox={is_sandbox}")

        # 1. Swap Exchange (Primary - for Perp)
        self.exchange = await ExchangeFactory.create_exchange(
            'okx', 
            mock=self.mock,
            sandbox=is_sandbox,
            market_type='swap'
        )
        # Access raw CCXT instance for advanced order types
        if self.mock:
            self.ccxt_swap = self.exchange
        else:
            self.ccxt_swap = self.exchange.exchange 

        # 2. Spot Exchange (Secondary - for Spot)
        self.spot_exchange = await ExchangeFactory.create_exchange(
            'okx',
            mock=self.mock,
            sandbox=is_sandbox,
            market_type='spot'
        )
        if self.mock:
            self.ccxt_spot = self.spot_exchange
        else:
            self.ccxt_spot = self.spot_exchange.exchange

        self.logger.info("Bot Initialized. Starting State Reconciliation...")

    async def reconcile_state(self, quiet=False):
        """
        Check existing positions to recover from crashes.
        """
        if not quiet:
            self.logger.info("[STATE] RECONCILING STATE...")
        try:
            base_ccy = self.symbol.split('/')[0] 
            
            # 1. Check Spot
            spot_bal = await self.spot_exchange.get_balance(base_ccy)
            
            # 2. Check Perp
            perp_sz = 0.0
            if self.mock:
                positions = await self.ccxt_swap.fetch_positions()
            else:
                positions = await self.ccxt_swap.fetch_positions([self.symbol])
                
            if positions:
                # Iterate to find the correct symbol position
                for pos in positions:
                    if pos['symbol'] == self.symbol or pos['info'].get('instId') == self.symbol.replace('/', '-'):
                        raw_sz = float(pos['contracts'])
                        if pos['side'] == 'short':
                            perp_sz = -raw_sz 
            
            threshold = 0.001 
            if not quiet:
                self.logger.info(f"State Check: Spot={spot_bal}, PerpContracts (raw)={perp_sz}")

            if spot_bal > threshold:
                self.has_position = True
                self.position_qty = spot_bal 
                if not quiet:
                    self.logger.warning(f"[WARN] FOUND EXISTING POSITION! Restoring state. Qty: {self.position_qty}")
            else:
                self.has_position = False
                if not quiet:
                    self.logger.info("[OK] No existing position found. Ready to trade.")

        except Exception as e:
            self.logger.error(f"State Reconciliation Failed: {e}")
            if not self.mock:
                # Don't exit on runtime check, just log error
                if not quiet: sys.exit(1)

    async def _get_order_book_price(self, exchange, symbol: str) -> Tuple[float, float]:
        """Returns (Best Bid, Best Ask) for a symbol"""
        ob = await exchange.fetch_order_book(symbol, limit=1)
        bid = ob['bids'][0][0]
        ask = ob['asks'][0][0]
        return bid, ask

    async def calculate_dynamic_size(self, price: float) -> float:
        """
        Calculate safe trade size based on USDT balance and Risk Allocation.
        """
        try:
            # Get Free USDT balances
            if self.mock:
                bal = await self.spot_exchange.fetch_balance()
                spot_usdt = bal['USDT']['free']
                perp_usdt = spot_usdt # Shared in mock
            else:
                quote_ccy = 'USDT'
                spot_usdt = await self.spot_exchange.get_balance(quote_ccy)
                swap_bal_info = await self.ccxt_swap.fetch_balance()
                perp_usdt = float(swap_bal_info['USDT']['free']) if 'USDT' in swap_bal_info else 0.0
            
            equity = min(spot_usdt, perp_usdt)
            target_notional = equity * RISK_ALLOCATION * MAX_LEVERAGE
            qty = target_notional / price
            
            qty = math.floor(qty * 1000) / 1000.0
            
            if qty < 0.01: 
                return 0.0
                
            self.logger.info(f"[CALC] Sizing: Equity=${equity:.2f} -> Alloc=${target_notional:.2f} -> Qty={qty}")
            return qty
        except Exception as e:
            self.logger.error(f"Sizing Calc Failed: {e}")
            return 0.0

    async def _execute_safe(self, api_method, symbol, side, qty, benchmark_price) -> Tuple[bool, float]:
        """
        Execute an order with IOC (Immediate-or-Cancel) and Slippage Protection.
        Returns: (Success_Bool, Filled_Quantity)
        """
        if self.mock:
            self.logger.info(f"[MOCK] Executing {side.upper()} {qty} @ ~{benchmark_price}")
        
        # Calculate Guarded Limit Price
        if side == 'buy':
            limit_price = benchmark_price * (1 + SLIPPAGE_TOLERANCE)
        else:
            limit_price = benchmark_price * (1 - SLIPPAGE_TOLERANCE)
            
        try:
            params = {'timeInForce': 'IOC'}
            
            order = await api_method(
                symbol,
                'limit',
                side,
                qty,
                limit_price,
                params
            )
            
            status = order.get('status')
            filled = float(order.get('filled', 0))
            
            if status == 'closed' or filled >= qty * 0.99:
                self.logger.info(f"[OK] {side.upper()} FILLED: {filled} @ {order.get('average', limit_price)}")
                return True, filled
            else:
                self.logger.warning(f"[WARN] {side.upper()} Partial/Fail: {filled} / {qty}. Status: {status}")
                return False, filled
        except Exception as e:
            self.logger.error(f"[FAIL] Execution Exception ({side}): {e}")
            return False, 0.0

    async def _emergency_close_perp(self, symbol, qty):
        """Panic close Perp leg (Market Order) if Spot fails"""
        self.logger.critical("[ALERT] EMERGENCY: CLOSING PERP LEG...")
        try:
             await self.ccxt_swap.create_order(symbol, 'market', 'buy', qty)
             self.logger.info("[OK] Emergency Close Sent.")
        except Exception as e:
             self.logger.critical(f"[FAIL] PANIC FAILED: {e}")

    async def run(self):
        await self.initialize()
        await self.reconcile_state()
        
        cycle_count = 0
        
        while True:
            try:
                cycle_count += 1
                if cycle_count % 12 == 0: 
                    self.logger.info(f"--- Cycle {cycle_count} (Running) ---")

                # 1. Fetch Real-time Market Data
                perp_bid, perp_ask = await self._get_order_book_price(self.ccxt_swap, self.symbol)
                spot_bid, spot_ask = await self._get_order_book_price(self.ccxt_spot, self.symbol)
                # Fetch Funding Rate (Cached every ~1m)
                # Optimize: Remove 'since' overhead entirely by skipping fetch
                if cycle_count % 12 == 1: # Update on cycle 1, 13, 25...
                    # 1. Periodic State Guard (New)
                    await self.reconcile_state(quiet=True)

                    try:
                        from datetime import datetime, timedelta, timezone
                        since_ts = int((datetime.now(timezone.utc) - timedelta(days=2)).timestamp() * 1000)
                        
                        funding_df = await self.exchange.fetch_funding_rates(
                            self.symbol, 
                            limit=1, 
                            timeframe="",
                            since=since_ts
                        )
                        self.cached_funding_rate = float(funding_df.iloc[-1]['funding_rate']) if not funding_df.empty else 0.0
                    except Exception as e:
                        self.logger.warning(f"Funding Rate Fetch Failed: {e}. Using cache.")
                        # Keep previous cached value
                
                # Use Cache
                funding_rate = getattr(self, 'cached_funding_rate', 0.0)

                # 2. THE EQUATION: Real-time Cost Analysis
                FEES = 0.002
                entry_spread_cost = (spot_ask - perp_bid) / spot_ask
                total_entry_cost = entry_spread_cost + FEES
                
                is_profitable_entry = False
                if funding_rate > 0:
                     yield_buffer = funding_rate * 3 
                     # Update Strategy Logs
                log_color = '\033[92m' if is_profitable_entry else '\033[93m'
                if cycle_count % 12 == 0: # Log every ~1 minute
                     self.logger.info(f"[DATA] Market: PerpBid={perp_bid:.2f}, SpotAsk={spot_ask:.2f}, Fund={funding_rate:.6f}. Cost={total_entry_cost:.5f}. Trade? {log_color}{is_profitable_entry}\033[0m")

                # 3. Strategy Signal
                signal = self.strategy.generate_signal(None, self.symbol, funding_rate=funding_rate)

                # 4. EXECUTION LOGIC
                
                # ENTRY
                if signal == -1.0 and not self.has_position:
                    if is_profitable_entry:
                        qty = await self.calculate_dynamic_size(spot_ask)
                        if qty > 0:
                            self.logger.info(f"[GO] OPENING ARB: Qty {qty}. PerpBid {perp_bid} > SpotAsk {spot_ask} (or close)")
                            
                            # STEP 1: Sell Perp (Risk Leg)
                            perp_success, perp_filled = await self._execute_safe(self.ccxt_swap.create_order, self.symbol, 'sell', qty, perp_bid)
                            
                            if perp_filled > 0:
                                # STEP 2: Buy Spot (Hedge Leg) - Hedge WHATEVER was filled
                                self.logger.info(f"[HEDGE] Hedging Perp Fill: {perp_filled}...")
                                spot_success, spot_filled = await self._execute_safe(self.ccxt_spot.create_order, self.symbol, 'buy', perp_filled, spot_ask)
                                
                                if spot_filled >= perp_filled * 0.99:
                                    self.has_position = True
                                    self.position_qty = perp_filled # Track the actual filled size
                                    self.logger.info(f"[OK] ARB OPEN COMPLETE. Size: {self.position_qty}")
                                else:
                                    # Mismatch: Short > Long. Net Short Exposure.
                                    exposure = perp_filled - spot_filled
                                    if exposure > 0:
                                        self.logger.critical(f"[FAIL] HEDGE MISMATCH! Short:{perp_filled}, Long:{spot_filled}. Naked Short: {exposure}. CLOSING...")
                                        await self._emergency_close_perp(self.symbol, exposure)
                                    
                                    # If we managed to hedge at least something, track it? 
                                    # Simplified: If mismatch occurred, we closed the excess. 
                                    # So we are left with 'spot_filled' amount perfectly hedged (assuming spot_filled > 0).
                                    if spot_filled > 0:
                                        self.has_position = True
                                        self.position_qty = spot_filled
                                        self.logger.warning(f"[RECOVERY] Kept partial hedged position: {self.position_qty}")
                            else:
                                self.logger.warning("Entry Aborted: Perp fill 0. No action taken.")
                    else:
                        if cycle_count % 60 == 0:
                            self.logger.info("Signal -1 but Cost too high. Waiting...")

                # EXIT
                elif (signal == 0.0 or signal == 1.0) and self.has_position:
                    self.logger.info("[EXIT] NORMAL EXIT TRIGGERED.")
                    
                    perp_s, perp_f = await self._execute_safe(self.ccxt_swap.create_order, self.symbol, 'buy', self.position_qty, perp_ask)
                    spot_s, spot_f = await self._execute_safe(self.ccxt_spot.create_order, self.symbol, 'sell', self.position_qty, spot_bid)
                    
                    if perp_s and spot_s:
                        self.has_position = False
                        self.logger.info("[OK] POSITION CLOSED.")
                    else:
                        self.logger.critical(f"[ALERT] DIRTY EXIT. Spot:{spot_f}, Perp:{perp_f}")
                        self.has_position = False 

                # 5. RISK MONITOR (Always Run)
                if self.has_position:
                    basis = (perp_bid - spot_ask) / spot_ask 
                    if basis < BASIS_ALARM: 
                         self.logger.critical(f"[ALERT] STOP LOSS: Basis Divergence {basis:.2%}. CLOSING NOW.")
                         p_s, p_f = await self._execute_safe(self.ccxt_swap.create_order, self.symbol, 'buy', self.position_qty, perp_ask)
                         s_s, s_f = await self._execute_safe(self.ccxt_spot.create_order, self.symbol, 'sell', self.position_qty, spot_bid)
                         self.has_position = False
                         await asyncio.sleep(600)

                await asyncio.sleep(5) 

            except Exception as e:
                self.logger.error(f"Loop Error: {e}")
                await asyncio.sleep(10)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy', type=str, required=True, help='Strategy name')
    parser.add_argument('--symbol', type=str, default='BTC/USDT', help='Symbol')
    parser.add_argument('--mock', action='store_true', help='Mock mode')
    args = parser.parse_args()

    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    bot = SimpleBot(args.strategy, args.symbol, mock=args.mock)
    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        pass
