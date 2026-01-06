import asyncio
import argparse
import sys
import os
import logging
from typing import Dict
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from exchange.factory import ExchangeFactory
from strategies.funding_arb import FundingRateArbitrageStrategy
from trader.execution_handler import ExecutionHandler
from utils.logger import setup_script_logger as setup_logger

class SimpleBot:
    """
    A minimal, robust bot runner for a single strategy.
    Designed for stability and determinism (P1-2).
    """
    def __init__(self, strategy_name: str, symbol: str, interval: str = "1H", mock: bool = False):
        self.symbol = symbol
        self.interval = interval
        self.mock = mock
        
        # 1. Setup Logger (Distinct from main bot)
        # Fix: correctly pass dir and filename separately
        self.logger = setup_logger("logs", f"simple_{strategy_name}.log")
        self.logger.info(f"Initializing SimpleBot for {strategy_name} on {symbol} (Mock={mock})")

        # 2. Config & Exchange
        load_dotenv()
        try:
            # We construct basic credentials manually or load from env/settings
            # For simplicity, we rely on standard ExchangeFactory which loads from settings/env
            self.logger.info("Connecting to Exchange...")
            # We assume settings.yaml is valid or env vars are set.
            # Using 'aggregated' to keep compatible with factory, though 'okx' direct would be simpler.
            self.exchange = None 
        except Exception as e:
            self.logger.error(f"Failed to setup config: {e}")
            sys.exit(1)

        # 3. Strategy
        if strategy_name == 'funding_arb':
            # Use Dynamic Thresholds based on Cost Recovery
            # Cost = 0.3% (Fee+Slip), Target = 5 Days (150 payouts)
            # This triggers if Rate > ~0.02%
            self.logger.info("Using DYNAMIC PROFIT CALCULATION (Cost: 0.3%, Target: 5 Days)")
            self.strategy = FundingRateArbitrageStrategy(
                positive_threshold=None, # Auto-calculate
                negative_threshold=None,
                neutral_threshold=0.0001,
                transaction_cost=0.003,
                target_days=5.0
            )
        else:
            self.logger.error(f"Unknown strategy: {strategy_name}")
            sys.exit(1)
            
        self.execution = None

    async def initialize(self):
        """Async initialization"""
        # Determine Sandbox Mode from Env
        # OKX Convention: 0 = Live, 1 = Sandbox (Simulated Trading)
        # Default to Live (0) if not set, to match Docker hardcoding for Arb
        flag = os.environ.get('OKX_FLAG', '0')
        is_sandbox = (str(flag) == '1')
        self.logger.info(f"Environment OKX_FLAG={flag} -> Sandbox={is_sandbox}")

        # 1. Swap Exchange (Primary - for Funding Rates & Perp Orders)
        self.exchange = await ExchangeFactory.create_exchange(
            'okx', 
            mock=self.mock,
            sandbox=is_sandbox,
            market_type='swap'
        )
        self.execution = ExecutionHandler(self.exchange)

        # 2. Spot Exchange (Secondary - for Hedging)
        self.spot_exchange = await ExchangeFactory.create_exchange(
            'okx',
            mock=self.mock,
            sandbox=is_sandbox,
            market_type='spot'
        )
        self.spot_execution = ExecutionHandler(self.spot_exchange)

        self.logger.info("Bot Initialized (Dual-Leg Mode). Starting Loop.")

    async def reconcile_state(self):
        """
        Safety Check: Query exchange to see if we ALREADY have a position.
        This handles crash recovery (Zombie State).
        """
        self.logger.info("♻️ RECONCILING STATE with Exchange...")
        try:
            # 1. Check Spot Holdings (Leg 1)
            # Assumes ETH/USDT -> base currency is ETH
            base_ccy = self.symbol.split('/')[0] 
            spot_bal = await self.spot_exchange.get_balance(base_ccy)
            
            # 2. Check Perp Position (Leg 2)
            # Use raw CCXT method as wrapper might not have specific fetch_position
            # OKX usually returns a list
            positions = await self.exchange.exchange.fetch_positions([self.symbol])
            perp_sz = 0.0
            if positions:
                # OKX returns 'contracts' or 'size' depending on mode, but 'contracts' is usually safe for swap
                # We care about direction. Short is negative? 
                # CCXT standard: 'side': 'short', 'contracts': 10
                pos = positions[0]
                if pos['side'] == 'short':
                    perp_sz = float(pos['contracts']) * float(pos['contractSize']) # Approximate logic, verify for OKX
                    # Simpler: 'info'['pos'] usually contains signed size strings on OKX
                    # Or verify 'side'
                    perp_sz = -abs(float(pos['contracts'])) # Treat short as negative
                elif pos['side'] == 'long':
                     perp_sz = abs(float(pos['contracts']))
            
            self.logger.info(f"🧐 State Check: Spot {base_ccy}={spot_bal:.4f}, Perp Pos={perp_sz:.4f}")

            # 3. Determine Logic
            # Threshold: e.g. 0.005 ETH to account for dust
            threshold = 0.005 
            
            # If we hold Spot AND Short Perp => We are likely in an Arb
            if spot_bal > threshold and perp_sz < -threshold:
                self.logger.warning(f"⚠️ FOUND EXISTING ARB POSITION! Restoring state to OPEN.")
                return True
            
            # Partial states risks
            if spot_bal > threshold and perp_sz == 0:
                self.logger.critical(f"🚨 DANGER: Unhedged Spot Position detected! ({spot_bal} {base_ccy}). Please check manually.")
                # Optional: self.spot_execution.execute_order(..., 'sell', ...) ? Too risky to auto-close.
            
            if spot_bal < threshold and perp_sz < -threshold:
                 self.logger.critical(f"🚨 DANGER: Naked Short detected! ({perp_sz} contracts). Please check manually.")

            return False

        except Exception as e:
            self.logger.error(f"State Reconciliation Failed: {e}")
            return False

    async def run(self):
        await self.initialize()
        
        # Recover state from actual exchange data
        has_position = await self.reconcile_state() 
        
        cycle_count = 0
        trade_qty = 0.02     # Updated for $139 capital (approx $62 Spot + $62 Perp)

        while True:
            try:
                cycle_count += 1
                self.logger.info(f"--- Cycle {cycle_count} ---")
                
                # 1. Fetch Data
                funding_df = await self.exchange.fetch_funding_rates(self.symbol, limit=1, timeframe="")
                current_price = await self.exchange.get_current_price(self.symbol)
                
                # Fetch Balance (Quote Currency for Funding Arb)
                quote_ccy = self.symbol.split('/')[1]
                balance = await self.spot_exchange.get_balance(quote_ccy)

                current_rate = 0.0
                if not funding_df.empty:
                    current_rate = float(funding_df.iloc[-1]['funding_rate'])
                
                # Colors
                C_GREEN = '\033[92m'
                C_YELLOW = '\033[93m'
                C_CYAN = '\033[96m'
                C_RESET = '\033[0m'
                
                self.logger.info(f"[{self.symbol}] Price: {current_price:.2f}, {C_GREEN}Funding: {current_rate:.6f}{C_RESET}, {C_YELLOW}Balance: {balance:.2f} {quote_ccy}{C_RESET}")

                # 2. Generate Signal
                # Strategy tracks its own state, but returns code:
                # -1.0: Rate is High (Enter Short Arb)
                # 0.0: Rate is Normal (Exit/Neutral)
                # 1.0: Rate is Low (Enter Long Arb - Rare)
                signal = self.strategy.generate_signal(None, self.symbol, funding_rate=current_rate)
                
                self.logger.info(f"Signal: {signal} (StratState: {self.strategy.current_state} | BotPos: {has_position})")

                # 3. Execute (Dual-Leg Hedging)
                if self.mock:
                     if signal != 0:
                        self.logger.info(f"[MOCK] Signal {signal}. Position: {has_position}")
                else:
                    # ENTRY Logic (Positive Arb: Short Perp + Buy Spot)
                    if signal == -1.0 and not has_position:
                        self.logger.info(f"⚡ OPPORTUNITY! Opening Delta-Neutral Arb (Size: {trade_qty} ETH)...")
                        
                        # Leg 1: Buy Spot (Hedge)
                        spot_res = await self.spot_execution.execute_order(self.symbol, 'buy', trade_qty, type='market')
                        if spot_res:
                            self.logger.info(f"✅ Leg 1: Spot BUY Executed.")
                            
                            # Leg 2: Sell Perp (Income)
                            perp_res = await self.execution.execute_order(self.symbol, 'sell', trade_qty, type='market')
                            if perp_res:
                                self.logger.info(f"✅ Leg 2: Perp SELL Executed.")
                                has_position = True
                                self.logger.info(f"🚀 ARBITRAGE POSITION OPENED SUCCESSFULLY.")
                            else:
                                self.logger.critical(f"❌ CRITICAL: Perp LEG FAILED. You have unhedged Spot position!")
                                # TODO: Emergency Close Spot?
                        else:
                            self.logger.error("❌ Spot Leg Failed. Aborting Arb entry.")

                    # EXIT Logic (Neutral: Close Both)
                    elif signal == 0.0 and has_position:
                        self.logger.info(f"📉 NORMALIZATION. Closing Arb Position...")
                        
                        # Close Leg 1: Sell Spot
                        spot_res = await self.spot_execution.execute_order(self.symbol, 'sell', trade_qty, type='market')
                        
                        # Close Leg 2: Buy Perp
                        perp_res = await self.execution.execute_order(self.symbol, 'buy', trade_qty, type='market')
                        
                        if spot_res and perp_res:
                            self.logger.info(f"✅ Position Closed Successfully.")
                            has_position = False
                        else:
                             self.logger.error(f"⚠️ Close Error. Spot: {bool(spot_res)}, Perp: {bool(perp_res)}")

                # 4. Sleep
                # Check every 5 minutes
                await asyncio.sleep(300)

            except Exception as e:
                self.logger.error(f"Error in loop: {e}")
                await asyncio.sleep(60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy', type=str, required=True, help='Strategy name (e.g., funding_arb)')
    parser.add_argument('--symbol', type=str, default='BTC/USDT', help='Symbol to trade')
    parser.add_argument('--mock', action='store_true', help='Run in mock mode')
    args = parser.parse_args()

    # Windows Fix
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    bot = SimpleBot(args.strategy, args.symbol, mock=args.mock)
    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        pass
