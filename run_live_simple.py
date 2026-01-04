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
from utils.logger import setup_logger

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
        self.logger = setup_logger(f"SimpleBot_{strategy_name}", f"logs/simple_{strategy_name}.log")
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
            self.strategy = FundingRateArbitrageStrategy(
                positive_threshold=0.0001, # 0.01%
                negative_threshold=-0.0001
            )
        else:
            self.logger.error(f"Unknown strategy: {strategy_name}")
            sys.exit(1)
            
        self.execution = None

    async def initialize(self):
        """Async initialization"""
        self.exchange = await ExchangeFactory.create_exchange(
            'okx', # Use direct OKX for simplicity? Or aggregated. Let's stick to factory default.
            mock=self.mock
        )
        self.execution = ExecutionHandler(self.exchange)
        self.logger.info("Bot Initialized. Starting Loop.")

    async def run(self):
        await self.initialize()
        
        cycle_count = 0
        while True:
            try:
                cycle_count += 1
                self.logger.info(f"--- Cycle {cycle_count} ---")
                
                # 1. Fetch Data
                # Funding Arb only needs Funding Rate & Price
                funding_info = await self.exchange.client.fetch_funding_rate(self.symbol)
                ticker = await self.exchange.client.fetch_ticker(self.symbol)
                
                current_rate = funding_info.get('fundingRate', 0.0)
                current_price = ticker.get('last', 0.0)
                
                self.logger.info(f"[{self.symbol}] Price: {current_price:.2f}, Funding: {current_rate:.6f}")

                # 2. Generate Signal
                signal = self.strategy.generate_signal(None, self.symbol, funding_rate=current_rate)
                
                self.logger.info(f"Signal: {signal} (State: {self.strategy.current_state})")

                # 3. Execute
                if signal != 0:
                    if self.mock:
                        self.logger.info(f"[MOCK] Would execute signal {signal} on {self.symbol}")
                    else:
                        # Real Execution Logic
                        # Warning: This is "Naked" signal execution. 
                        # Ideally we check current position first to avoid double entries.
                        # For Funding Arb, generate_signal manages state transitions (Neutral -> Arb -> Neutral)
                        # So we trust the state change.
                        
                        # Generate Order Dict
                        side = 'sell' if signal < 0 else 'buy'
                        # Size? We need a fixed size or config size. For now hardcode or use small amount.
                        # P1-1 Goal: Prove it works.
                         
                        self.logger.info(f"Executing {side.upper()} order for Arb...")
                        # await self.execution.execute_order(...) 
                        # Placeholder: We need to implement dual-leg execution for Arb here.
                        # But standard ExecutionHandler is single-leg.
                        # For P1-1 MVP, we might just log "ACTION REQUIRED" or execute simple Perp leg.
                        pass

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
