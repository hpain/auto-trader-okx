import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

class MockExchange:
    def __init__(self, exchange_id='mock', market_type='spot', api_key=None, api_secret=None, passphrase=None, sandbox=True, **kwargs):
        self.logger = logging.getLogger(__name__)
        self.exchange_id = exchange_id 
        self.market_type = market_type
        self.logger.info(f"Initializing MockExchange ({self.exchange_id}, {self.market_type})...")
        self.sandbox = sandbox
        self.balance = {'USDT': 10000.0, 'BTC': 0.1, 'ETH': 1.0} # Reverted to defaultce = 50000.0
        self.current_price = 50000.0
        self.positions = [] # Track positions
        
    async def load(self):
        self.logger.info("MockExchange loaded.")

    async def fetch_candles(self, symbol, timeframe='1H', since=None, limit=100, **kwargs):
        # ... existing ...
        self.logger.info(f"Mock fetching candles for {symbol}")
        return pd.DataFrame() # minimal return to avoid crash if called, though Arb Bot doesn't call this directly in loop.

    async def get_balance(self, currency):
        return self.balance.get(currency, 0.0)

    async def fetch_balance(self):
        """CCXT compliant fetch_balance"""
        return {
            'USDT': {'free': self.balance.get('USDT', 0), 'used': 0, 'total': self.balance.get('USDT', 0)},
            'free': self.balance,
            'total': self.balance
        }

    async def get_current_price(self, symbol):
        self.current_price *= (1 + np.random.normal(0, 0.001))
        return self.current_price

    async def fetch_order_book(self, symbol, limit=10):
        """Mock Order Book with Spread"""
        price = await self.get_current_price(symbol)
        spread = 0.0002 # 0.02% spread
        bid = price * (1 - spread/2)
        ask = price * (1 + spread/2)
        # Structure: {'bids': [[price, qty], ...], 'asks': ...}
        return {
            'bids': [[bid, 1.0]],
            'asks': [[ask, 1.0]]
        }

    async def fetch_funding_rates(self, symbol, limit=1, timeframe=""):
        """Mock Funding Rate. Returns High Rate to trigger Arb."""
        return pd.DataFrame([{'funding_rate': 0.0006, 'timestamp': datetime.utcnow()}]) # Default low rate

    async def fetch_positions(self, symbols=None):
        return self.positions

    async def create_order(self, symbol, order_type, side, amount, price=None, params={}):
        """Supported params check"""
        self.logger.info(f"Mock create_order: {side} {amount} {symbol} @ {price} (Params: {params})")
        
        status = 'closed'
        filled = amount
        average = price if price else self.current_price
        
        # Update Balance Logic (Simple)
        # ... (same as before) ...
        
        # Update Position Logic (Simple)
        found = False
        for pos in self.positions:
            if pos['symbol'] == symbol:
                # Update existing size (Simple add/sub)
                found = True
                curr_size = float(pos['contracts'])
                if side == 'buy': curr_size += amount
                else: curr_size -= amount
                pos['contracts'] = curr_size
        
        if not found:
            self.positions.append({
                'symbol': symbol,
                'side': 'short' if side == 'sell' else 'long',
                'contracts': amount if side == 'buy' else -amount, # Signed?
                'info': {'instId': symbol}
            })
            # Fix: Ensure logic matches Arb Bot expectations
            if side == 'sell': # Short
                 self.positions[-1]['contracts'] = amount 
                 self.positions[-1]['side'] = 'short'
        
        return {
            'id': 'mock_id',
            'status': status,
            'filled': filled,
            'average': average
        }

    async def close(self):
        pass
