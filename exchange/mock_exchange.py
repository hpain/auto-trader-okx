import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

class MockExchange:
    def __init__(self, api_key=None, api_secret=None, passphrase=None, sandbox=True):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing MockExchange...")
        self.sandbox = sandbox
        self.balance = {'USDT': 10000.0, 'BTC': 0.1}
        self.current_price = 50000.0
        self.orders = {}

    async def load(self):
        self.logger.info("MockExchange loaded.")

    async def fetch_candles(self, symbol, timeframe='1H', limit=100):
        self.logger.info(f"Mock fetching candles for {symbol} {timeframe} limit={limit}")
        # Generate random OHLCV data
        end_time = datetime.utcnow()
        if timeframe == '1H':
            delta = timedelta(hours=1)
        elif timeframe == '1m':
            delta = timedelta(minutes=1)
        else:
            delta = timedelta(hours=1) # Default

        timestamps = [end_time - i * delta for i in range(limit)]
        timestamps.reverse()

        data = []
        price = self.current_price
        for ts in timestamps:
            open_p = price
            close_p = price * (1 + np.random.normal(0, 0.01))
            high_p = max(open_p, close_p) * (1 + abs(np.random.normal(0, 0.005)))
            low_p = min(open_p, close_p) * (1 - abs(np.random.normal(0, 0.005)))
            vol = abs(np.random.normal(100, 20))
            data.append([ts, open_p, high_p, low_p, close_p, vol])
            price = close_p
        
        self.current_price = price # Update current price
        
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'vol'])
        df.set_index('timestamp', inplace=True)
        return df

    async def get_balance(self, currency):
        self.logger.info(f"Mock get_balance for {currency}")
        return self.balance.get(currency, 0.0)

    async def get_current_price(self, symbol):
        self.logger.info(f"Mock get_current_price for {symbol}")
        # Simulate slight price movement
        self.current_price *= (1 + np.random.normal(0, 0.001))
        return self.current_price

    async def create_order(self, symbol, order_type, side, amount, price=None):
        self.logger.info(f"Mock create_order: {side} {amount} {symbol} @ {price}")
        order_id = f"mock_order_{datetime.utcnow().timestamp()}"
        
        # Update mock balance
        base, quote = symbol.split('-')
        if side == 'buy':
            cost = amount * self.current_price
            if self.balance.get(quote, 0) >= cost:
                self.balance[quote] -= cost
                self.balance[base] = self.balance.get(base, 0) + amount
                status = 'filled'
            else:
                self.logger.warning("Mock order failed: Insufficient funds")
                status = 'rejected'
        elif side == 'sell':
            if self.balance.get(base, 0) >= amount:
                self.balance[base] -= amount
                revenue = amount * self.current_price
                self.balance[quote] = self.balance.get(quote, 0) + revenue
                status = 'filled'
            else:
                self.logger.warning("Mock order failed: Insufficient funds")
                status = 'rejected'
        
        return {
            'id': order_id,
            'symbol': symbol,
            'type': order_type,
            'side': side,
            'amount': amount,
            'price': self.current_price,
            'status': status
        }

    async def close(self):
        self.logger.info("MockExchange closed.")
