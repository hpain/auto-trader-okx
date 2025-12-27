
import asyncio
import ccxt.async_support as ccxt
import time
from datetime import datetime, timedelta, timezone

async def probe_exchange(exchange_id, symbol):
    print(f"\n--- Probing {exchange_id} for {symbol} ---")
    try:
        exchange_class = getattr(ccxt, exchange_id)
        exchange = exchange_class()
        
        # Check capabilities
        print(f"Has fetchOpenInterestHistory: {exchange.has.get('fetchOpenInterestHistory')}")
        print(f"Has fetchOpenInterestOHLCV: {exchange.has.get('fetchOpenInterestOHLCV')}")
        
        # Try fetching from 1 year ago
        since_dt = datetime.now(timezone.utc) - timedelta(days=365)
        since_ms = int(since_dt.timestamp() * 1000)
        
        # Method 1: fetchOpenInterestHistory (Known to be limited)
        if exchange.has.get('fetchOpenInterestHistory'):
            try:
                print(f"Attempting fetchOpenInterestHistory since {since_dt}...")
                data = await exchange.fetch_open_interest_history(symbol, "1h", since=since_ms, limit=10)
                print(f"fetchOpenInterestHistory result count: {len(data)}")
                if data:
                    print(f"First record time: {datetime.fromtimestamp(data[0]['timestamp']/1000, timezone.utc)}")
            except Exception as e:
                print(f"fetchOpenInterestHistory failed: {e}")

        # Method 2: fetchOpenInterestOHLCV (Hopeful alternative)
        if exchange.has.get('fetchOpenInterestOHLCV'):
            try:
                print(f"Attempting fetchOpenInterestOHLCV since {since_dt}...")
                # Note: fetch_open_interest_ohlcv signature might vary, standard is symbol, timeframe, since, limit
                data = await exchange.fetch_open_interest_ohlcv(symbol, "1h", since=since_ms, limit=10)
                print(f"fetchOpenInterestOHLCV result count: {len(data)}")
                if data:
                    print(f"First record time: {datetime.fromtimestamp(data[0][0]/1000, timezone.utc)}") # OHLCV is list of lists
            except Exception as e:
                print(f"fetchOpenInterestOHLCV failed: {e}")
                
        await exchange.close()
        
    except Exception as e:
        print(f"General error probing {exchange_id}: {e}")

async def main():
    await probe_exchange('binance', 'BTC/USDT:USDT')
    await probe_exchange('okx', 'BTC/USDT:USDT')

if __name__ == "__main__":
    asyncio.run(main())
