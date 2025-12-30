
import asyncio
import os
import yaml
from exchange.ccxt_exchange import CcxtExchange
# from config.logging_config import setup_trader_logger

# Mock Settings
TEST_SETTINGS = {
    'exchanges': {
        'binance': {
            'api_key': os.getenv('BINANCE_API_KEY', ''),
            'api_secret': os.getenv('BINANCE_API_SECRET', ''),
            'enabled': True
        },
        'okx': {
            'api_key': os.getenv('OKX_API_KEY', ''),
            'api_secret': os.getenv('OKX_API_SECRET', ''),
            'password': os.getenv('OKX_PASSWORD', ''),
            'enabled': True
        }
    }
}

async def test_exchange_class(exchange_id):
    print(f"\n--- Testing CcxtExchange: {exchange_id} ---")
    try:
        # Initialize
        exchange = CcxtExchange(exchange_id, TEST_SETTINGS['exchanges'][exchange_id], market_type='swap')
        await exchange.load()
        
        symbol = 'BTC/USDT' if exchange_id == 'binance' else 'BTC/USDT:USDT' 
        # Note: CcxtExchange.fetch_long_short_ratio handles symbol formatting internally usually, 
        # but let's pass the standard CCXT symbol.
        
        print(f"Fetching L/S Ratio for {symbol}...")
        try:
            df = await exchange.fetch_long_short_ratio(symbol, timeframe='1h', limit=5)
            if not df.empty:
                print(f"SUCCESS TopTrader L/S: {len(df)} records")
                print(df.head(1))
            else:
                print("FAILURE TopTrader L/S: DF is empty")
                
            # Global
            print(f"Fetching Global L/S for {symbol}...")
            global_df = await exchange.fetch_global_long_short_ratio(symbol, timeframe='1h', limit=5)
            if not global_df.empty:
                print(f"SUCCESS Global L/S: {len(global_df)} records")
                print(global_df.head(1))
            else:
                print("FAILURE Global L/S: DF is empty")

            # Taker
            print(f"Fetching Taker Vol Ratio for {symbol}...")
            taker_df = await exchange.fetch_taker_buy_sell_vol_ratio(symbol, timeframe='1h', limit=5)
            if not taker_df.empty:
                print(f"SUCCESS Taker Ratio: {len(taker_df)} records")
                print(taker_df.head(1))
            else:
                print("FAILURE Taker Ratio: DF is empty")

        except Exception as e:
            print(f"EXCEPTION in fetch: {e}")
            
        await exchange.close()
        
    except Exception as e:
        print(f"CRITICAL INIT ERROR: {e}")

async def main():
    await test_exchange_class('binance')
    await test_exchange_class('okx')

if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(main())
