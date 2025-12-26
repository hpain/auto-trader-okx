import asyncio
import ccxt.pro as ccxt  # Use ccxt.pro for consistency if project uses it, or standard ccxt
import datetime
import time

async def test_funding_oi():
    # Configuration
    symbol = 'BTC/USDT:USDT'  # Standard CCXT unified symbol for linear swap
    # For Binance, it might be BTC/USDT:USDT. For OKX, same.
    # Let's try to be specific or rely on CCXT defaults.
    
    proxies = {
        'http': 'http://127.0.0.1:7890',
        'https': 'http://127.0.0.1:7890'
    }
    
    # Initialize exchanges
    exchanges = {}
    try:
        exchanges['binance'] = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'future'},
            'proxies': proxies,
            'httpsProxy': 'http://127.0.0.1:7890', # Explicitly set for testing
        })
        exchanges['okx'] = ccxt.okx({
            'enableRateLimit': True,
            'options': {'defaultType': 'swap'},
            'proxies': proxies,
             'httpsProxy': 'http://127.0.0.1:7890',
        })
    except Exception as e:
        print(f"Initialization error: {e}")
        return

    # Time setup
    now = int(time.time() * 1000)
    one_year_ago = now - (365 * 24 * 60 * 60 * 1000)
    
    print(f"Current Time (ms): {now}")
    print(f"1 Year Ago (ms): {one_year_ago}")
    print(f"Test Symbol: {symbol}")
    print("-" * 50)

    for name, exchange in exchanges.items():
        print(f"\n[{name.upper()}] Testing...")
        
        # 1. Fetch Funding Rate
        print(f"[{name.upper()}] Fetching Funding Rate since {one_year_ago}...")
        try:
            # Note: CCXT fetch_funding_rate_history might behave differently per exchange
            funding_rates = await exchange.fetch_funding_rate_history(symbol, since=one_year_ago, limit=100)
            print(f"SUCCESS: Fetched {len(funding_rates)} funding rates.")
            if funding_rates:
                print(f"First: {funding_rates[0]['datetime']}, Last: {funding_rates[-1]['datetime']}")
        except Exception as e:
            print(f"FAILURE (Funding Rate): {e}")

        # 2. Fetch Open Interest
        print(f"[{name.upper()}] Fetching Open Interest since {one_year_ago}...")
        try:
            # CCXT method for OI history
            # Not all exchanges support fetching history via a unified method efficiently
            # We check for fetch_open_interest_history
            if exchange.has.get('fetchOpenInterestHistory'):
                ois = await exchange.fetch_open_interest_history(symbol, "1h", since=one_year_ago, limit=100)
                print(f"SUCCESS: Fetched {len(ois)} open interest records.")
                if ois:
                    print(f"First: {ois[0]['datetime']}, Last: {ois[-1]['datetime']}")
            else:
                print("SKIPPED: fetchOpenInterestHistory not supported/enabled in CCXT metadata.")
                
        except Exception as e:
            print(f"FAILURE (Open Interest): {e}")

        await exchange.close()

if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(test_funding_oi())