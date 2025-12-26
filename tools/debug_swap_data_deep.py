import ccxt
import datetime
import time
import sys
import os

# Ensure we can import from parent directory if needed (standard boilerplate)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_fetch_swap_data(exchange_id, symbol, since_ms):
    print(f"\n--- Testing {exchange_id} [{symbol}] ---")
    
    try:
        exchange_class = getattr(ccxt, exchange_id)
        exchange = exchange_class({
            'enableRateLimit': True,
            'options': {'defaultType': 'swap'} 
        })
        
        # Check proxy env vars (User mentioned using TUN, so this might not be strictly needed but good for debug)
        # print(f"  Env HTTP_PROXY: {os.environ.get('HTTP_PROXY')}")
        # print(f"  Env HTTPS_PROXY: {os.environ.get('HTTPS_PROXY')}")

        print(f"  Loading markets...")
        exchange.load_markets()
        
        # Verify symbol existence and get ID
        if symbol not in exchange.markets:
            print(f"  ERROR: Symbol {symbol} not found in {exchange_id} markets.")
            market_keys = list(exchange.markets.keys())
            print(f"  Available symbols sample: {market_keys[:5]}")
            return

        print(f"  Target Symbol: {symbol}")
        print(f"  Since: {datetime.datetime.fromtimestamp(since_ms/1000, datetime.timezone.utc)} (Timestamp: {since_ms})")

        # 1. Test Funding Rate History
        print(f"\n  [1] Fetching Funding Rate History...")
        try:
            # CCXT unified method for funding rate history
            # Some exchanges use fetch_funding_rate_history, others might differ.
            if exchange.has['fetchFundingRateHistory']:
                funding_rates = exchange.fetch_funding_rate_history(symbol, since=since_ms, limit=1000)
                print(f"  SUCCESS: Fetched {len(funding_rates)} funding rate records.")
                if len(funding_rates) > 0:
                    first = funding_rates[0]
                    last = funding_rates[-1]
                    print(f"    First: {first['datetime']}")
                    print(f"    Last:  {last['datetime']}")
            else:
                print("  SKIP: Exchange does not support fetchFundingRateHistory")
        except Exception as e:
            print(f"  FAILURE (Funding Rate): {str(e)}")

        # 2. Test Open Interest History
        print(f"\n  [2] Fetching Open Interest History...")
        try:
            if exchange.has['fetchOpenInterestHistory']:
                # Note: Binance often has strict limits on 'limit' (e.g. 30, 50, 500) and time range
                # We try a standard call first
                ois = exchange.fetch_open_interest_history(symbol, "1h", since=since_ms, limit=500)
                print(f"  SUCCESS: Fetched {len(ois)} open interest records.")
                if len(ois) > 0:
                    first = ois[0]
                    last = ois[-1]
                    print(f"    First: {first['datetime']}")
                    print(f"    Last:  {last['datetime']}")
            else:
                print("  SKIP: Exchange does not support fetchOpenInterestHistory")
        except Exception as e:
            print(f"  FAILURE (Open Interest): {str(e)}")

    except Exception as e:
        print(f"  CRITICAL ERROR initializing {exchange_id}: {str(e)}")
    finally:
        if 'exchange' in locals():
            exchange.close()

def main():
    # 1 Year ago
    now = datetime.datetime.now(datetime.timezone.utc)
    one_year_ago = now - datetime.timedelta(days=365)
    since_ms = int(one_year_ago.timestamp() * 1000)

    # Test Binance
    # CCXT symbol for Binance USDT swap is usually BTC/USDT:USDT
    test_fetch_swap_data('binance', 'BTC/USDT:USDT', since_ms)

    # Test OKX
    # CCXT symbol for OKX Swap is usually BTC/USDT:USDT or BTC-USDT-SWAP depending on mapping. 
    # Standard CCXT unified is BTC/USDT:USDT for linear swap.
    test_fetch_swap_data('okx', 'BTC/USDT:USDT', since_ms)

if __name__ == "__main__":
    main()
