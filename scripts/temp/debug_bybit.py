import ccxt
import sys
import traceback

print("--- Script start ---")

def test_bybit():
    print("--- Testing BYBIT ---")
    exchange = None
    try:
        print("Initializing ccxt.bybit...")
        exchange = ccxt.bybit({
            'options': {
                'defaultType': 'swap',
            },
            'proxies': {
                'http': 'http://localhost:10808',
                'https': 'http://localhost:10808',
            }
        })
        exchange.verbose = True
        
        print("Loading specific 'linear' markets to avoid spot API issues...")
        # Bybit's unified API requires a 'category' parameter for the instruments-info endpoint.
        # CCXT's load_markets can accept params that are passed down to the request.
        # We explicitly load only 'linear' markets to bypass the failing 'spot' market request.
        exchange.load_markets(params={'category': 'linear'})
        print("Markets loaded successfully.")

        symbol = 'BTC/USDT:USDT'

        if symbol in exchange.markets:
            print(f"Market for {symbol} found.")
            print(f"Attempting to fetch open interest history for {symbol}...")
            
            # For Bybit v5, fetch_open_interest_history also requires the 'category'
            params = {'category': 'linear'}
            open_interest = exchange.fetch_open_interest_history(symbol, limit=10, params=params)
            
            if open_interest:
                print(f"Successfully fetched {len(open_interest)} records from BYBIT.")
                print("First record preview:")
                print(open_interest[0])
            else:
                print("Fetched no open interest data from BYBIT. The request was successful but returned no data.")
        else:
            print(f" Symbol {symbol} not found in the loaded Bybit markets.")
            print("Loaded markets are:")
            print(list(exchange.markets.keys())[:20]) # Print first 20 markets as a sample

    except Exception as e:
        print(f" An error occurred while testing BYBIT: {e}", file=sys.stderr)
        print("Traceback:", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
    finally:
        # No close method for synchronous exchange
        pass

if __name__ == "__main__":
    print("--- Running main ---")
    test_bybit()
    print("--- Script end ---")