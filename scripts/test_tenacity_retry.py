
import asyncio
import logging
import time
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from exchange.ccxt_exchange import CcxtExchange

# Configure logging to see the retry warnings
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# Set tenacity logger to WARNING to see the "Retrying..." logs if configured, 
# but my code logs warnings in before_sleep.

async def test_retry():
    print("--- Starting Tenacity Retry Test ---")
    print("Target: Binance Testnet (Known to be DOWN/502)")
    
    # Initialize Exchange (Binance Testnet)
    exchange = CcxtExchange('binance', sandbox=True)
    
    start_time = time.time()
    
    try:
        print("\n[1] Calling get_current_price('BTC/USDT')...")
        print("    Expectation: It should hang for a few seconds (retrying) and then return 0.0.")
        
        price = await exchange.get_current_price('BTC/USDT')
        
        duration = time.time() - start_time
        print(f"\n[Result] Call finished in {duration:.2f} seconds.")
        print(f"[Result] Type: {type(price)}")
        print(f"[Result] Value: {price}")
        
        if price == 0.0 and duration > 2:
            print("\n✅ TEST PASSED!")
            print("   - Graceful degradation worked (returned 0.0).")
            print("   - Retry delay observed (wait > 2s).")
        elif price > 0:
            print("\n⚠️ TEST INCONCLUSIVE: Binance Testnet seems to be UP? It returned a price.")
        else:
            print("\n❌ TEST FAILED: Did not return expected 0.0 or was too fast.")
            
    except Exception as e:
        print(f"\n❌ TEST FAILED: Exception was raised! {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        await exchange.close()

if __name__ == "__main__":
    asyncio.run(test_retry())
