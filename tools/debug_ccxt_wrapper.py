
import asyncio
import os
import sys
import logging
from datetime import datetime, timedelta

# Add project root to sys.path
sys.path.append(os.getcwd())

from exchange.ccxt_exchange import CcxtExchange

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_wrapper_logic():
    # Rely on system environment or TUN mode for connectivity
    # Do not force proxy settings here
    
    # Initialize exchanges
    print("Initializing Exchanges...")
    
    # Binance Swap
    binance_swap = CcxtExchange('binance', market_type='swap', sandbox=False)
    # OKX Swap
    okx_swap = CcxtExchange('okx', market_type='swap', sandbox=False)

    await binance_swap.load()
    await okx_swap.load()

    symbol = 'BTC-USDT'
    
    # 1 Year Ago
    since_1y = int((datetime.utcnow() - timedelta(days=365)).timestamp() * 1000)
    # 30 Days Ago
    since_30d = int((datetime.utcnow() - timedelta(days=30)).timestamp() * 1000)

    print(f"\n--- Testing Funding Rate (1 Year Ago: {since_1y}) ---")
    
    # Test Binance Funding
    print("\n[BINANCE] Fetching Funding Rate (1 Year)...")
    fr_binance = await binance_swap.fetch_funding_rates(symbol, '8h', since=since_1y, limit=100)
    print(f"[BINANCE] Result: {len(fr_binance)} rows")
    if not fr_binance.empty:
        print(fr_binance.head(1))
        print(fr_binance.tail(1))

    # Test OKX Funding
    print("\n[OKX] Fetching Funding Rate (1 Year)...")
    fr_okx = await okx_swap.fetch_funding_rates(symbol, '8h', since=since_1y, limit=100)
    print(f"[OKX] Result: {len(fr_okx)} rows")
    if not fr_okx.empty:
        print(fr_okx.head(1))
        print(fr_okx.tail(1))

    print(f"\n--- Testing Open Interest (30 DaysAgo: {since_30d}) ---")
    # Note: Binance OI history is tricky, often fails with long lookbacks
    
    # Test Binance OI
    print("\n[BINANCE] Fetching Open Interest (30 Days)...")
    oi_binance = await binance_swap.fetch_open_interest(symbol, '1h', since=since_30d, limit=100)
    print(f"[BINANCE] Result: {len(oi_binance)} rows")
    if not oi_binance.empty:
        print(oi_binance.head(1))
        print(oi_binance.tail(1))

    # Test OKX OI
    print("\n[OKX] Fetching Open Interest (30 Days)...")
    oi_okx = await okx_swap.fetch_open_interest(symbol, '1h', since=since_30d, limit=100)
    print(f"[OKX] Result: {len(oi_okx)} rows")
    if not oi_okx.empty:
        print(oi_okx.head(1))
        print(oi_okx.tail(1))

    await binance_swap.close()
    await okx_swap.close()

if __name__ == "__main__":
    # Fix for Windows loop policy
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(test_wrapper_logic())
