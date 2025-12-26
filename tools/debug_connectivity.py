import asyncio
import ccxt.async_support as ccxt
import os
import sys
import time
from datetime import datetime, timedelta

print(f"--- DIAGNOSTIC START (DUAL EXCHANGE TUN MODE) ---")
print(f"Time: {time.ctime()}")
print(f"Platform: {sys.platform}")

# -------------------------------------------------------------------------
# 关键修复：即使是 TUN 模式，Windows 依然建议使用 SelectorEventLoop 以避免 SSL 报错
# -------------------------------------------------------------------------
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    print("APPLIED FIX: WindowsSelectorEventLoopPolicy")

async def fetch_exchange_data(exchange_id):
    print(f"\n[{exchange_id.upper()}] Starting task...")
    
    # 动态创建交易所对象
    exchange_class = getattr(ccxt, exchange_id)
    # TUN 模式下，通常不需要显式设置 proxy，让流量自然走网卡
    # 除非你的 TUN 模式配置很特殊。这里我们先尝试“裸连”（依赖 TUN）
    exchange = exchange_class({
        'timeout': 30000,
        'enableRateLimit': True,
    })
    
    symbol = 'BTC/USDT'
    timeframe = '1h'
    since_dt = datetime.utcnow() - timedelta(days=180) # 半年
    since = int(since_dt.timestamp() * 1000)
    
    all_candles = []
    start_time = time.time()
    
    try:
        print(f"    [{exchange_id.upper()}] Fetching from {since_dt}...")
        while True:
            # 限制单次条数，Binance/OKX 限制不同，100 是安全值
            candles = await exchange.fetch_ohlcv(symbol, timeframe, since, limit=100)
            
            if not candles:
                print(f"    [{exchange_id.upper()}] No more data.")
                break
            
            all_candles.extend(candles)
            last_time = candles[-1][0]
            since = last_time + 1
            
            last_dt = datetime.fromtimestamp(last_time/1000)
            
            # 简单的进度条
            if len(all_candles) % 500 == 0:
                 print(f"    [{exchange_id.upper()}] Progress: {len(all_candles)} candles (Last: {last_dt})")

            # 只要能获取到最近 1 天的数据，就算成功
            if (datetime.utcnow() - last_dt).days < 1:
                print(f"    [{exchange_id.upper()}] Reached current date!")
                break
                
            # 测试阈值：每个交易所获取 500 条即可证明稳定性
            if len(all_candles) >= 500:
                print(f"    [{exchange_id.upper()}] TEST PASSED: Fetched > 500 candles.")
                break

    except Exception as e:
        print(f"    [{exchange_id.upper()}] CRITICAL ERROR: {e}")
    finally:
        await exchange.close()
        duration = time.time() - start_time
        print(f"[{exchange_id.upper()}] Finished. Total: {len(all_candles)} candles in {duration:.2f}s")

async def main():
    # 并发执行两个任务
    await asyncio.gather(
        fetch_exchange_data('binance'),
        fetch_exchange_data('okx')
    )

if __name__ == "__main__":
    asyncio.run(main())