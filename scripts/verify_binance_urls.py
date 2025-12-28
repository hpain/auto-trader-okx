import requests

def check_url(url):
    try:
        r = requests.head(url)
        print(f"{r.status_code} | {url}")
    except Exception as e:
        print(f"Error: {e}")

base = "https://data.binance.vision/data/futures/um/monthly"
symbol = "BTCUSDT"
date = "2024-01"

print("Checking URL Patterns...")
# 1. Funding Rate
url_fr = f"{base}/fundingRate/{symbol}/{symbol}-fundingRate-{date}.zip"
check_url(url_fr)

# 2. Metrics / Open Interest Probes (DAILY)
# User confirmed: data/futures/cm/daily/metrics/BTCUSD_230331/
# We need BTCUSDT (UM Perp) or BTCUSD_PERP (CM Perp)

base_daily_um = "https://data.binance.vision/data/futures/um/daily"
base_daily_cm = "https://data.binance.vision/data/futures/cm/daily"
date_daily = "2024-01-01"

candidates = [
    # Target: USDT Perp
    f"{base_daily_um}/metrics/BTCUSDT/BTCUSDT-metrics-{date_daily}.zip",
    
    # Fallback: Coin Perp
    f"{base_daily_cm}/metrics/BTCUSD_PERP/BTCUSD_PERP-metrics-{date_daily}.zip",
    
    # Control: User's example (need older date for 230331 contract?)
    # Let's try a known date for user's example
    "https://data.binance.vision/data/futures/cm/daily/metrics/BTCUSD_230331/BTCUSD_230331-metrics-2023-03-30.zip"
]

print("--- Probing Metrics (DAILY) ---")
for url in candidates:
    check_url(url)

# 3. Klines (Control)
url_klines = f"{base}/klines/{symbol}/1h/{symbol}-1h-{date}.zip"
check_url(url_klines)
