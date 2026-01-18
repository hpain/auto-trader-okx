
import requests
import time

def check_binance(url, name):
    print(f"Checking {name} ({url})...")
    try:
        start = time.time()
        response = requests.get(url, timeout=5)
        latency = (time.time() - start) * 1000
        
        if response.status_code == 200:
            print(f"✅ {name} is UP! (Status: {response.status_code}, Latency: {latency:.2f}ms)")
        else:
            print(f"❌ {name} returned error status: {response.status_code}")
            print(f"Response: {response.text[:200]}")
            
    except Exception as e:
        print(f"❌ {name} connection FAILED: {e}")

if __name__ == "__main__":
    # Testnet
    check_binance("https://testnet.binance.vision/api/v3/ping", "Binance Spot Testnet")
    # Mainnet (for comparison)
    check_binance("https://api.binance.com/api/v3/ping", "Binance Mainnet")
