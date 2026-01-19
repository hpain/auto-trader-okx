import asyncio
import os
import sys
import yaml
import json
from pprint import pprint

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from exchange.okx_exchange import OKXExchange
from dotenv import load_dotenv

# Load env vars from .env file to match Docker behavior
load_dotenv()

def load_config(path):
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Inject Env Vars into Config (Simulating config_loader.py)
    if os.getenv('OKX_API_KEY'):
        config.setdefault('okx', {})['api_key'] = os.getenv('OKX_API_KEY')
    if os.getenv('OKX_SECRET_KEY'):
        config.setdefault('okx', {})['secret_key'] = os.getenv('OKX_SECRET_KEY')
    if os.getenv('OKX_PASSPHRASE'):
        config.setdefault('okx', {})['passphrase'] = os.getenv('OKX_PASSPHRASE')
        
    return config

async def main():
    config = load_config('config/settings.yaml')
    creds = config['okx']
    
    # Force flag='0' (Live) or '1' (Sim) based on user report
    # The log implies Live (User mentioned "Live Bot" and "Order Executed")
    # But let's respect the config.
    is_sandbox = str(creds.get('flag', '0')) == '1'
    
    print(f"Connecting to OKX (Sandbox: {is_sandbox})...")
    
    exchange = OKXExchange(
        api_key=creds['api_key'],
        api_secret=creds['secret_key'],
        passphrase=creds['passphrase'],
        sandbox=is_sandbox
    )
    
    # Show user which key is being used (Masked)
    masked_key = creds['api_key'][:4] + "****" + creds['api_key'][-4:] if creds['api_key'] and len(creds['api_key']) > 8 else "INVALID_LENGTH"
    print(f"Loaded API Key: {masked_key}")

    
    print("\n--- 1. Testing GET /account/balance ---")
    try:
        balance_resp = exchange._request('GET', '/api/v5/account/balance', authenticated=True)
        print("Raw Balance Response:")
        print(json.dumps(balance_resp, indent=2))
    except Exception as e:
        print(f"Error fetching balance: {e}")

    print("\n--- 2. Testing GET /account/positions (SWAP/FUTURES/MARGIN) ---")
    try:
        # Check positions specifically for SWAP which is common for bots
        pos_resp = exchange._request('GET', '/api/v5/account/positions', authenticated=True)
        print("Raw Positions Response:")
        print(json.dumps(pos_resp, indent=2))
        
        if pos_resp and pos_resp.get('data'):
            print("\nAnalysis:")
            for pos in pos_resp['data']:
                print(f"  Symbol: {pos['instId']}, Size: {pos['pos']}, Type: {pos['instType']}, Mode: {pos['mgnMode']}")
        else:
            print("\nNo open positions found in /account/positions.")
            
    except Exception as e:
        print(f"Error fetching positions: {e}")

        
    print("\n--- 3. Testing GET /trade/orders-pending (Locked Funds?) ---")
    try:
        orders_resp = exchange._request('GET', '/api/v5/trade/orders-pending', authenticated=True)
        print("Raw Pending Orders Response:")
        print(json.dumps(orders_resp, indent=2))
    except Exception as e:
        print(f"Error fetching pending orders: {e}")

if __name__ == "__main__":
    asyncio.run(main())
