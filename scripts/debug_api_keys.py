import os
import sys
from dotenv import load_dotenv

# Load from .env file directly to check file content vs os.environ
load_dotenv(override=True)

def check_key(name):
    val = os.getenv(name)
    print(f"Checking {name}...")
    if not val:
        print(f"  ❌ ERROR: {name} is NOT set or empty.")
        return
    
    print(f"  ✅ Found. Length: {len(val)}")
    
    if val.startswith('"') or val.startswith("'"):
        print(f"  ⚠️ WARNING: Value starts with a quote ({val[0]}). This might be interpreted literally!")
    
    if val.endswith('"') or val.endswith("'"):
        print(f"  ⚠️ WARNING: Value ends with a quote ({val[-1]}).")
        
    if " " in val:
        print(f"  ⚠️ WARNING: Value contains spaces. Check for accidental whitespace.")
        
    masked = val[:4] + "****" + val[-4:] if len(val) > 8 else "****"
    print(f"  👀 Masked: {masked}")

print("--- API Key Diagnostic ---")
check_key("OKX_API_KEY")
check_key("OKX_SECRET_KEY")
check_key("OKX_PASSPHRASE")

print("\n--- Live Key Diagnostic (for Arb Bot) ---")
check_key("LIVE_OKX_API_KEY")
