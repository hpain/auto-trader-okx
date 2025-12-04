import sys
import os

print("--- sys.path ---")
for p in sys.path:
    print(p)

print("\n--- Investigating 'ccxt' module ---")
try:
    import ccxt
    print(f"Successfully imported 'ccxt'.")
    print(f"Location of imported 'ccxt' module: {ccxt.__file__}")
    
    # Check if it's a package (directory) or a single file
    if '__init__.py' in os.path.basename(ccxt.__file__):
        print("'ccxt' appears to be a package (directory), which is correct.")
    else:
        print("'ccxt' appears to be a single file, which is the problem.")

except ImportError as e:
    print(f"Failed to import 'ccxt': {e}")
except AttributeError:
    print("Could not determine 'ccxt' file location ('__file__' attribute is missing). This is unusual.")
    print(f"ccxt module object: {ccxt}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

print("\n--- Attempting to import 'ccxt.async_support' ---")
try:
    import ccxt.async_support
    print(f"Successfully imported 'ccxt.async_support'.")
    print(f"Location of 'ccxt.async_support': {ccxt.async_support.__file__}")
except ImportError as e:
    print(f"Failed to import 'ccxt.async_support': {e}")
