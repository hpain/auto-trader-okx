#!/usr/bin/env python
"""Debug script for ccxt import issue"""
import sys
print(f"Python executable: {sys.executable}")
print(f"Python path: {sys.path}")

try:
    import ccxt
    print(f"CCXT version: {ccxt.__version__}")
    print(f"CCXT path: {ccxt.__file__ if hasattr(ccxt, '__file__') else 'N/A'}")
    
    # Check for async_support
    import ccxt.async_support
    print("ccxt.async_support imported successfully!")
    print(f"ccxt.async_support path: {ccxt.async_support.__file__}")
    
    # Try the specific import method used in the code
    import importlib
    ccxt_dynamic = importlib.import_module("ccxt.async_support")
    print("Dynamic import of ccxt.async_support successful!")
    
except ImportError as e:
    print(f"Import error: {e}")
    import traceback
    traceback.print_exc()
    
except Exception as e:
    print(f"Other error: {e}")
    import traceback
    traceback.print_exc()