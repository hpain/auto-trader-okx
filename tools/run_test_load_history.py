import sys
import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import tests.test_load_history as t

try:
    t.test_load_history_short_names()
    print("TEST PASSED: test_load_history_short_names")
except AssertionError as e:
    print("TEST FAILED: test_load_history_short_names", e)
except Exception as e:
    print("TEST ERROR:", e)
