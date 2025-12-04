import pandas as pd
import numpy as np

def test_assertion_logic():
    # Test case 1: regime_changed is True, expect False - should fail
    result1 = {'regime_changed': True}
    try:
        assert result1['regime_changed'] is False
        print("Test 1 passed (unexpected)")
    except AssertionError:
        print("Test 1 failed as expected (regime_changed=True, assertion expects False)")
    
    # Test case 2: regime_changed is False, expect False - should pass
    result2 = {'regime_changed': False}
    try:
        assert result2['regime_changed'] is False
        print("Test 2 passed as expected (regime_changed=False, assertion expects False)")
    except AssertionError:
        print("Test 2 failed (unexpected, regime_changed=False, assertion expects False)")

test_assertion_logic()