#!/usr/bin/env python3
"""
Comprehensive test to verify clamp_score() ensures no reward is exactly 0.0 or 1.0.
Tests the fixed graders.py and environment.py.
"""

import sys
import random
from server.rubric import clamp_score

def test_boundary_clamping():
    """Test 1: Boundary values should be clamped."""
    print('=== Test 1: Boundary Clamping ===')
    test_values = [0.0, 0.001, 0.01, 0.5, 0.99, 0.999, 1.0]
    for val in test_values:
        clamped = clamp_score(val)
        print(f'  clamp_score({val}) = {clamped}')
        if clamped == 0.0 or clamped == 1.0:
            print(f'    ❌ FAIL: Got exact boundary value!')
            return False
        if not (0.01 <= clamped <= 0.99):
            print(f'    ❌ FAIL: Outside valid range!')
            return False
    return True


def test_random_values():
    """Test 2: 1000 random values."""
    print('\n=== Test 2: 1000 Random Values ===')
    random.seed(42)
    min_val, max_val = 1.0, 0.0
    
    for i in range(1000):
        raw = random.random()
        clamped = clamp_score(raw)
        min_val = min(min_val, clamped)
        max_val = max(max_val, clamped)
        
        if clamped == 0.0 or clamped == 1.0:
            print(f'  ❌ Step {i}: clamp_score({raw}) = {clamped} (EXACT boundary!)')
            return False
        if not (0.01 <= clamped <= 0.99):
            print(f'  ❌ Step {i}: clamp_score({raw}) = {clamped} (OUT OF RANGE!)')
            return False
    
    print(f'  ✅ All 1000 values strictly in (0.01, 0.99)')
    print(f'  Min observed: {min_val}')
    print(f'  Max observed: {max_val}')
    return True


def test_rounding():
    """Test 3: Rounding behavior."""
    print('\n=== Test 3: Rounding Behavior ===')
    rounding_tests = [0.015, 0.0145, 0.9999, 0.9994]
    for val in rounding_tests:
        clamped = clamp_score(val)
        print(f'  clamp_score({val}) = {clamped}')
    return True


def test_environment_returns():
    """Test 4: Environment.step() returns clamped rewards."""
    print('\n=== Test 4: Environment Integration ===')
    from server.ticket_system_environment import TicketSystemEnvironment
    from models import TicketSystemAction
    
    env = TicketSystemEnvironment()
    env.reset(task="easy")
    
    # Run multiple steps and collect all returned rewards
    rewards = []
    for i in range(5):
        action = TicketSystemAction(action_type="read_ticket")
        obs = env.step(action)
        reward = obs.reward
        rewards.append(reward)
        
        print(f'  Step {i+1}: returned reward = {reward}')
        
        if reward == 0.0 or reward == 1.0:
            print(f'    ❌ FAIL: Got exact boundary!')
            return False
        if not (0.01 <= reward <= 0.99):
            print(f'    ❌ FAIL: Outside valid range!')
            return False
    
    print(f'  ✅ All environment step() rewards clamped correctly')
    return True


if __name__ == "__main__":
    tests = [
        test_boundary_clamping,
        test_random_values,
        test_rounding,
        test_environment_returns,
    ]
    
    all_passed = True
    for test in tests:
        if not test():
            all_passed = False
    
    print('\n' + '='*50)
    if all_passed:
        print('✅ ALL TESTS PASSED - No reward will ever be exactly 0.0 or 1.0')
        sys.exit(0)
    else:
        print('❌ SOME TESTS FAILED')
        sys.exit(1)
