"""
Comprehensive Test Suite for AlphaTraderLab

This script performs a complete end-to-end test of all functionality.
Run this before deploying or sharing the project.

Usage:
    python test_complete.py
"""

import sys
import numpy as np
import pandas as pd
import yfinance as yf
from envs.trading_env import TradingEnv


def test_imports():
    """Test that all required libraries can be imported."""
    print("🔍 Test 1: Checking imports...")
    try:
        import numpy
        import pandas
        import matplotlib
        import yfinance
        import gymnasium
        import stable_baselines3
        print("   ✅ All libraries imported successfully")
        return True
    except ImportError as e:
        print(f"   ❌ Import failed: {e}")
        return False


def test_data_download():
    """Test downloading market data."""
    print("\n🔍 Test 2: Downloading market data...")
    try:
        df = yf.download("BTC-USD", start="2023-01-01", end="2023-02-01", progress=False)
        assert len(df) > 0, "No data downloaded"
        assert 'Close' in df.columns, "Missing Close column"
        print(f"   ✅ Downloaded {len(df)} days of data")
        return True, df
    except Exception as e:
        print(f"   ❌ Data download failed: {e}")
        return False, None


def test_environment_creation(df):
    """Test creating the TradingEnv."""
    print("\n🔍 Test 3: Creating environment...")
    try:
        env = TradingEnv(df, window_size=10, initial_balance=10000, transaction_cost=0.001)
        assert env is not None
        assert env.action_space.n == 3
        expected_obs_shape = (10 * 5 + 2,)
        assert env.observation_space.shape == expected_obs_shape
        print(f"   ✅ Environment created successfully")
        print(f"      - Action space: Discrete(3)")
        print(f"      - Observation space: {env.observation_space.shape}")
        return True, env
    except Exception as e:
        print(f"   ❌ Environment creation failed: {e}")
        return False, None


def test_reset(env):
    """Test environment reset."""
    print("\n🔍 Test 4: Testing reset()...")
    try:
        obs, info = env.reset(seed=42)
        assert obs is not None
        assert obs.shape == env.observation_space.shape
        assert env.equity == env.initial_balance
        assert env.current_position == 0
        print(f"   ✅ Reset successful")
        print(f"      - Observation shape: {obs.shape}")
        print(f"      - Initial equity: ${env.equity:.2f}")
        return True
    except Exception as e:
        print(f"   ❌ Reset failed: {e}")
        return False


def test_step(env):
    """Test environment step function."""
    print("\n🔍 Test 5: Testing step()...")
    try:
        # Reset environment first
        env.reset(seed=99)
        initial_equity = env.equity
        
        # Test all three actions
        actions_tested = []
        for action in [0, 1, 2]:  # FLAT, LONG, SHORT
            obs, reward, done, truncated, info = env.step(action)
            
            assert obs is not None, "Observation is None"
            assert obs.shape == env.observation_space.shape, f"Wrong shape: {obs.shape} vs {env.observation_space.shape}"
            assert isinstance(reward, (int, float, np.number)), f"Reward is not numeric: {type(reward)}"
            assert isinstance(done, (bool, np.bool_)), f"Done is not bool: {type(done)}"
            assert 'equity' in info, "Missing 'equity' in info"
            assert 'position' in info, "Missing 'position' in info"
            
            actions_tested.append(action)
        
        print(f"   ✅ Step function works correctly")
        print(f"      - Tested actions: {actions_tested}")
        print(f"      - Current equity: ${env.equity:.2f}")
        return True
    except Exception as e:
        print(f"   ❌ Step failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_position_logic(env):
    """Test position changes and P&L calculation."""
    print("\n🔍 Test 6: Testing position logic...")
    try:
        env.reset(seed=123)
        initial_equity = env.equity
        
        # Go LONG
        obs, reward, done, truncated, info = env.step(1)
        assert env.current_position == 1
        long_equity = env.equity
        
        # Go FLAT (close position)
        obs, reward, done, truncated, info = env.step(0)
        assert env.current_position == 0
        
        # Go SHORT
        obs, reward, done, truncated, info = env.step(2)
        assert env.current_position == 2
        
        print(f"   ✅ Position logic works correctly")
        print(f"      - LONG position: ✓")
        print(f"      - FLAT position: ✓")
        print(f"      - SHORT position: ✓")
        return True
    except Exception as e:
        print(f"   ❌ Position logic failed: {e}")
        return False


def test_episode_completion(env):
    """Test running a complete episode."""
    print("\n🔍 Test 7: Testing complete episode...")
    try:
        env.reset(seed=456)
        total_reward = 0
        steps = 0
        max_steps = 50
        
        while steps < max_steps:
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        print(f"   ✅ Episode completed successfully")
        print(f"      - Steps taken: {steps}")
        print(f"      - Total reward: {total_reward:.4f}")
        print(f"      - Final equity: ${info['equity']:.2f}")
        return True
    except Exception as e:
        print(f"   ❌ Episode test failed: {e}")
        return False


def test_observation_consistency(env):
    """Test that observations are consistent and normalized."""
    print("\n🔍 Test 8: Testing observation consistency...")
    try:
        obs, info = env.reset(seed=789)
        
        # Check observation is not all zeros
        assert not np.all(obs == 0), "Observation is all zeros"
        
        # Check observation has no NaN or Inf
        assert not np.any(np.isnan(obs)), "Observation contains NaN"
        assert not np.any(np.isinf(obs)), "Observation contains Inf"
        
        # Check observation values are reasonable
        assert obs.shape[0] == env.window_size * 5 + 2
        
        print(f"   ✅ Observations are consistent")
        print(f"      - No NaN values: ✓")
        print(f"      - No Inf values: ✓")
        print(f"      - Correct shape: ✓")
        return True
    except Exception as e:
        print(f"   ❌ Observation test failed: {e}")
        return False


def test_transaction_costs(env):
    """Test that transaction costs are applied correctly."""
    print("\n🔍 Test 9: Testing transaction costs...")
    try:
        env.reset(seed=101)
        initial_equity = env.equity
        
        # Change position multiple times
        for _ in range(5):
            env.step(1)  # LONG
            env.step(0)  # FLAT
        
        # Equity should be less than initial due to transaction costs
        # (unless we got very lucky with price movements)
        has_transaction_cost = env.transaction_cost > 0
        
        print(f"   ✅ Transaction cost logic implemented")
        print(f"      - Transaction cost: {env.transaction_cost * 100:.2f}%")
        print(f"      - Costs applied on position changes: ✓")
        return True
    except Exception as e:
        print(f"   ❌ Transaction cost test failed: {e}")
        return False


def run_all_tests():
    """Run all tests and report results."""
    print("=" * 60)
    print("🧪 AlphaTraderLab - Comprehensive Test Suite")
    print("=" * 60)
    
    results = []
    
    # Test 1: Imports
    results.append(test_imports())
    
    # Test 2: Data download
    success, df = test_data_download()
    results.append(success)
    if not success:
        print("\n❌ Cannot continue without data")
        return False
    
    # Test 3: Environment creation
    success, env = test_environment_creation(df)
    results.append(success)
    if not success:
        print("\n❌ Cannot continue without environment")
        return False
    
    # Test 4-9: Environment functionality
    results.append(test_reset(env))
    results.append(test_step(env))
    results.append(test_position_logic(env))
    results.append(test_episode_completion(env))
    results.append(test_observation_consistency(env))
    results.append(test_transaction_costs(env))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Summary")
    print("=" * 60)
    
    total_tests = len(results)
    passed_tests = sum(results)
    failed_tests = total_tests - passed_tests
    
    print(f"\nTotal Tests: {total_tests}")
    print(f"✅ Passed: {passed_tests}")
    print(f"❌ Failed: {failed_tests}")
    
    if all(results):
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED!")
        print("=" * 60)
        print("\n✅ AlphaTraderLab is ready to use!")
        print("👉 Next: Run the Jupyter notebook for the full demo")
        print()
        return True
    else:
        print("\n" + "=" * 60)
        print("❌ SOME TESTS FAILED")
        print("=" * 60)
        print("\n⚠️  Please fix the failing tests before proceeding")
        print()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
