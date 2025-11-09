"""
Quick test to verify the complete pipeline works
Tests with minimal training timesteps
"""

import sys

print("="*70)
print("QUICK PIPELINE TEST")
print("="*70)

# Test 1: Data Merger
print("\n[1/3] Testing data merger...")
try:
    import data_merger
    df = data_merger.main()
    print(f"[OK] Data merger works! Created dataset with {len(df)} days")
except Exception as e:
    print(f"[FAIL] Data merger failed: {e}")
    sys.exit(1)

# Test 2: Environment
print("\n[2/3] Testing trading environment...")
try:
    from trading_environment import JPYUSDTradingEnv
    import pandas as pd

    df = pd.read_csv('merged_trading_data.csv')
    df['date'] = pd.to_datetime(df['date'])

    env = JPYUSDTradingEnv(df, episode_length=63)
    obs, info = env.reset()

    # Take a few random actions
    for _ in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    print(f"[OK] Environment works!")
    print(f"  Observation shape: {obs.shape}")
    print(f"  Action space: {env.action_space}")
except Exception as e:
    print(f"[FAIL] Environment test failed: {e}")
    sys.exit(1)

# Test 3: Quick Training (just 1000 steps)
print("\n[3/3] Testing training with minimal timesteps...")
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor

    # Create simple environment
    def make_env():
        env = JPYUSDTradingEnv(df, episode_length=63)
        return Monitor(env, filename=None, allow_early_resets=True)

    env = DummyVecEnv([make_env])

    # Train for just 1000 steps
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=1000, progress_bar=True)

    # Test prediction
    obs = env.reset()
    action, _ = model.predict(obs)

    print(f"[OK] Training works!")
    print(f"  Model can predict actions: {action}")

except Exception as e:
    print(f"[FAIL] Training test failed: {e}")
    sys.exit(1)

print("\n" + "="*70)
print("ALL TESTS PASSED!")
print("="*70)
print("\nYou can now run the full pipeline:")
print("  1. python data_merger.py")
print("  2. python training.py")
print("  3. python evaluation.py")
print("="*70)
