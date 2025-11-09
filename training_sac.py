"""
Alternative training with SAC (Soft Actor-Critic)
SAC is often better than PPO for continuous action spaces
"""

import pandas as pd
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
import os

from trading_environment import JPYUSDTradingEnv
from config import (
    MERGED_DATA_OUTPUT, TRAIN_SPLIT, VAL_SPLIT,
    EPISODE_LENGTH
)

def load_and_split_data():
    """Load data and split into train/val/test"""
    df = pd.read_csv(MERGED_DATA_OUTPUT)
    df['date'] = pd.to_datetime(df['date'])

    n = len(df)
    train_end = int(n * TRAIN_SPLIT)
    val_end = int(n * (TRAIN_SPLIT + VAL_SPLIT))

    df_train = df.iloc[:train_end].copy()
    df_val = df.iloc[train_end:val_end].copy()
    df_test = df.iloc[val_end:].copy()

    print(f"\nData splits:")
    print(f"  Train: {len(df_train)} days")
    print(f"  Val:   {len(df_val)} days")
    print(f"  Test:  {len(df_test)} days")

    return df_train, df_val, df_test

def make_env(df):
    """Create environment wrapped in Monitor"""
    def _init():
        env = JPYUSDTradingEnv(df, episode_length=EPISODE_LENGTH)
        env = Monitor(env, filename=None, allow_early_resets=True)
        return env
    return _init

def train_agent(df_train, df_val):
    """Train SAC agent"""

    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('models/sac_best', exist_ok=True)
    os.makedirs('models/sac_checkpoints', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    # Create vectorized environments
    train_env = DummyVecEnv([make_env(df_train)])
    val_env = DummyVecEnv([make_env(df_val)])

    # Callbacks
    eval_callback = EvalCallback(
        val_env,
        best_model_save_path='./models/sac_best',
        log_path='./logs',
        eval_freq=5000,
        deterministic=True,
        render=False,
        verbose=1
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path='./models/sac_checkpoints',
        name_prefix='jpyusd_sac'
    )

    # Create SAC model
    print("\nInitializing SAC agent...")
    model = SAC(
        "MlpPolicy",
        train_env,
        learning_rate=0.0003,
        buffer_size=100000,
        learning_starts=1000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=1,
        tensorboard_log="./tensorboard_logs_sac/"
    )

    print(f"\nTraining SAC for 100,000 timesteps...")

    # Train
    model.learn(
        total_timesteps=100000,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True
    )

    # Save final model
    model.save('models/jpyusd_sac_agent')
    print(f"\n✓ Training complete! SAC model saved")

    return model

def main():
    print("="*60)
    print("TRAINING SAC AGENT (Alternative to PPO)")
    print("="*60)

    # Load data
    df_train, df_val, df_test = load_and_split_data()

    # Train
    model = train_agent(df_train, df_val)

    print("\n" + "="*60)
    print("SAC TRAINING COMPLETE!")
    print("="*60)
    print("To evaluate: Modify evaluation.py to load 'models/jpyusd_sac_agent'")

if __name__ == "__main__":
    main()
