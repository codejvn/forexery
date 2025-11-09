"""
Train reinforcement learning agent on USD/JPY trading
Uses PPO (Proximal Policy Optimization) with continuous actions
Episodes are quarterly (63 trading days)
"""

import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
import os

from trading_environment import JPYUSDTradingEnv
from config import (
    MERGED_DATA_OUTPUT, MODEL_SAVE_PATH,
    TRAIN_SPLIT, VAL_SPLIT, LEARNING_RATE,
    TRAINING_TIMESTEPS, N_STEPS, BATCH_SIZE,
    GAMMA, GAE_LAMBDA, CLIP_RANGE, ENT_COEF,
    EPISODE_LENGTH
)

def load_and_split_data():
    """Load data and split into train/val/test"""
    df = pd.read_csv(MERGED_DATA_OUTPUT)
    df['date'] = pd.to_datetime(df['date'])

    print(f"\nLoaded data: {len(df)} days")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")

    n = len(df)
    train_end = int(n * TRAIN_SPLIT)
    val_end = int(n * (TRAIN_SPLIT + VAL_SPLIT))

    df_train = df.iloc[:train_end].copy()
    df_val = df.iloc[train_end:val_end].copy()
    df_test = df.iloc[val_end:].copy()

    # Calculate episodes per split
    train_episodes = len(df_train) // EPISODE_LENGTH
    val_episodes = len(df_val) // EPISODE_LENGTH
    test_episodes = len(df_test) // EPISODE_LENGTH

    print(f"\nData splits:")
    print(f"  Train: {len(df_train)} days ({df_train['date'].min()} to {df_train['date'].max()})")
    print(f"    → {train_episodes} quarters")
    print(f"  Val:   {len(df_val)} days ({df_val['date'].min()} to {df_val['date'].max()})")
    print(f"    → {val_episodes} quarters")
    print(f"  Test:  {len(df_test)} days ({df_test['date'].min()} to {df_test['date'].max()})")
    print(f"    → {test_episodes} quarters")

    return df_train, df_val, df_test

def make_env(df, rank=0):
    """Create environment wrapped in Monitor"""
    def _init():
        env = JPYUSDTradingEnv(df, episode_length=EPISODE_LENGTH)
        env = Monitor(env, filename=None, allow_early_resets=True)
        return env
    return _init

def train_agent(df_train, df_val):
    """Train PPO agent with continuous actions"""

    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('models/best', exist_ok=True)
    os.makedirs('models/checkpoints', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    os.makedirs('tensorboard_logs', exist_ok=True)

    print("\n" + "="*60)
    print("Creating training environments...")
    print("="*60)

    # Create vectorized environments
    train_env = DummyVecEnv([make_env(df_train, i) for i in range(1)])
    val_env = DummyVecEnv([make_env(df_val, 0)])

    # Callbacks
    eval_callback = EvalCallback(
        val_env,
        best_model_save_path='./models/best',
        log_path='./logs',
        eval_freq=5000,
        deterministic=True,
        render=False,
        verbose=1,
        n_eval_episodes=5
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path='./models/checkpoints',
        name_prefix='jpyusd_agent'
    )

    # Create PPO model for continuous action space
    print("\n" + "="*60)
    print("Initializing PPO agent...")
    print("="*60)

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=LEARNING_RATE,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=10,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_range=CLIP_RANGE,
        clip_range_vf=None,
        normalize_advantage=True,
        ent_coef=ENT_COEF,
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_sde=False,  # State Dependent Exploration
        sde_sample_freq=-1,
        target_kl=None,
        tensorboard_log="./tensorboard_logs/",
        policy_kwargs=dict(
            net_arch=[dict(pi=[256, 256], vf=[256, 256])]  # Larger network for complex trading
        ),
        verbose=1,
        seed=42
    )

    print(f"\nModel architecture:")
    print(f"  Policy: MlpPolicy (continuous actions)")
    print(f"  Network: [256, 256] for both policy and value")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Gamma: {GAMMA}")

    print("\n" + "="*60)
    print(f"Training for {TRAINING_TIMESTEPS} timesteps...")
    print("="*60)
    print("This may take 20-60 minutes depending on your hardware.")
    print("Monitor progress in tensorboard:")
    print("  tensorboard --logdir ./tensorboard_logs")
    print("")

    # Train
    model.learn(
        total_timesteps=TRAINING_TIMESTEPS,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True
    )

    # Save final model
    model.save(MODEL_SAVE_PATH)
    print(f"\n✓ Training complete! Model saved to {MODEL_SAVE_PATH}")

    return model

def main():
    print("="*60)
    print("TRAINING RL AGENT FOR USD/JPY TRADING")
    print("="*60)
    print("Configuration:")
    print(f"  Episode length: {EPISODE_LENGTH} days (quarterly)")
    print(f"  Training timesteps: {TRAINING_TIMESTEPS}")
    print(f"  Learning rate: {LEARNING_RATE}")

    # Load data
    df_train, df_val, df_test = load_and_split_data()

    # Save test set for evaluation
    df_test.to_csv('test_data.csv', index=False)
    print(f"\n✓ Test data saved to test_data.csv")

    # Train
    model = train_agent(df_train, df_val)

    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print("Next steps:")
    print("  1. Run evaluation.py to test performance")
    print("  2. View training logs: tensorboard --logdir ./tensorboard_logs")
    print("  3. Best model saved in: ./models/best/")
    print("="*60)

if __name__ == "__main__":
    main()
