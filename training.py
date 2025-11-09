"""
Train reinforcement learning agent on JPY/USD trading
Uses PPO (Proximal Policy Optimization) algorithm
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
    TRAINING_TIMESTEPS, N_STEPS, BATCH_SIZE
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
    
    print(f"Data splits:")
    print(f"  Train: {len(df_train)} days ({df_train['date'].min()} to {df_train['date'].max()})")
    print(f"  Val:   {len(df_val)} days ({df_val['date'].min()} to {df_val['date'].max()})")
    print(f"  Test:  {len(df_test)} days ({df_test['date'].min()} to {df_test['date'].max()})")
    
    return df_train, df_val, df_test

def make_env(df, rank=0):
    """Create environment wrapped in Monitor"""
    def _init():
        env = JPYUSDTradingEnv(df)
        env = Monitor(env, filename=None, allow_early_resets=True)
        return env
    return _init

def train_agent(df_train, df_val):
    """Train PPO agent"""
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
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
        verbose=1
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path='./models/checkpoints',
        name_prefix='jpyusd_agent'
    )
    
    # Create PPO model
    print("\nInitializing PPO agent...")
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=LEARNING_RATE,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,  # Encourage exploration
        verbose=1,
        tensorboard_log="./tensorboard_logs/"
    )
    
    print(f"\nTraining for {TRAINING_TIMESTEPS} timesteps...")
    print("This may take 10-30 minutes depending on your hardware.")
    print("Monitor progress in tensorboard: tensorboard --logdir ./tensorboard_logs\n")
    
    # Train
    model.learn(
        total_timesteps=TRAINING_TIMESTEPS,
        callback=[eval_callback, checkpoint_callback]
    )
    
    # Save final model
    model.save(MODEL_SAVE_PATH)
    print(f"\n✓ Training complete! Model saved to {MODEL_SAVE_PATH}")
    
    return model

def main():
    print("="*50)
    print("STEP 5: Training RL Agent")
    print("="*50)
    
    # Load data
    df_train, df_val, df_test = load_and_split_data()
    
    # Save test set for later
    df_test.to_csv('test_data.csv', index=False)
    
    # Train
    model = train_agent(df_train, df_val)
    
    print("\n" + "="*50)
    print("Training complete!")
    print("Next step: Run 6_evaluate_agent.py to test performance")
    print("="*50)

if __name__ == "__main__":
    main()