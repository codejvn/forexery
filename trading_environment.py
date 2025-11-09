"""
Enhanced Gym environment for USD/JPY forex trading with continuous actions
Agent observes: sentiment features, technical indicators, portfolio state
Agent actions: continuous values [-1, 1] for position sizing
Reward: mixed daily P&L + quarterly performance
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from config import (
    INITIAL_BALANCE, INITIAL_USD_RATIO, INITIAL_JPY_RATIO,
    TRANSACTION_COST, EPISODE_LENGTH, DAILY_REWARD_WEIGHT,
    QUARTERLY_REWARD_WEIGHT, SHARPE_BONUS_WEIGHT,
    ACTION_LOW, ACTION_HIGH, TRADING_PENALTY
)

class JPYUSDTradingEnv(gym.Env):
    """
    Continuous action forex trading environment for USD/JPY

    Action Space: Box([-1, 1])
        - Positive: Buy USD (sell JPY) with that fraction of balance
        - Negative: Sell USD (buy JPY) with that fraction of balance
        - Zero: Hold current position

    Observation Space: Box with:
        - Technical indicators (USDJPY price, returns, RSI, MACD, etc.)
        - USA sentiment features (mean, MA, volatility, momentum, etc.)
        - Japan sentiment features (mean, MA, volatility, momentum, etc.)
        - Derived features (sentiment divergence, combined scores)
        - Portfolio state (USD ratio, JPY ratio, total value)

    Reward: Mixed daily + quarterly
        - Daily: P&L change * DAILY_REWARD_WEIGHT
        - Quarterly: Final return * QUARTERLY_REWARD_WEIGHT
        - Bonus: Sharpe ratio * SHARPE_BONUS_WEIGHT
    """

    metadata = {'render_modes': ['human']}

    def __init__(self, df, initial_balance=INITIAL_BALANCE, episode_length=EPISODE_LENGTH):
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.episode_length = episode_length

        # Identify feature columns (exclude 'date')
        self.feature_columns = [col for col in df.columns if col != 'date']

        # Verify required columns exist
        required_cols = ['USDJPY_Close']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Action space: continuous [-1, 1]
        self.action_space = spaces.Box(
            low=ACTION_LOW,
            high=ACTION_HIGH,
            shape=(1,),
            dtype=np.float32
        )

        # Observation space: all features + portfolio state (3 values)
        n_features = len(self.feature_columns) + 3  # +3 for USD ratio, JPY ratio, total return
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(n_features,),
            dtype=np.float32
        )

        # Calculate maximum possible episodes
        self.max_episodes = len(self.df) // self.episode_length

        print(f"Environment initialized:")
        print(f"  Data points: {len(self.df)}")
        print(f"  Episode length: {self.episode_length} days")
        print(f"  Max episodes: {self.max_episodes}")
        print(f"  Features: {len(self.feature_columns)}")
        print(f"  Observation space: {self.observation_space.shape}")
        print(f"  Action space: {self.action_space.shape}")

    def reset(self, seed=None, options=None):
        """Reset environment for new episode"""
        super().reset(seed=seed)

        # Random starting point (ensure full episode fits)
        max_start = len(self.df) - self.episode_length - 1
        if max_start <= 0:
            self.episode_start = 0
        else:
            self.episode_start = self.np_random.integers(0, max_start)

        self.current_step = 0
        self.global_step = self.episode_start

        # Initialize portfolio: 50% USD, 50% JPY
        self.usd_balance = self.initial_balance * INITIAL_USD_RATIO
        self.jpy_balance = self.initial_balance * INITIAL_JPY_RATIO

        # Get initial exchange rate
        self.initial_exchange_rate = self.df.iloc[self.global_step]['USDJPY_Close']

        # Calculate initial total value in USD
        self.initial_value = self.usd_balance + (self.jpy_balance / self.initial_exchange_rate)

        # Track performance
        self.daily_returns = []
        self.daily_pnls = []
        self.trade_history = []
        self.total_trades = 0

        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def _get_observation(self):
        """Get current state observation"""
        row = self.df.iloc[self.global_step]

        # Extract all features
        obs = []
        for feature in self.feature_columns:
            value = row[feature]
            # Handle any NaN values
            if pd.isna(value):
                value = 0.0
            obs.append(float(value))

        # Add portfolio state
        current_exchange_rate = row['USDJPY_Close']
        total_value_usd = self.usd_balance + (self.jpy_balance / current_exchange_rate)

        usd_ratio = self.usd_balance / total_value_usd if total_value_usd > 0 else 0.5
        jpy_ratio = (self.jpy_balance / current_exchange_rate) / total_value_usd if total_value_usd > 0 else 0.5
        total_return = (total_value_usd / self.initial_value - 1.0) if self.initial_value > 0 else 0.0

        obs.extend([usd_ratio, jpy_ratio, total_return])

        return np.array(obs, dtype=np.float32)

    def _get_info(self):
        """Get additional info dict"""
        current_exchange_rate = self.df.iloc[self.global_step]['USDJPY_Close']
        total_value_usd = self.usd_balance + (self.jpy_balance / current_exchange_rate)

        return {
            'step': self.current_step,
            'global_step': self.global_step,
            'date': self.df.iloc[self.global_step]['date'],
            'usd_balance': self.usd_balance,
            'jpy_balance': self.jpy_balance,
            'total_value_usd': total_value_usd,
            'total_return': (total_value_usd / self.initial_value - 1.0),
            'exchange_rate': current_exchange_rate,
            'total_trades': self.total_trades
        }

    def step(self, action):
        """Execute one time step"""

        # Clip action to valid range
        action = np.clip(action[0], ACTION_LOW, ACTION_HIGH)

        # Get current state
        current_row = self.df.iloc[self.global_step]
        current_exchange_rate = current_row['USDJPY_Close']

        # Calculate value before action
        value_before = self.usd_balance + (self.jpy_balance / current_exchange_rate)

        # Execute trading action
        if action > 0.01:  # Buy USD (sell JPY)
            # Calculate how much USD to buy
            jpy_available = self.jpy_balance
            jpy_to_sell = jpy_available * action
            usd_to_buy = (jpy_to_sell / current_exchange_rate) * (1 - TRANSACTION_COST)

            self.jpy_balance -= jpy_to_sell
            self.usd_balance += usd_to_buy
            self.total_trades += 1

            self.trade_history.append({
                'step': self.current_step,
                'action': action,
                'type': 'BUY_USD',
                'amount': usd_to_buy,
                'rate': current_exchange_rate
            })

        elif action < -0.01:  # Sell USD (buy JPY)
            # Calculate how much USD to sell
            usd_available = self.usd_balance
            usd_to_sell = usd_available * abs(action)
            jpy_to_buy = (usd_to_sell * current_exchange_rate) * (1 - TRANSACTION_COST)

            self.usd_balance -= usd_to_sell
            self.jpy_balance += jpy_to_buy
            self.total_trades += 1

            self.trade_history.append({
                'step': self.current_step,
                'action': action,
                'type': 'SELL_USD',
                'amount': usd_to_sell,
                'rate': current_exchange_rate
            })

        # Move to next step
        self.current_step += 1
        self.global_step += 1

        # Get next state
        next_row = self.df.iloc[self.global_step]
        next_exchange_rate = next_row['USDJPY_Close']

        # Calculate value after market movement
        value_after = self.usd_balance + (self.jpy_balance / next_exchange_rate)

        # Calculate daily P&L
        daily_pnl = value_after - value_before
        daily_return = (value_after / value_before - 1.0) if value_before > 0 else 0.0

        self.daily_pnls.append(daily_pnl)
        self.daily_returns.append(daily_return)

        # Calculate reward (daily component)
        reward = daily_pnl * DAILY_REWARD_WEIGHT

        # Apply trading penalty if agent traded (to discourage overtrading)
        if abs(action) > 0.01:  # If agent made a trade
            reward -= TRADING_PENALTY

        # Check if episode is done
        done = self.current_step >= self.episode_length

        if done:
            # Add quarterly reward component
            total_return = (value_after / self.initial_value - 1.0)
            quarterly_reward = total_return * QUARTERLY_REWARD_WEIGHT
            reward += quarterly_reward

            # Add Sharpe ratio bonus (risk-adjusted returns)
            if len(self.daily_returns) > 1:
                returns_array = np.array(self.daily_returns)
                sharpe_ratio = np.mean(returns_array) / (np.std(returns_array) + 1e-6) * np.sqrt(252)
                sharpe_bonus = sharpe_ratio * SHARPE_BONUS_WEIGHT
                reward += sharpe_bonus

        # Get next observation
        obs = self._get_observation()
        info = self._get_info()

        # Add episode stats to info if done
        if done:
            info['episode'] = {
                'total_return': (value_after / self.initial_value - 1.0),
                'total_trades': self.total_trades,
                'mean_daily_return': np.mean(self.daily_returns),
                'std_daily_return': np.std(self.daily_returns),
                'sharpe_ratio': np.mean(self.daily_returns) / (np.std(self.daily_returns) + 1e-6) * np.sqrt(252),
                'max_drawdown': self._calculate_max_drawdown(),
                'final_value': value_after
            }

        terminated = done
        truncated = False

        return obs, reward, terminated, truncated, info

    def _calculate_max_drawdown(self):
        """Calculate maximum drawdown during episode"""
        if len(self.daily_returns) == 0:
            return 0.0

        cumulative = np.cumprod(1 + np.array(self.daily_returns))
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return np.min(drawdown)

    def render(self, mode='human'):
        """Render current state"""
        info = self._get_info()

        print(f"\n{'='*60}")
        print(f"Step {self.current_step}/{self.episode_length} | Date: {info['date']}")
        print(f"{'='*60}")
        print(f"Portfolio:")
        print(f"  USD Balance: ${info['usd_balance']:.2f}")
        print(f"  JPY Balance: ¥{info['jpy_balance']:.2f}")
        print(f"  Total Value (USD): ${info['total_value_usd']:.2f}")
        print(f"  Return: {info['total_return']*100:+.2f}%")
        print(f"Exchange Rate: {info['exchange_rate']:.4f}")
        print(f"Total Trades: {info['total_trades']}")
        print(f"{'='*60}\n")

    def get_trade_history(self):
        """Return trade history as DataFrame"""
        if len(self.trade_history) == 0:
            return pd.DataFrame()
        return pd.DataFrame(self.trade_history)


# Test the environment
if __name__ == "__main__":
    print("Testing JPYUSDTradingEnv with continuous actions...\n")

    # Load merged data
    try:
        df = pd.read_csv('merged_trading_data.csv')
        df['date'] = pd.to_datetime(df['date'])
        print(f"Loaded data: {len(df)} days")
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"Features: {len([col for col in df.columns if col != 'date'])}")
    except FileNotFoundError:
        print("Error: merged_trading_data.csv not found!")
        print("Run data_merger.py first to create the merged dataset.")
        exit(1)

    # Create environment
    env = JPYUSDTradingEnv(df, episode_length=63)

    print("\n" + "="*60)
    print("Running test episode with random actions...")
    print("="*60)

    # Run one episode
    obs, info = env.reset()
    print(f"\nInitial observation shape: {obs.shape}")
    print(f"Initial state:")
    print(f"  USD: ${info['usd_balance']:.2f}")
    print(f"  JPY: ¥{info['jpy_balance']:.2f}")
    print(f"  Total: ${info['total_value_usd']:.2f}")

    episode_rewards = []

    for i in range(63):
        # Random action in [-1, 1]
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        episode_rewards.append(reward)

        # Render every 20 steps
        if i % 20 == 0 or terminated:
            print(f"\nStep {i}: Action={action[0]:.3f}, Reward={reward:.4f}")
            print(f"  Portfolio Value: ${info['total_value_usd']:.2f} ({info['total_return']*100:+.2f}%)")

        if terminated:
            break

    print("\n" + "="*60)
    print("EPISODE COMPLETE")
    print("="*60)
    print(f"Final Stats:")
    if 'episode' in info:
        ep = info['episode']
        print(f"  Total Return: {ep['total_return']*100:+.2f}%")
        print(f"  Sharpe Ratio: {ep['sharpe_ratio']:.3f}")
        print(f"  Max Drawdown: {ep['max_drawdown']*100:.2f}%")
        print(f"  Total Trades: {ep['total_trades']}")
        print(f"  Mean Daily Return: {ep['mean_daily_return']*100:.4f}%")
        print(f"  Std Daily Return: {ep['std_daily_return']*100:.4f}%")
        print(f"  Final Value: ${ep['final_value']:.2f}")

    print(f"\nTotal Episode Reward: {sum(episode_rewards):.4f}")
    print("\n✓ Environment test complete!")
