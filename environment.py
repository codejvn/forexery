"""
Custom Gym environment for JPY/USD forex trading
Agent observes: sentiment, technical indicators, normalized returns
Agent actions: sell (-1), hold (0), buy (+1)
Reward: P&L from trading decisions
"""

import gym
from gym import spaces
import numpy as np
import pandas as pd

class JPYUSDTradingEnv(gym.Env):
    """
    Forex trading environment for USD/JPY with sentiment
    """
    
    metadata = {'render.modes': ['human']}
    
    def __init__(self, df, initial_balance=10000, transaction_cost=0.0001):
        super(JPYUSDTradingEnv, self).__init__()
        
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        
        # Feature columns for observation
        self.feature_columns = [
            'jpy_normalized_return',
            'jpy_raw_return',
            'usd_index_return',
            'jpy_volatility',
            'jpy_price_vs_sma5',
            'jpy_price_vs_sma20',
            'jpy_rsi',
            'jpy_momentum_5',
            'sentiment_lag1',
            'sentiment_ma5',
            'sentiment_change',
            'sentiment_weighted'
        ]
        
        # Verify features exist
        missing = [f for f in self.feature_columns if f not in df.columns]
        if missing:
            raise ValueError(f"Missing features in dataframe: {missing}")
        
        # Action space: 0=sell, 1=hold, 2=buy
        self.action_space = spaces.Discrete(3)
        
        # Observation space: features + position + unrealized P&L
        n_features = len(self.feature_columns) + 2
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(n_features,),
            dtype=np.float32
        )
        
        # Initialize state
        self.reset()
    
    def reset(self):
        """Reset environment to start of episode"""
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0  # -1=short, 0=neutral, 1=long
        self.entry_price = 0
        self.total_pnl = 0
        self.total_trades = 0
        self.trade_history = []
        
        return self._get_observation()
    
    def _get_observation(self):
        """Get current state observation"""
        row = self.df.iloc[self.current_step]
        
        # Extract features
        obs = []
        for feature in self.feature_columns:
            value = row[feature]
            # Handle any remaining NaN
            if pd.isna(value):
                value = 0.0
            obs.append(value)
        
        # Add position info
        obs.append(float(self.position))
        
        # Add normalized P&L
        obs.append(self.total_pnl / self.initial_balance)
        
        return np.array(obs, dtype=np.float32)
    
    def step(self, action):
        """Execute one time step"""
        
        # Get current price and return
        current_return = self.df.iloc[self.current_step]['jpy_normalized_return']
        
        # Map action to position: 0→-1 (short), 1→0 (hold), 2→1 (long)
        new_position = action - 1
        
        # Calculate reward from current position
        reward = 0
        if self.position != 0:
            # P&L from holding position
            position_pnl = self.position * current_return * self.balance
            reward += position_pnl
            self.total_pnl += position_pnl
        
        # Handle position changes (transaction costs)
        if new_position != self.position:
            # Transaction cost proportional to position size change
            cost = abs(new_position - self.position) * self.transaction_cost * self.balance
            reward -= cost
            self.total_pnl -= cost
            self.total_trades += 1
            
            self.trade_history.append({
                'step': self.current_step,
                'date': self.df.iloc[self.current_step]['date'],
                'old_position': self.position,
                'new_position': new_position,
                'return': current_return,
                'pnl': position_pnl if self.position != 0 else 0
            })
        
        # Update position
        self.position = new_position
        
        # Move to next step
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        
        # Get next observation
        obs = self._get_observation() if not done else np.zeros(self.observation_space.shape)
        
        # Info dict
        info = {
            'total_pnl': self.total_pnl,
            'position': self.position,
            'total_trades': self.total_trades,
            'balance': self.balance + self.total_pnl
        }
        
        return obs, reward, done, info
    
    def render(self, mode='human'):
        """Render current state"""
        current_value = self.balance + self.total_pnl
        roi = (current_value / self.initial_balance - 1) * 100
        
        print(f"Step: {self.current_step}/{len(self.df)}")
        print(f"Date: {self.df.iloc[self.current_step]['date']}")
        print(f"Position: {self.position:+d} | P&L: ${self.total_pnl:+.2f} ({roi:+.2f}%)")
        print(f"Trades: {self.total_trades}")
        print("-" * 50)
    
    def get_trade_history(self):
        """Return trade history as DataFrame"""
        return pd.DataFrame(self.trade_history)

# Test the environment
if __name__ == "__main__":
    print("Testing JPYUSDTradingEnv...")
    
    # Load sample data
    df = pd.read_csv('training_data.csv')
    df['date'] = pd.to_datetime(df['date'])
    
    # Create environment
    env = JPYUSDTradingEnv(df)
    
    # Run random episode
    obs = env.reset()
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Initial observation shape: {obs.shape}")
    
    for i in range(100):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        
        if i % 20 == 0:
            env.render()
        
        if done:
            break
    
    print("\n✓ Environment test complete!")
    print(f"Final P&L: ${info['total_pnl']:.2f}")