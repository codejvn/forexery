"""
Configuration file for forex trading agent
Stores all hyperparameters and file paths in one place
"""

# File paths
USA_NEWS_PATH = 'news_articles_usa.csv'
JAPAN_NEWS_PATH = 'news_articles_japan.csv'
CURRENCY_DATA_PATH = 'all_currencies_with_indicators.csv'

# Output paths
MERGED_DATA_OUTPUT = 'merged_trading_data.csv'
MODEL_SAVE_PATH = 'models/jpyusd_sentiment_agent'

# Currency pairs in your data
TARGET_PAIR = 'USDJPY'  # The pair you're trading
REFERENCE_PAIRS = ['EURUSD', 'GBPUSD', 'AUDUSD']  # For context

# Portfolio parameters
INITIAL_BALANCE = 10000
INITIAL_USD_RATIO = 0.5  # Start with 50% USD
INITIAL_JPY_RATIO = 0.5  # Start with 50% JPY
TRANSACTION_COST = 0.0002  # 2 pip spread (0.02% per trade) - increased from 0.0001
OVERTRADING_PENALTY = 1.0  # Drastically increased from 0.01 (was too weak)

# Episode structure
DAYS_PER_QUARTER = 63  # Trading days in a quarter (~252 trading days/year ÷ 4)
EPISODE_LENGTH = DAYS_PER_QUARTER  # One quarter per episode

# Reward structure
DAILY_REWARD_WEIGHT = 1.0  # Increased from 0.1 - make daily P&L more important
QUARTERLY_REWARD_WEIGHT = 100.0  # Reduced from 200.0 - was overshadowing daily signals
SHARPE_BONUS_WEIGHT = 10.0  # Reduced from 20.0 - simplify reward

# Data splits
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# RL hyperparameters
LEARNING_RATE = 0.0003  # Increased back to 0.0003 for faster learning
TRAINING_TIMESTEPS = 300000  # Reduced from 500k - maybe overtrained
N_STEPS = 2048
BATCH_SIZE = 64
GAMMA = 0.99  # Back to 0.99 (was 0.995)
GAE_LAMBDA = 0.95  # Generalized Advantage Estimation
CLIP_RANGE = 0.2  # PPO clipping parameter
ENT_COEF = 0.001  # DRASTICALLY reduced from 0.05 - too much random exploration causing bad trades

# Action space bounds
ACTION_LOW = -1.0  # Maximum sell proportion
ACTION_HIGH = 1.0  # Maximum buy proportion