"""
Configuration file for forex trading agent
Stores all hyperparameters and file paths in one place
"""

# File paths
NEWS_ARTICLES_PATH = 'news_articles.csv'  # Your news data
CURRENCY_DATA_PATH = 'currency_data.csv'  # Your forex data

# Output paths
SENTIMENT_OUTPUT = 'news_with_sentiment.csv'
NORMALIZED_CURRENCY_OUTPUT = 'currency_normalized.csv'
MERGED_DATA_OUTPUT = 'training_data.csv'
MODEL_SAVE_PATH = 'models/jpyusd_sentiment_agent'

# Currency pairs in your data
TARGET_PAIR = 'USDJPY'  # The pair you're trading
REFERENCE_PAIRS = ['EURUSD', 'GBPUSD', 'AUDUSD', 'USDCAD', 'USDCHF']  # For normalization

# Training parameters
INITIAL_BALANCE = 10000
TRANSACTION_COST = 0.0001  # 1 pip spread
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# RL hyperparameters
LEARNING_RATE = 0.0003
TRAINING_TIMESTEPS = 50000
N_STEPS = 2048
BATCH_SIZE = 64