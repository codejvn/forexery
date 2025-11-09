"""
Data Merger for RL Trading Agent
Merges currency data with news sentiment from USA and Japan
Creates historical sentiment features with lookback windows
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def load_news_data(usa_path, japan_path):
    """Load and process news sentiment data"""
    print("Loading news data...")

    # Load USA news (handle commas in headlines)
    df_usa = pd.read_csv(usa_path, on_bad_lines='skip', encoding='utf-8')
    df_usa.columns = ['Date', 'Headline', 'Sentiment']
    df_usa['Date'] = pd.to_datetime(df_usa['Date'])

    # Load Japan news (handle commas in headlines)
    df_japan = pd.read_csv(japan_path, on_bad_lines='skip', encoding='utf-8')
    df_japan.columns = ['Date', 'Headline', 'Sentiment']
    df_japan['Date'] = pd.to_datetime(df_japan['Date'])

    print(f"  USA news: {len(df_usa)} articles from {df_usa['Date'].min()} to {df_usa['Date'].max()}")
    print(f"  Japan news: {len(df_japan)} articles from {df_japan['Date'].min()} to {df_japan['Date'].max()}")

    # Convert sentiment to numeric: Positive=1, Neutral=0, Negative=-1
    sentiment_map = {'Positive': 1, 'Neutral': 0, 'Negative': -1}
    df_usa['sentiment_score'] = df_usa['Sentiment'].map(sentiment_map)
    df_japan['sentiment_score'] = df_japan['Sentiment'].map(sentiment_map)

    # Handle any unmapped values
    df_usa['sentiment_score'].fillna(0, inplace=True)
    df_japan['sentiment_score'].fillna(0, inplace=True)

    return df_usa, df_japan

def aggregate_daily_sentiment(df):
    """Aggregate multiple articles per day"""
    # Group by date and calculate mean sentiment
    daily = df.groupby('Date').agg({
        'sentiment_score': ['mean', 'std', 'count']
    }).reset_index()

    daily.columns = ['Date', 'sentiment_mean', 'sentiment_std', 'article_count']
    daily['sentiment_std'].fillna(0, inplace=True)  # Single article days

    return daily

def create_historical_features(df, max_lookback=30):
    """
    Create historical sentiment features for each day
    Agent will have access to all news from before current date
    """
    print(f"Creating historical sentiment features (lookback={max_lookback} days)...")

    df = df.sort_values('Date').reset_index(drop=True)

    # Rolling averages with different windows
    df['sentiment_ma3'] = df['sentiment_mean'].rolling(window=3, min_periods=1).mean()
    df['sentiment_ma5'] = df['sentiment_mean'].rolling(window=5, min_periods=1).mean()
    df['sentiment_ma7'] = df['sentiment_mean'].rolling(window=7, min_periods=1).mean()
    df['sentiment_ma10'] = df['sentiment_mean'].rolling(window=10, min_periods=1).mean()
    df['sentiment_ma20'] = df['sentiment_mean'].rolling(window=20, min_periods=1).mean()

    # Rolling standard deviation (sentiment volatility)
    df['sentiment_volatility_7d'] = df['sentiment_mean'].rolling(window=7, min_periods=1).std()
    df['sentiment_volatility_20d'] = df['sentiment_mean'].rolling(window=20, min_periods=1).std()

    # Momentum (rate of change)
    df['sentiment_momentum_3d'] = df['sentiment_mean'] - df['sentiment_mean'].shift(3)
    df['sentiment_momentum_5d'] = df['sentiment_mean'] - df['sentiment_mean'].shift(5)
    df['sentiment_momentum_10d'] = df['sentiment_mean'] - df['sentiment_mean'].shift(10)

    # Trend strength (current vs longer-term average)
    df['sentiment_trend_short'] = df['sentiment_ma5'] - df['sentiment_ma20']
    df['sentiment_trend_medium'] = df['sentiment_ma10'] - df['sentiment_ma20']

    # Exponentially weighted moving average (recent days weighted more)
    df['sentiment_ema5'] = df['sentiment_mean'].ewm(span=5, adjust=False).mean()
    df['sentiment_ema20'] = df['sentiment_mean'].ewm(span=20, adjust=False).mean()

    # Fill NaN values from shifts
    df.fillna(method='bfill', inplace=True)
    df.fillna(0, inplace=True)

    return df

def load_currency_data(currency_path):
    """Load currency data with technical indicators"""
    print("Loading currency data...")

    df = pd.read_csv(currency_path)
    df['date'] = pd.to_datetime(df['date'])

    print(f"  Currency data: {len(df)} days from {df['date'].min()} to {df['date'].max()}")

    return df

def merge_data(df_currency, df_usa_sentiment, df_japan_sentiment):
    """Merge currency data with sentiment features"""
    print("Merging datasets...")

    # Rename date columns for merge
    df_usa_sentiment = df_usa_sentiment.rename(columns={'Date': 'date'})
    df_japan_sentiment = df_japan_sentiment.rename(columns={'Date': 'date'})

    # Add prefixes to distinguish USA vs Japan features
    usa_cols = {col: f'usa_{col}' for col in df_usa_sentiment.columns if col != 'date'}
    japan_cols = {col: f'japan_{col}' for col in df_japan_sentiment.columns if col != 'date'}

    df_usa_sentiment = df_usa_sentiment.rename(columns=usa_cols)
    df_japan_sentiment = df_japan_sentiment.rename(columns=japan_cols)

    # Merge with currency data
    df_merged = df_currency.merge(df_usa_sentiment, on='date', how='left')
    df_merged = df_merged.merge(df_japan_sentiment, on='date', how='left')

    # Forward fill sentiment for days without news (weekends, holidays)
    sentiment_cols = [col for col in df_merged.columns if 'sentiment' in col or 'article' in col]
    df_merged[sentiment_cols] = df_merged[sentiment_cols].fillna(method='ffill')
    df_merged[sentiment_cols] = df_merged[sentiment_cols].fillna(0)  # Fill remaining NaN at start

    print(f"  Merged data: {len(df_merged)} days")
    print(f"  Missing values: {df_merged.isnull().sum().sum()}")

    return df_merged

def select_features(df):
    """Select only relevant features for USDJPY trading"""
    print("Selecting features for USDJPY trading...")

    # USDJPY price and technical indicators
    usdjpy_features = [
        'USDJPY_Close',
        'USDJPY_return',
        'USDJPY_sma_5',
        'USDJPY_sma_20',
        'USDJPY_sma_50',
        'USDJPY_ema_20',
        'USDJPY_macd_line',
        'USDJPY_macd_signal',
        'USDJPY_macd_hist',
        'USDJPY_rsi_14',
        'USDJPY_roc_5',
        'USDJPY_roc_20',
        'USDJPY_price_vs_sma20',
        'USDJPY_atr_14',
        'USDJPY_stoch_k',
        'USDJPY_stoch_d',
    ]

    # JPYUSD (inverse pair) features
    jpyusd_features = [
        'JPYUSD_Close',
        'JPYUSD_return',
        'JPYUSD_rsi_14',
        'JPYUSD_macd_hist',
    ]

    # Related currency pairs for context
    related_features = [
        'EURUSD_Close',
        'EURUSD_return',
        'GBPUSD_Close',
        'GBPUSD_return',
        'AUDUSD_Close',
        'AUDUSD_return',
    ]

    # USA sentiment features
    usa_sentiment_features = [col for col in df.columns if 'usa_sentiment' in col]
    usa_sentiment_features.append('usa_article_count')

    # Japan sentiment features
    japan_sentiment_features = [col for col in df.columns if 'japan_sentiment' in col]
    japan_sentiment_features.append('japan_article_count')

    # Combine all features
    required_features = ['date'] + usdjpy_features + jpyusd_features + related_features + \
                       usa_sentiment_features + japan_sentiment_features

    # Filter to only existing features
    available_features = [f for f in required_features if f in df.columns]
    missing_features = [f for f in required_features if f not in df.columns]

    if missing_features:
        print(f"  Warning: Missing {len(missing_features)} features:")
        for f in missing_features[:10]:  # Show first 10
            print(f"    - {f}")

    df_selected = df[available_features].copy()

    print(f"  Selected {len(available_features)} features")

    return df_selected

def add_derived_features(df):
    """Add additional derived features"""
    print("Adding derived features...")

    # Sentiment divergence (USA vs Japan)
    if 'usa_sentiment_mean' in df.columns and 'japan_sentiment_mean' in df.columns:
        df['sentiment_divergence'] = df['usa_sentiment_mean'] - df['japan_sentiment_mean']
        df['sentiment_divergence_ma5'] = df['sentiment_divergence'].rolling(window=5, min_periods=1).mean()

    # Combined sentiment score (weighted average)
    if 'usa_sentiment_mean' in df.columns and 'japan_sentiment_mean' in df.columns:
        df['sentiment_combined'] = (df['usa_sentiment_mean'] + df['japan_sentiment_mean']) / 2
        df['sentiment_combined_ma7'] = df['sentiment_combined'].rolling(window=7, min_periods=1).mean()

    # Volatility ratio
    if 'USDJPY_atr_14' in df.columns:
        df['volatility_percentile'] = df['USDJPY_atr_14'].rolling(window=60, min_periods=20).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1]
        )

    # Price momentum
    if 'USDJPY_Close' in df.columns:
        df['price_momentum_20d'] = df['USDJPY_Close'].pct_change(20)
        df['price_momentum_60d'] = df['USDJPY_Close'].pct_change(60)

    df.fillna(method='bfill', inplace=True)
    df.fillna(0, inplace=True)

    print(f"  Total features: {len(df.columns) - 1}")  # -1 for date column

    return df

def main():
    """Main data processing pipeline"""
    print("="*60)
    print("DATA MERGER FOR RL TRADING AGENT")
    print("="*60)

    # File paths
    USA_NEWS_PATH = 'news_articles_usa.csv'
    JAPAN_NEWS_PATH = 'news_articles_japan.csv'
    CURRENCY_PATH = 'all_currencies_with_indicators.csv'
    OUTPUT_PATH = 'merged_trading_data.csv'

    # Step 1: Load news data
    df_usa, df_japan = load_news_data(USA_NEWS_PATH, JAPAN_NEWS_PATH)

    # Step 2: Aggregate daily sentiment
    df_usa_daily = aggregate_daily_sentiment(df_usa)
    df_japan_daily = aggregate_daily_sentiment(df_japan)

    # Step 3: Create historical features
    df_usa_features = create_historical_features(df_usa_daily, max_lookback=30)
    df_japan_features = create_historical_features(df_japan_daily, max_lookback=30)

    # Step 4: Load currency data
    df_currency = load_currency_data(CURRENCY_PATH)

    # Step 5: Merge all data
    df_merged = merge_data(df_currency, df_usa_features, df_japan_features)

    # Step 6: Select relevant features
    df_final = select_features(df_merged)

    # Step 7: Add derived features
    df_final = add_derived_features(df_final)

    # Step 8: Remove rows with NaN (beginning of time series)
    initial_rows = len(df_final)
    df_final = df_final.dropna()
    removed_rows = initial_rows - len(df_final)
    print(f"\nRemoved {removed_rows} rows with NaN values")

    # Step 9: Save to CSV
    df_final.to_csv(OUTPUT_PATH, index=False)
    print(f"\n[OK] Data saved to {OUTPUT_PATH}")
    print(f"  Final dataset: {len(df_final)} days, {len(df_final.columns)} features")
    print(f"  Date range: {df_final['date'].min()} to {df_final['date'].max()}")

    # Display sample
    print("\nSample features:")
    print(df_final.head(3))

    print("\n" + "="*60)
    print("DATA MERGER COMPLETE!")
    print("="*60)

    return df_final

if __name__ == "__main__":
    df = main()
