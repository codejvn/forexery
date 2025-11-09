"""
Merge daily sentiment with normalized currency data
Creates final training dataset with aligned timestamps
"""

import pandas as pd
import numpy as np
from config import SENTIMENT_OUTPUT, NORMALIZED_CURRENCY_OUTPUT, MERGED_DATA_OUTPUT

def merge_sentiment_and_currency(sentiment_df, currency_df):
    """
    Merge sentiment and currency data on date
    Add lag features to avoid look-ahead bias
    """
    
    # Merge on date
    merged = pd.merge(
        currency_df,
        sentiment_df,
        on='date',
        how='inner'  # Only keep dates where we have both
    )
    
    print(f"Merged dataset: {len(merged)} days")
    print(f"Date range: {merged['date'].min()} to {merged['date'].max()}")
    
    return merged

def add_lagged_sentiment_features(df):
    """
    Add lagged sentiment features to avoid look-ahead bias
    Use yesterday's sentiment to predict today's movement
    """
    
    # Lag sentiment by 1 day (use yesterday's sentiment)
    df['sentiment_lag1'] = df['sentiment_mean'].shift(1)
    df['sentiment_lag2'] = df['sentiment_mean'].shift(2)
    df['sentiment_lag3'] = df['sentiment_mean'].shift(3)
    
    # Rolling sentiment features
    df['sentiment_ma5'] = df['sentiment_mean'].shift(1).rolling(5).mean()
    df['sentiment_ma20'] = df['sentiment_mean'].shift(1).rolling(20).mean()
    
    # Sentiment momentum
    df['sentiment_change'] = df['sentiment_mean'].diff()
    df['sentiment_acceleration'] = df['sentiment_change'].diff()
    
    # Interaction: sentiment × volatility
    df['sentiment_x_volatility'] = df['sentiment_lag1'] * df['jpy_volatility']
    
    # Sentiment × article volume (higher volume = more confidence)
    df['sentiment_weighted'] = df['sentiment_lag1'] * np.log1p(df['article_count'])
    
    return df

def create_target_variable(df):
    """
    Create target: next day's return
    This is what the agent will try to predict/trade on
    """
    df['target_return'] = df['jpy_normalized_return'].shift(-1)
    
    return df

def main():
    print("="*50)
    print("STEP 3: Merging Data")
    print("="*50)
    
    # Load processed data
    print("Loading sentiment data...")
    sentiment_df = pd.read_csv(SENTIMENT_OUTPUT)
    sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
    
    print("Loading currency data...")
    currency_df = pd.read_csv(NORMALIZED_CURRENCY_OUTPUT)
    currency_df['date'] = pd.to_datetime(currency_df['date'])
    
    # Merge
    print("\nMerging datasets...")
    merged = merge_sentiment_and_currency(sentiment_df, currency_df)
    
    # Add lagged features
    print("Creating lagged sentiment features...")
    merged = add_lagged_sentiment_features(merged)
    
    # Create target
    print("Creating target variable...")
    merged = create_target_variable(merged)
    
    # Remove NaN from lagging/rolling
    merged_clean = merged.dropna()
    
    # Save
    merged_clean.to_csv(MERGED_DATA_OUTPUT, index=False)
    
    print(f"\n✓ Final dataset: {len(merged_clean)} days")
    print(f"✓ Features: {len(merged_clean.columns)} columns")
    print(f"✓ Date range: {merged_clean['date'].min()} to {merged_clean['date'].max()}")
    
    # Summary statistics
    print("\nKey statistics:")
    print(f"  Avg daily return: {merged_clean['jpy_normalized_return'].mean():.6f}")
    print(f"  Daily volatility: {merged_clean['jpy_normalized_return'].std():.6f}")
    print(f"  Avg sentiment: {merged_clean['sentiment_mean'].mean():.3f}")
    print(f"  Articles per day: {merged_clean['article_count'].mean():.1f}")
    
    print(f"\n✓ Saved to: {MERGED_DATA_OUTPUT}")

if __name__ == "__main__":
    main()