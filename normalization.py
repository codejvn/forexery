# """
# Normalize USD/JPY movements using other currency pairs
# This helps distinguish between:
#   - General USD strength/weakness (affects all USD pairs)
#   - JPY-specific movements

# Method: Calculate USD index and JPY index from other pairs
# """

# import pandas as pd
# import numpy as np
# from config import CURRENCY_DATA_PATH, NORMALIZED_CURRENCY_OUTPUT, TARGET_PAIR, REFERENCE_PAIRS

# def load_currency_data(filepath):
#     """Load and validate currency data"""
#     df = pd.read_csv(filepath)
#     print(f"Loaded currency data: {len(df)} rows")
    
#     # Ensure date column
#     if 'date' not in df.columns and 'Date' not in df.columns:
#         raise ValueError("CSV must have 'date' or 'Date' column")
    
#     date_col = 'date' if 'date' in df.columns else 'Date'
#     df = df.rename(columns={date_col: 'date'})
#     df['date'] = pd.to_datetime(df['date'])
    
#     print(f"Available columns: {df.columns.tolist()}")
#     print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    
#     return df

# def calculate_returns(df, pairs):
#     """Calculate daily returns for all currency pairs"""
#     for pair in pairs:
#         if pair in df.columns:
#             # Returns = (price_today - price_yesterday) / price_yesterday
#             df[f'{pair}_return'] = df[pair].pct_change()
#         else:
#             print(f"Warning: {pair} not found in data")
    
#     return df

# def create_usd_index(df, usd_pairs):
#     """
#     Create USD strength index from USD-based pairs
#     Average movement of USD against multiple currencies
    
#     For pairs like EUR/USD, GBP/USD: positive return = USD weakened
#     So we invert them: USD index = -mean(EURUSD_return, GBPUSD_return, ...)
#     """
#     usd_returns = []
    
#     for pair in usd_pairs:
#         return_col = f'{pair}_return'
#         if return_col in df.columns:
#             # Invert: when EUR/USD goes up, USD got weaker
#             usd_returns.append(-df[return_col])
    
#     if usd_returns:
#         df['usd_index_return'] = pd.concat(usd_returns, axis=1).mean(axis=1)
#     else:
#         df['usd_index_return'] = 0
    
#     return df

# def normalize_usdjpy(df):
#     """
#     Normalize USD/JPY returns by removing general USD strength
    
#     Normalized JPY return = USDJPY_return - USD_index_return
    
#     Interpretation:
#     - If normalized > 0: JPY specifically weakened (not just USD strength)
#     - If normalized < 0: JPY specifically strengthened (not just USD weakness)
#     """
    
#     if 'USDJPY_return' in df.columns and 'usd_index_return' in df.columns:
#         # USD/JPY return minus general USD movement = JPY-specific movement
#         df['jpy_normalized_return'] = df['USDJPY_return'] - df['usd_index_return']
        
#         # Also keep the raw return for comparison
#         df['jpy_raw_return'] = df['USDJPY_return']
#     else:
#         print("Warning: Could not normalize USD/JPY")
#         df['jpy_normalized_return'] = df.get('USDJPY_return', 0)
#         df['jpy_raw_return'] = df.get('USDJPY_return', 0)
    
#     return df

# def add_technical_features(df):
#     """Add technical indicators for USD/JPY"""
    
#     # Volatility (20-day rolling std of returns)
#     df['jpy_volatility'] = df['jpy_raw_return'].rolling(20).std()
    
#     # Moving averages of price
#     if TARGET_PAIR in df.columns:
#         df['jpy_sma_5'] = df[TARGET_PAIR].rolling(5).mean()
#         df['jpy_sma_20'] = df[TARGET_PAIR].rolling(20).mean()
#         df['jpy_sma_50'] = df[TARGET_PAIR].rolling(50).mean()
        
#         # Price relative to moving averages
#         df['jpy_price_vs_sma5'] = (df[TARGET_PAIR] / df['jpy_sma_5']) - 1
#         df['jpy_price_vs_sma20'] = (df[TARGET_PAIR] / df['jpy_sma_20']) - 1
    
#     # RSI (Relative Strength Index)
#     delta = df['jpy_raw_return'].fillna(0)
#     gain = delta.where(delta > 0, 0).rolling(14).mean()
#     loss = -delta.where(delta < 0, 0).rolling(14).mean()
#     rs = gain / (loss + 1e-10)
#     df['jpy_rsi'] = 100 - (100 / (1 + rs))
    
#     # Momentum (5-day and 20-day)
#     if TARGET_PAIR in df.columns:
#         df['jpy_momentum_5'] = df[TARGET_PAIR] / df[TARGET_PAIR].shift(5) - 1
#         df['jpy_momentum_20'] = df[TARGET_PAIR] / df[TARGET_PAIR].shift(20) - 1
    
#     return df

# def main():
#     print("="*50)
#     print("STEP 2: Normalizing Currency Data")
#     print("="*50)
    
#     # Load data
#     df = load_currency_data(CURRENCY_DATA_PATH)
    
#     # Calculate returns
#     all_pairs = [TARGET_PAIR] + REFERENCE_PAIRS
#     df = calculate_returns(df, all_pairs)
    
#     # Create USD index from reference pairs
#     print(f"\nCreating USD index from: {REFERENCE_PAIRS}")
#     df = create_usd_index(df, REFERENCE_PAIRS)
    
#     # Normalize USD/JPY
#     print(f"Normalizing {TARGET_PAIR} movements...")
#     df = normalize_usdjpy(df)
    
#     # Add technical features
#     print("Adding technical indicators...")
#     df = add_technical_features(df)
    
#     # Remove NaN rows from rolling calculations
#     df_clean = df.dropna()
    
#     # Save
#     output_cols = [
#         'date', TARGET_PAIR,
#         'jpy_raw_return', 'jpy_normalized_return', 'usd_index_return',
#         'jpy_volatility', 'jpy_sma_5', 'jpy_sma_20', 'jpy_sma_50',
#         'jpy_price_vs_sma5', 'jpy_price_vs_sma20',
#         'jpy_rsi', 'jpy_momentum_5', 'jpy_momentum_20'
#     ]
    
#     # Only keep columns that exist
#     output_cols = [col for col in output_cols if col in df_clean.columns]
    
#     df_clean[output_cols].to_csv(NORMALIZED_CURRENCY_OUTPUT, index=False)
    
#     print(f"\n✓ Processed {len(df_clean)} days of currency data")
#     print(f"✓ Date range: {df_clean['date'].min()} to {df_clean['date'].max()}")
    
#     # Show normalization effect
#     raw_std = df_clean['jpy_raw_return'].std()
#     norm_std = df_clean['jpy_normalized_return'].std()
#     print(f"\nNormalization impact:")
#     print(f"  Raw USD/JPY volatility: {raw_std:.6f}")
#     print(f"  Normalized JPY volatility: {norm_std:.6f}")
#     print(f"  USD index volatility: {df_clean['usd_index_return'].std():.6f}")
    
#     print(f"\n✓ Saved to: {NORMALIZED_CURRENCY_OUTPUT}")

# if __name__ == "__main__":
#     main()