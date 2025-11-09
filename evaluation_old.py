"""
Evaluate trained RL agent on test data
Compares agent performance vs buy-and-hold baseline
Generates comprehensive metrics and visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
import os

from trading_environment import JPYUSDTradingEnv
from config import MODEL_SAVE_PATH, INITIAL_BALANCE, INITIAL_USD_RATIO, EPISODE_LENGTH

def load_test_data():
    """Load test dataset"""
    print("Loading test data...")
    df_test = pd.read_csv('test_data.csv')
    df_test['date'] = pd.to_datetime(df_test['date'])

    print(f"  Test data: {len(df_test)} days")
    print(f"  Date range: {df_test['date'].min()} to {df_test['date'].max()}")
    print(f"  Quarters: {len(df_test) // EPISODE_LENGTH}")

    return df_test

def evaluate_agent(model, env, n_episodes=None):
    """Evaluate agent performance"""
    print(f"\nEvaluating agent...")

    if n_episodes is None:
        n_episodes = len(env.df) // EPISODE_LENGTH

    all_episodes = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        episode_data = {
            'returns': [],
            'values': [],
            'actions': [],
            'usd_ratios': [],
            'dates': [],
            'trades': 0
        }

        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            episode_data['returns'].append(info['total_return'])
            episode_data['values'].append(info['total_value_usd'])
            episode_data['actions'].append(action[0])
            episode_data['usd_ratios'].append(info['usd_balance'] / info['total_value_usd'] if info['total_value_usd'] > 0 else 0.5)
            episode_data['dates'].append(info['date'])
            episode_data['trades'] = info['total_trades']

            done = terminated or truncated

        # Store episode summary
        if 'episode' in info:
            ep_stats = info['episode']
            all_episodes.append({
                'episode': ep,
                'total_return': ep_stats['total_return'],
                'sharpe_ratio': ep_stats['sharpe_ratio'],
                'max_drawdown': ep_stats['max_drawdown'],
                'total_trades': ep_stats['total_trades'],
                'final_value': ep_stats['final_value'],
                'mean_daily_return': ep_stats['mean_daily_return'],
                'std_daily_return': ep_stats['std_daily_return']
            })

            print(f"  Quarter {ep+1}: Return={ep_stats['total_return']*100:+.2f}%, Sharpe={ep_stats['sharpe_ratio']:.2f}, Trades={ep_stats['total_trades']}")

    return pd.DataFrame(all_episodes)

def calculate_buy_and_hold_baseline(df):
    """Calculate buy-and-hold performance (50% USD, 50% JPY)"""
    print("\nCalculating buy-and-hold baseline...")

    initial_usd = INITIAL_BALANCE * INITIAL_USD_RATIO
    initial_jpy = INITIAL_BALANCE * (1 - INITIAL_USD_RATIO)

    # Starting exchange rate
    start_rate = df.iloc[0]['USDJPY_Close']

    # Calculate value at each point
    values = []
    returns = []

    for idx, row in df.iterrows():
        current_rate = row['USDJPY_Close']
        total_value = initial_usd + (initial_jpy / current_rate)
        values.append(total_value)
        returns.append((total_value / INITIAL_BALANCE - 1.0))

    # Split into quarters for comparison
    quarters = []
    for i in range(0, len(df) - EPISODE_LENGTH, EPISODE_LENGTH):
        quarter_start = values[i]
        quarter_end = values[min(i + EPISODE_LENGTH, len(values) - 1)]
        quarter_return = (quarter_end / quarter_start - 1.0)

        # Calculate Sharpe for this quarter
        quarter_returns = [(values[j+1] / values[j] - 1.0) for j in range(i, min(i + EPISODE_LENGTH - 1, len(values) - 1))]
        if len(quarter_returns) > 1:
            sharpe = np.mean(quarter_returns) / (np.std(quarter_returns) + 1e-6) * np.sqrt(252)
        else:
            sharpe = 0.0

        quarters.append({
            'episode': len(quarters),
            'total_return': quarter_return,
            'sharpe_ratio': sharpe,
            'strategy': 'Buy-and-Hold'
        })

    final_return = (values[-1] / values[0] - 1.0)
    print(f"  Final return: {final_return*100:+.2f}%")
    print(f"  Final value: ${values[-1]:.2f}")

    return pd.DataFrame(quarters), values

def plot_performance_comparison(agent_results, baseline_results, baseline_values, df_test):
    """Create performance visualization"""
    print("\nGenerating performance plots...")

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('RL Agent vs Buy-and-Hold Performance', fontsize=16, fontweight='bold')
    
    # 1. Price chart with positions
    ax1 = plt.subplot(4, 2, 1)
    dates = results['dates']
    prices = results['prices']
    positions = results['positions']
    
    ax1.plot(dates, prices, label='USD/JPY Price', color='black', alpha=0.6)
    
    # Mark buy/long positions
    long_mask = [p == 1 for p in positions]
    ax1.scatter([d for d, m in zip(dates, long_mask) if m],
               [p for p, m in zip(prices, long_mask) if m],
               color='green', marker='^', s=50, label='Long', alpha=0.7)
    
    # Mark sell/short positions
    short_mask = [p == -1 for p in positions]
    ax1.scatter([d for d, m in zip(dates, short_mask) if m],
               [p for p, m in zip(prices, short_mask) if m],
               color='red', marker='v', s=50, label='Short', alpha=0.7)
    
    ax1.set_title('Trading Actions on Price Chart', fontsize=12, fontweight='bold')
    ax1.set_ylabel('USD/JPY Price')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Cumulative P&L
    ax2 = plt.subplot(4, 2, 2)
    ax2.plot(dates, results['pnl'], label='Cumulative P&L', color='blue', linewidth=2)
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax2.fill_between(dates, 0, results['pnl'], alpha=0.3)
    ax2.set_title('Cumulative Profit/Loss', fontsize=12, fontweight='bold')
    ax2.set_ylabel('P&L ($)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Sentiment vs Returns
    ax3 = plt.subplot(4, 2, 3)
    ax3.scatter(results['sentiment'], results['returns'], alpha=0.5, s=10)
    ax3.axhline(y=0, color='red', linestyle='--', alpha=0.3)
    ax3.axvline(x=0, color='red', linestyle='--', alpha=0.3)
    ax3.set_title('Sentiment vs Returns', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Sentiment Score')
    ax3.set_ylabel('Normalized Return')
    ax3.grid(True, alpha=0.3)
    
    # 4. Position distribution
    ax4 = plt.subplot(4, 2, 4)
    position_counts = pd.Series(positions).value_counts().sort_index()
    colors = ['red', 'gray', 'green']
    ax4.bar(position_counts.index, position_counts.values, color=colors)
    ax4.set_title('Position Distribution', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Position (-1=Short, 0=Neutral, 1=Long)')
    ax4.set_ylabel('Count')
    ax4.set_xticks([-1, 0, 1])
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Reward distribution
    ax5 = plt.subplot(4, 2, 5)
    ax5.hist(results['rewards'], bins=50, alpha=0.7, edgecolor='black')
    ax5.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax5.set_title('Reward Distribution', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Reward')
    ax5.set_ylabel('Frequency')
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Sentiment over time
    ax6 = plt.subplot(4, 2, 6)
    ax6.plot(dates, results['sentiment'], label='Sentiment', color='purple', alpha=0.7)
    ax6.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax6.fill_between(dates, 0, results['sentiment'], alpha=0.3)
    ax6.set_title('Sentiment Over Time', fontsize=12, fontweight='bold')
    ax6.set_ylabel('Sentiment Score')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 7. Metrics table
    ax7 = plt.subplot(4, 2, 7)
    ax7.axis('off')
    metrics_text = '\n'.join([f'{k}: {v:.2f}' for k, v in metrics.items()])
    ax7.text(0.1, 0.5, f'Performance Metrics\n{"="*25}\n{metrics_text}',
            fontsize=11, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 8. Rolling Sharpe
    ax8 = plt.subplot(4, 2, 8)
    window = 30
    returns_series = pd.Series(np.diff(results['pnl'], prepend=0))
    rolling_sharpe = returns_series.rolling(window).mean() / (returns_series.rolling(window).std() + 1e-10) * np.sqrt(252)
    ax8.plot(dates, rolling_sharpe, label=f'{window}-day Rolling Sharpe', color='darkgreen')
    ax8.axhline(y=0, color='red', linestyle='--', alpha=0.3)
    ax8.set_title('Rolling Sharpe Ratio', fontsize=12, fontweight='bold')
    ax8.set_ylabel('Sharpe Ratio')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('trading_results.png', dpi=300, bbox_inches='tight')
    print("✓ Saved visualization to trading_results.png")
    
    return fig

def compare_to_baseline(results, df):
    """Compare agent to buy-and-hold baseline"""
    
    # Buy-and-hold return
    initial_price = df.iloc[0]['USDJPY']
    final_price = df.iloc[-1]['USDJPY']
    bh_return = ((final_price - initial_price) / initial_price) * 100
    
    # Agent return
    agent_return = (results['pnl'][-1] / 10000) * 100
    
    print("\n" + "="*50)
    print("Comparison to Buy-and-Hold")
    print("="*50)
    print(f"Buy-and-Hold Return: {bh_return:+.2f}%")
    print(f"Agent Return:        {agent_return:+.2f}%")
    print(f"Outperformance:      {agent_return - bh_return:+.2f}%")
    print("="*50)

def main():
    print("="*50)
    print("STEP 6: Evaluating Agent")
    print("="*50)
    
    # Load model
    print("\nLoading trained model...")
    model = PPO.load(MODEL_SAVE_PATH)
    
    # Load test data
    print("Loading test data...")
    df_test = load_test_data()
    
    # Evaluate
    print("Running evaluation...")
    results, trade_history, final_info = evaluate_agent(model, df_test)
    
    # Calculate metrics
    print("\nCalculating metrics...")
    metrics = calculate_metrics(results)
    
    # Print metrics
    print("\n" + "="*50)
    print("PERFORMANCE METRICS")
    print("="*50)
    for key, value in metrics.items():
        print(f"{key:.<30} {value:>10.2f}")
    print("="*50)
    
    # Compare to baseline
    compare_to_baseline(results, df_test)
    
    # Create visualizations
    print("\nCreating visualizations...")
    fig = create_visualizations(results, trade_history, metrics)
    
    # Save trade history
    if len(trade_history) > 0:
        trade_history.to_csv('trade_history.csv', index=False)
        print(f"✓ Saved {len(trade_history)} trades to trade_history.csv")
    
    print("\n" + "="*50)
    print("Evaluation complete!")
    print("="*50)

if __name__ == "__main__":
    main()
    