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
os.environ['KMP_DUPLICATE_LIB_OK']='True'  # For Mac M1 compatibility

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

    # 1. Quarterly returns comparison
    ax1 = axes[0, 0]
    x = np.arange(len(agent_results))
    width = 0.35

    ax1.bar(x - width/2, agent_results['total_return'] * 100, width, label='RL Agent', alpha=0.8, color='steelblue')
    ax1.bar(x + width/2, baseline_results['total_return'] * 100, width, label='Buy-and-Hold', alpha=0.8, color='coral')
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.set_xlabel('Quarter')
    ax1.set_ylabel('Return (%)')
    ax1.set_title('Quarterly Returns')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Cumulative returns
    ax2 = axes[0, 1]
    agent_cumulative = (1 + agent_results['total_return']).cumprod()
    baseline_cumulative = (1 + baseline_results['total_return']).cumprod()

    ax2.plot(agent_cumulative.values, label='RL Agent', linewidth=2, color='steelblue')
    ax2.plot(baseline_cumulative.values, label='Buy-and-Hold', linewidth=2, color='coral')
    ax2.axhline(y=1, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
    ax2.set_xlabel('Quarter')
    ax2.set_ylabel('Cumulative Return (Multiple)')
    ax2.set_title('Cumulative Performance')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Sharpe ratio comparison
    ax3 = axes[1, 0]
    ax3.bar(x - width/2, agent_results['sharpe_ratio'], width, label='RL Agent', alpha=0.8, color='steelblue')
    ax3.bar(x + width/2, baseline_results['sharpe_ratio'], width, label='Buy-and-Hold', alpha=0.8, color='coral')
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax3.set_xlabel('Quarter')
    ax3.set_ylabel('Sharpe Ratio')
    ax3.set_title('Risk-Adjusted Returns (Sharpe Ratio)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Trade frequency
    ax4 = axes[1, 1]
    ax4.bar(x, agent_results['total_trades'], alpha=0.8, color='steelblue')
    ax4.set_xlabel('Quarter')
    ax4.set_ylabel('Number of Trades')
    ax4.set_title('Trading Frequency per Quarter')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
    print("  Saved: performance_comparison.png")

    plt.close()

def generate_report(agent_results, baseline_results):
    """Generate comprehensive performance report"""
    print("\n" + "="*70)
    print("PERFORMANCE REPORT")
    print("="*70)

    # Agent statistics
    agent_total_return = (1 + agent_results['total_return']).prod() - 1
    agent_mean_return = agent_results['total_return'].mean()
    agent_std_return = agent_results['total_return'].std()
    agent_mean_sharpe = agent_results['sharpe_ratio'].mean()
    agent_max_drawdown = agent_results['max_drawdown'].min()
    agent_total_trades = agent_results['total_trades'].sum()
    agent_win_rate = (agent_results['total_return'] > 0).mean()

    # Baseline statistics
    baseline_total_return = (1 + baseline_results['total_return']).prod() - 1
    baseline_mean_return = baseline_results['total_return'].mean()
    baseline_std_return = baseline_results['total_return'].std()
    baseline_mean_sharpe = baseline_results['sharpe_ratio'].mean()

    print("\nRL AGENT PERFORMANCE")
    print("-" * 70)
    print(f"  Total Return:          {agent_total_return*100:+.2f}%")
    print(f"  Mean Quarterly Return: {agent_mean_return*100:+.2f}%")
    print(f"  Std Quarterly Return:  {agent_std_return*100:.2f}%")
    print(f"  Mean Sharpe Ratio:     {agent_mean_sharpe:.2f}")
    print(f"  Max Drawdown:          {agent_max_drawdown*100:.2f}%")
    print(f"  Win Rate:              {agent_win_rate*100:.1f}%")
    print(f"  Total Trades:          {agent_total_trades:.0f}")
    print(f"  Final Value:           ${agent_results['final_value'].iloc[-1]:.2f}")

    print("\nBUY-AND-HOLD BASELINE")
    print("-" * 70)
    print(f"  Total Return:          {baseline_total_return*100:+.2f}%")
    print(f"  Mean Quarterly Return: {baseline_mean_return*100:+.2f}%")
    print(f"  Std Quarterly Return:  {baseline_std_return*100:.2f}%")
    print(f"  Mean Sharpe Ratio:     {baseline_mean_sharpe:.2f}")

    print("\nCOMPARISON (Agent vs Baseline)")
    print("-" * 70)
    outperformance = agent_total_return - baseline_total_return
    sharpe_improvement = agent_mean_sharpe - baseline_mean_sharpe

    print(f"  Return Difference:     {outperformance*100:+.2f}%")
    print(f"  Sharpe Improvement:    {sharpe_improvement:+.2f}")

    if outperformance > 0:
        print(f"\nAgent OUTPERFORMED buy-and-hold by {outperformance*100:.2f}%")
    else:
        print(f"\nAgent UNDERPERFORMED buy-and-hold by {abs(outperformance)*100:.2f}%")

    print("\n" + "="*70)

    # Save report to file
    with open('performance_report.txt', 'w') as f:
        f.write("="*70 + "\n")
        f.write("PERFORMANCE REPORT\n")
        f.write("="*70 + "\n\n")

        f.write("RL AGENT PERFORMANCE\n")
        f.write("-" * 70 + "\n")
        f.write(f"Total Return:          {agent_total_return*100:+.2f}%\n")
        f.write(f"Mean Quarterly Return: {agent_mean_return*100:+.2f}%\n")
        f.write(f"Std Quarterly Return:  {agent_std_return*100:.2f}%\n")
        f.write(f"Mean Sharpe Ratio:     {agent_mean_sharpe:.2f}\n")
        f.write(f"Max Drawdown:          {agent_max_drawdown*100:.2f}%\n")
        f.write(f"Win Rate:              {agent_win_rate*100:.1f}%\n")
        f.write(f"Total Trades:          {agent_total_trades:.0f}\n")
        f.write(f"Final Value:           ${agent_results['final_value'].iloc[-1]:.2f}\n\n")

        f.write("BUY-AND-HOLD BASELINE\n")
        f.write("-" * 70 + "\n")
        f.write(f"Total Return:          {baseline_total_return*100:+.2f}%\n")
        f.write(f"Mean Quarterly Return: {baseline_mean_return*100:+.2f}%\n")
        f.write(f"Std Quarterly Return:  {baseline_std_return*100:.2f}%\n")
        f.write(f"Mean Sharpe Ratio:     {baseline_mean_sharpe:.2f}\n\n")

        f.write("COMPARISON (Agent vs Baseline)\n")
        f.write("-" * 70 + "\n")
        f.write(f"Return Difference:     {outperformance*100:+.2f}%\n")
        f.write(f"Sharpe Improvement:    {sharpe_improvement:+.2f}\n")

    print("Report saved to: performance_report.txt")

def main():
    print("="*70)
    print("EVALUATING RL AGENT ON TEST DATA")
    print("="*70)

    # Load test data
    df_test = load_test_data()

    # Load trained model
    print(f"\nLoading model from {MODEL_SAVE_PATH}...")
    try:
        model = PPO.load(MODEL_SAVE_PATH)
        print("Model loaded successfully")
    except FileNotFoundError:
        print(f"Error: Model not found at {MODEL_SAVE_PATH}")
        print("Please train the model first by running training.py")
        return

    # Create test environment
    print("\nCreating test environment...")
    env = JPYUSDTradingEnv(df_test, episode_length=EPISODE_LENGTH)

    # Evaluate agent
    agent_results = evaluate_agent(model, env)

    # Calculate baseline
    baseline_results, baseline_values = calculate_buy_and_hold_baseline(df_test)

    # Generate visualizations
    plot_performance_comparison(agent_results, baseline_results, baseline_values, df_test)

    # Generate report
    generate_report(agent_results, baseline_results)

    print("\n" + "="*70)
    print("EVALUATION COMPLETE!")
    print("="*70)
    print("Generated files:")
    print("  - performance_comparison.png")
    print("  - performance_report.txt")
    print("="*70)

if __name__ == "__main__":
    main()
