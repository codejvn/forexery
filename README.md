# forexery

A reinforcement learning agent for USD/JPY forex trading that combines technical indicators with NLP-based news sentiment analysis.

## Overview

forexery trains a PPO (Proximal Policy Optimization) agent to trade the USD/JPY currency pair. The agent observes technical market indicators alongside sentiment signals derived from US and Japanese financial news, then outputs continuous position-sizing actions to manage a simulated portfolio.

## How It Works

### Data Pipeline

1. **News scraping** (`webscraper.py`) — Uses an AI agent (via Dedalus + Exa MCP) to collect daily financial and political headlines from the US and Japan, labeled with sentiment (Positive / Negative / Neutral).
2. **Data merging** (`data_merger.py`) — Joins scraped news sentiment with OHLCV forex data and technical indicators, builds historical sentiment features (moving averages, volatility, momentum) with a 30-day lookback, and outputs the final training dataset.

### Trading Environment

`trading_environment.py` implements a custom [Gymnasium](https://gymnasium.farama.org/) environment (`JPYUSDTradingEnv`):

- **Action space**: Continuous `[-1, 1]` — positive values buy USD (sell JPY), negative values sell USD (buy JPY), zero holds.
- **Observation space**: Technical indicators (price, returns, RSI, MACD, etc.) + US/Japan sentiment features (mean, moving averages, volatility, momentum, divergence) + portfolio state (USD ratio, JPY ratio, total value).
- **Reward**: Weighted mix of daily P&L, quarterly return, and Sharpe ratio bonus.
- **Episode length**: 63 trading days (one quarter).

### Training

`training.py` trains a PPO agent using [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) with:

- 70/15/15 train/val/test split
- EvalCallback for best-model checkpointing
- TensorBoard logging

### Evaluation

`evaluation.py` benchmarks the trained agent against a buy-and-hold baseline across 4 quarters of held-out test data.

## Results

Evaluated over 4 quarters of held-out test data.

| Metric | RL Agent | Buy & Hold |
|---|---|---|
| Total Return | **+0.84%** | -0.02% |
| Outperformance | **+0.86%** | — |
| Sharpe Ratio | **1.67** | -0.61 |
| Max Drawdown | -0.14% | — |
| Win Rate (quarterly) | **100%** | — |
| Total Trades | 247 | — |

## Project Structure

```
forexery/
├── config.py                # All hyperparameters and file paths
├── webscraper.py            # AI-powered news headline scraper
├── data_merger.py           # Merges news sentiment + market data
├── trading_environment.py   # Custom Gymnasium trading environment
├── training.py              # PPO training script
├── evaluation.py            # Agent evaluation & benchmarking
├── models/                  # Saved model checkpoints
├── logs/                    # Training logs
└── tensorboard_logs/        # TensorBoard event files
```

## Setup

```bash
pip install stable-baselines3 gymnasium pandas numpy matplotlib dedalus-labs python-dotenv
```

Create a `.env` file with any required API keys for the Dedalus/Exa scraper.

## Usage

```bash
# 1. Scrape news data
python webscraper.py

# 2. Merge and preprocess data
python data_merger.py

# 3. Train the agent
python training.py

# 4. Evaluate performance
python evaluation.py
```

Monitor training with TensorBoard:

```bash
tensorboard --logdir tensorboard_logs/
```

## Configuration

All hyperparameters live in `config.py`:

| Parameter | Default | Description |
|---|---|---|
| `INITIAL_BALANCE` | $10,000 | Starting portfolio value |
| `TRANSACTION_COST` | 0.0001 | Cost per trade (1 pip) |
| `EPISODE_LENGTH` | 63 days | One trading quarter |
| `TRAINING_TIMESTEPS` | 100,000 | Total RL training steps |
| `LEARNING_RATE` | 0.0003 | PPO learning rate |
| `QUARTERLY_REWARD_WEIGHT` | 100.0 | Weight on quarterly return reward |
