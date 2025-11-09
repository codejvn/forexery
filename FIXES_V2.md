# V2 Fixes - Addressing Major Failures

## 🔴 **What Went Wrong:**

### Previous Results:
- **Total Return**: -6.13% (lost half the balance!)
- **Total Trades**: 252 (63 per quarter - STILL overtrading!)
- **Win Rate**: 25% (75% of trades lost money)
- **Sharpe Ratio**: -1.24 (terrible)
- **Underperformance**: -6.11% vs buy-and-hold

### Root Causes:
1. **Overtrading penalty too weak** (0.01 did nothing)
2. **Too much exploration** (0.05 entropy = random bad trades)
3. **Network too complex** (512-512-256 overfitting)
4. **Reward normalization** (VecNormalize confused the agent)
5. **Unclear reward signals** (complex multi-component rewards)

---

## ✅ **Changes Applied:**

### 1. **DRASTICALLY Increased Overtrading Penalty**
```python
OVERTRADING_PENALTY = 1.0  # Was 0.01 (100x increase!)
TRANSACTION_COST = 0.0002   # Was 0.0001 (2x increase)
```
**Why**: 0.01 penalty did NOTHING (still 252 trades). Now each trade costs 1.0 reward points, which should reduce trading to 5-15 per quarter.

### 2. **Killed Random Exploration**
```python
ENT_COEF = 0.001  # Was 0.05 (50x reduction!)
```
**Why**: High entropy (0.05) caused random exploration = random bad trades. With 0.001, the agent will be much more deterministic and only trade when confident.

### 3. **Simplified Network**
```python
net_arch = [256, 128]  # Was [512, 512, 256]
```
**Why**: Deep complex network was likely overfitting the training data. Simpler network should generalize better.

### 4. **Removed VecNormalize**
```python
# REMOVED: VecNormalize wrapper
```
**Why**: Normalizing rewards (-6% becomes some normalized value) made it hard for agent to understand actual P&L. Raw rewards are clearer.

### 5. **Simplified Rewards**
```python
DAILY_REWARD_WEIGHT = 1.0      # Was 0.1 (10x increase)
QUARTERLY_REWARD_WEIGHT = 100  # Was 200 (2x decrease)
SHARPE_BONUS_WEIGHT = 10       # Was 20 (2x decrease)
```
**Why**: Daily P&L is now the PRIMARY signal. Quarterly and Sharpe are bonuses. Clearer learning signal.

### 6. **Reduced Training Time**
```python
TRAINING_TIMESTEPS = 300000  # Was 500000
```
**Why**: Maybe we overtrained and the agent learned bad habits. Shorter training with better rewards should work better.

### 7. **Restored Learning Rate**
```python
LEARNING_RATE = 0.0003  # Was 0.0001
GAMMA = 0.99            # Was 0.995
```
**Why**: Faster learning with less focus on distant future rewards.

---

## 🎯 **Expected Impact:**

| Metric | Before | Target |
|--------|--------|--------|
| Trades per Quarter | 63 | 5-15 |
| Win Rate | 25% | 50%+ |
| Total Return | -6.13% | +2% to +5% |
| Sharpe Ratio | -1.24 | > 0.5 |

### Key Improvements:
1. **Fewer Trades**: 1.0 penalty + 0.0002 transaction cost = agent will only trade when VERY confident
2. **Better Trades**: 0.001 entropy = less random, more strategic
3. **Clearer Learning**: Raw rewards (no normalization) + daily P&L focus = agent understands profit/loss better
4. **Less Overfitting**: Simpler network should generalize better to test data

---

## 📊 **Trade-off Analysis:**

### What We Sacrificed:
- ❌ Deep network capacity (might miss complex patterns)
- ❌ Exploration (might not discover creative strategies)
- ❌ Long training (less experience)

### What We Gained:
- ✅ Much stronger anti-overtrading signal
- ✅ Deterministic, confident trading
- ✅ Clear reward understanding
- ✅ Better generalization (less overfitting)

---

## 🚀 **Next Steps:**

```bash
# Retrain with new settings (~45 min instead of 2 hours)
python training.py

# Evaluate
python evaluation.py
```

---

## 📈 **What to Look For:**

### Good Signs:
- ✅ Total trades: 20-60 (5-15 per quarter)
- ✅ Win rate: > 40%
- ✅ Return: -2% to +5%
- ✅ Beats or matches buy-and-hold

### Bad Signs (if still failing):
- ❌ Still 200+ trades = penalty still too weak
- ❌ Still negative return = fundamental issue with features/data
- ❌ Win rate < 35% = agent can't predict direction

---

## 🔄 **If This Fails Too:**

### Option A: Nuclear Option - Simple Buy-and-Hold Strategy
Just predict USD price direction (up/down), hold position for full quarter:
- If predict up → 100% USD
- If predict down → 100% JPY
- Only 4 trades per year (1 per quarter)

### Option B: Try Different Algorithm
Switch from PPO to SAC (better for continuous actions):
```python
from stable_baselines3 import SAC
model = SAC("MlpPolicy", env, learning_rate=0.0003)
```

### Option C: Check Data Quality
Maybe the features aren't predictive:
- Plot sentiment vs returns correlation
- Check if USDJPY had any trend 2006-2021
- Verify technical indicators are calculated correctly

---

## 💡 **Key Insight:**

**The core problem was: agent learned to trade, but learned to trade BADLY.**

With these changes, we're forcing it to trade LESS and only when CONFIDENT. Sometimes in trading, doing nothing is better than doing something.

**Quote**: "In trading, you make money in two ways: being right, or not trading."

Let's see if the agent learns the second lesson now! 🤞
