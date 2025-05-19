
# 🧠 ZiplineBot: Machine Learning-Driven Trading Agent

**Built at age 16**, ZiplineBot is a full-stack automated trading system designed to simulate and optimize trading strategies using machine learning, reinforcement learning, and real-world financial data.

## 🚀 Features

- 📈 **Asset Universe**: Supports BTC, ETH, XRP via Yahoo Finance (`yfinance`)
- 🤖 **Modeling**: Ensemble of LSTM + Random Forest for asset movement prediction
- 🎯 **Q-Learning Agent**: Custom-built reinforcement learning with adaptive reward system
- 📊 **Technical Indicators**: RSI, MACD, Bollinger Bands, Momentum, ADX, and more
- 🧮 **Portfolio Optimization**: Mean-variance optimization with Sharpe ratio objective
- 🔁 **Backtesting & Simulation**: Trade history tracking, live PnL computation
- 📤 **Discord Alerts**: Real-time updates on trading actions
- 📉 **Performance Metrics**: Sharpe, Sortino, Max Drawdown, Cumulative Return

## 🧠 Tech Stack

- Python 3.10+
- NumPy, Pandas, Scikit-learn, TensorFlow (Keras)
- YFinance, Matplotlib, Seaborn
- Discord Webhooks
- SciPy optimizer, TA-lib (or custom indicators)

## 📦 Project Structure

```
ZiplineBot.py
└── Core Logic:
    ├── Feature Engineering
    ├── ML Model Training (LSTM, RF)
    ├── Q-Learning Agent
    ├── Portfolio Optimization
    └── Backtest + Discord Integration
```

## 📉 Sample Output (Backtest)

- Sharpe Ratio: ~1.35
- Sortino Ratio: ~1.68
- Max Drawdown: ~12%
- Cumulative Return: +27% (3-month simulated test)

> ⚠️ **Note**: This is a **public academic version** of a proprietary bot that was sold privately. Live trading logic, sensitive models, and signal infrastructure have been removed or redacted to protect IP.

## 📚 Lessons Learned

- Built modular systems integrating ML, RL, and finance
- Learned to debug unstable agents in volatile environments
- Optimized tradeoff between prediction accuracy vs. execution latency
- Gained understanding of financial engineering at a practical level

## 🏁 Status

📦 **Archived**  
This repo remains as a portfolio demonstration. The original live-trading system was monetized and is no longer public.

---

© 2025 – Original Author: [QuackTheDuck](https://github.com/QuackTheBigDuck)
