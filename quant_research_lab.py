import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

spy = yf.download(
    "SPY",
    start="2023-01-01",
    end="2024-01-01",
    auto_adjust=False,
    progress=False
)

if isinstance(spy.columns, pd.MultiIndex):
    prices = spy["Adj Close"].iloc[:, 0]
else:
    prices = spy["Adj Close"]

prices = prices.astype(float)

daily_returns = prices.pct_change()

ma20 = prices.rolling(20).mean()
ma50 = prices.rolling(50).mean()

signal = (ma20 > ma50).astype(int)

position = signal.shift(1).fillna(0)

strategy_returns = position * daily_returns

buy_and_hold = (1 + daily_returns).cumprod()
moving_average_strategy = (1 + strategy_returns).cumprod()

plt.figure(figsize=(12, 6))
plt.plot(prices, label="SPY Price")
plt.plot(ma20, label="20-Day Moving Average")
plt.plot(ma50, label="50-Day Moving Average")
plt.title("SPY Price With Moving Averages")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(buy_and_hold, label="Buy and Hold SPY")
plt.plot(
    moving_average_strategy,
    label="Moving Average Strategy"
)
plt.title("Moving Average Strategy vs. Buy and Hold")
plt.legend()
plt.grid()
plt.show()

valid_returns = pd.DataFrame({
    "Buy and Hold": daily_returns,
    "Strategy": strategy_returns
}).dropna()

buy_hold_return = buy_and_hold.dropna().iloc[-1] - 1
strategy_return = moving_average_strategy.dropna().iloc[-1] - 1

buy_hold_volatility = (
    valid_returns["Buy and Hold"].std() * (252 ** 0.5)
)

strategy_volatility = (
    valid_returns["Strategy"].std() * (252 ** 0.5)
)

buy_hold_sharpe = buy_hold_return / buy_hold_volatility
strategy_sharpe = strategy_return / strategy_volatility

print("----- RESULTS -----")
print("Buy and Hold Return:",
      round(buy_hold_return * 100, 2), "%")

print("Moving Average Strategy Return:",
      round(strategy_return * 100, 2), "%")

print("Buy and Hold Volatility:",
      round(buy_hold_volatility * 100, 2), "%")

print("Strategy Volatility:",
      round(strategy_volatility * 100, 2), "%")

print("Buy and Hold Sharpe Ratio:",
      round(buy_hold_sharpe, 2))

print("Strategy Sharpe Ratio:",
      round(strategy_sharpe, 2))