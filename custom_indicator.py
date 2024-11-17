import numpy as np
import vectorbt as vbt

# get data
btc_price = vbt.YFData.download(["BTC-USD"], missing_index="drop").get("Close")
rsi = vbt.RSI.run(btc_price, window=14)


# create custom indicator
def custom_indicator(close, rsi_window=14, ma_window=50):
    rsi = vbt.RSI.run(close, window=rsi_window).rsi.to_numpy()
    ma = vbt.MA.run(close, window=ma_window).ma.to_numpy()
    trend = np.where(rsi > 70, -1, 0)
    trend = np.where((rsi < 30) & (close < ma), 1, trend)
    return trend, rsi


ind = vbt.IndicatorFactory(
    class_name="Combined",
    short_name="comb",
    input_names=["close"],
    param_names=["rsi_window", "ma_window"],
    output_names=["value", "rsi"],
).from_apply_func(custom_indicator, rsi_window=14, ma_window=50)

res = ind.run(btc_price, rsi_window=21, ma_window=50)
entries = res.value == 1
exits = res.value == -1


# backtest portfolio with entry and exit signals
portfolio = vbt.Portfolio.from_signals(btc_price, entries, exits)
print(portfolio.stats())
