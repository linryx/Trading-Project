import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def calculate_momentum(df, periods):
    # Simple momentum 
    for period in periods:
        df[f"mom_{period}d"] = df["Close"].pct_change(period)
    return df

def calculate_log_momentum(df):
    # Log momentum
    log_ret = np.log(df["Close"] / df["Close"].shift(1))
"""
df["log_mom_20d"] = log_ret.rolling(20).sum()
df["log_mom_60d"] = log_ret.rolling(60).sum()

# Price vs Moving Averages
df["ma_20"] = df["Close"].rolling(20).mean()
df["ma_50"] = df["Close"].rolling(50).mean()

df["price_vs_ma20"] = df["Close"] / df["ma_20"]
df["price_vs_ma50"] = df["Close"] / df["ma_50"]

# Cross-sectional momentum
returns_60d = df["Close"].pct_change(60)

momentum_rank = returns_60d.rank(axis=1, pct=True)

# Risk-adjusted momentum
df["mom_20d"] = df["Close"].pct_change(20)

df["vol_20d"] = (
    np.log(df["Close"] / df["Close"].shift(1))
    .rolling(20)
    .std()
)

df["risk_adj_mom"] = df["mom_20d"] / df["vol_20d"]

# RSI
delta = df["Close"].diff()

gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)

avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()

rs = avg_gain / avg_loss

df["rsi"] = 100 - (100 / (1 + rs))

# ML
features = [
    "mom_5d",
    "mom_20d",
    "mom_60d",
    "vol_20d",
    "vol_60d",
    "price_vs_ma20",
    "price_vs_ma50",
    "ma_spread",
    "risk_adj_mom",
]

# Predict next 5-day return
df["target"] = df["Close"].pct_change(5).shift(-5)
"""