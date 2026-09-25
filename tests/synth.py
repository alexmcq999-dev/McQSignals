import numpy as np, pandas as pd

def synth(n=3000, seed=0, start="2026-06-01", tf_min=15, drift_regimes=True, p0=100.0):
    rng = np.random.default_rng(seed)
    mu = np.zeros(n)
    if drift_regimes:
        i = 0
        while i < n:
            L = rng.integers(100, 400); mu[i:i+L] = rng.choice([-1, 0, 0, 1]) * 0.0006; i += L
    ret = mu + rng.normal(0, 0.004, n)
    close = p0 * np.exp(np.cumsum(ret))
    op = np.r_[p0, close[:-1]]
    wig = np.abs(rng.normal(0, 0.002, n))
    high = np.maximum(op, close) * (1 + wig); low = np.minimum(op, close) * (1 - wig)
    vol = rng.lognormal(10, 0.5, n) * (1 + 20 * np.abs(ret))
    t = pd.date_range(start, periods=n, freq=f"{tf_min}min", tz="UTC")
    df = pd.DataFrame({"time": t, "open": op, "high": high, "low": low, "close": close, "volume": vol})
    df["close_time"] = df["time"] + pd.Timedelta(minutes=tf_min) - pd.Timedelta(milliseconds=1)
    return df

def resample(df, rule="1h"):
    g = df.set_index("time").resample(rule, label="left", closed="left")
    h = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
                      "close": g["close"].last(), "volume": g["volume"].sum()}).dropna().reset_index()
    h["close_time"] = h["time"] + pd.Timedelta(rule) - pd.Timedelta(milliseconds=1)
    return h
