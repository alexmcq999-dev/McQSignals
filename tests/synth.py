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


def breakout_pattern(n_pre=800, base=36, seed=0, start="2026-01-01", after=12, retest=False):
    """1H-свечи: тренд вверх → восходящий треугольник (плоский верх 105, растущие минимумы,
    затухающий объём) → свеча пробоя с объёмом → `after` свечей продолжения (или ретест уровня)."""
    rng = np.random.default_rng(seed)
    pre = np.linspace(60, 102.5, n_pre) * np.exp(rng.normal(0, 0.002, n_pre))
    # треугольник: пилообразно между растущим минимумом и потолком 105
    tri = []
    for k in range(base):
        ph = (k % 12) / 12.0                       # 3 волны по 12 свечей
        lo_b = 102.6 + 1.2 * k / base               # растущие минимумы
        top = 104.85
        tri.append(lo_b + (top - lo_b) * (1 - abs(2 * ph - 1)))
    tri = np.array(tri)
    bo_close = 105.3 if not retest else 106.6
    post = (np.linspace(bo_close, bo_close + 3, after) if not retest
            else np.r_[np.linspace(106.3, 105.1, 4), np.linspace(105.4, 108, max(0, after - 4))])
    close = np.r_[pre, tri, bo_close, post]
    n = len(close)
    op = np.r_[close[0], close[:-1]]
    wig = np.full(n, 0.25)
    high = np.maximum(op, close) + wig
    low = np.minimum(op, close) - wig
    # верх базы — ровно 105 на пиках волн (касания)
    b0 = n_pre
    for k in range(base):
        if k % 12 == 6:
            high[b0 + k] = 105.0
    high[b0:b0 + base] = np.minimum(high[b0:b0 + base], 105.0)
    i_bo = b0 + base
    op[i_bo], low[i_bo], high[i_bo] = 104.7, 104.6, close[i_bo] + 0.1
    vol = np.r_[rng.lognormal(10, 0.2, n_pre), np.linspace(1.0, 0.5, base) * np.exp(10), [np.exp(10) * 4],
                np.full(len(post), np.exp(10) * 1.5)]
    t = pd.date_range(start, periods=n, freq="60min", tz="UTC")
    df = pd.DataFrame({"time": t, "open": op, "high": high, "low": low, "close": close, "volume": vol})
    df["close_time"] = df["time"] + pd.Timedelta(minutes=60) - pd.Timedelta(milliseconds=1)
    return df, i_bo


def mirror(df, c=220.0):
    """Зеркальное отражение цены (для проверки шортов): p → c − p."""
    m = df.copy()
    m["open"], m["close"] = c - df["open"], c - df["close"]
    m["high"], m["low"] = c - df["low"], c - df["high"]
    return m
