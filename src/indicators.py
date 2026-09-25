"""Технические индикаторы на pandas/numpy.

Все функции КАУЗАЛЬНЫ: значение на баре i зависит только от баров <= i.
Это важно для честного бэктеста (никакого заглядывания в будущее).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def rma(s: pd.Series, n: int) -> pd.Series:
    """Сглаживание Уайлдера (используется в RSI/ATR/ADX)."""
    return s.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = rma(d.clip(lower=0), n)
    dn = rma((-d).clip(lower=0), n)
    rs = up / dn.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.fillna(100).where(dn.notna())


def macd(close: pd.Series, fast=12, slow=26, signal=9):
    line = ema(close, fast) - ema(close, slow)
    sig = ema(line, signal)
    return line, sig, line - sig


def true_range(df: pd.DataFrame) -> pd.Series:
    pc = df["close"].shift()
    return pd.concat(
        [df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1
    ).max(axis=1)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    return rma(true_range(df), n)


def adx(df: pd.DataFrame, n: int = 14):
    up = df["high"].diff()
    dn = -df["low"].diff()
    plus_dm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    tr = rma(true_range(df), n)
    plus_di = 100 * rma(plus_dm, n) / tr
    minus_di = 100 * rma(minus_dm, n) / tr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return rma(dx, n), plus_di, minus_di


def bollinger(close: pd.Series, n=20, k=2.0):
    mid = sma(close, n)
    sd = close.rolling(n, min_periods=n).std(ddof=0)
    return mid, mid + k * sd, mid - k * sd


def keltner(df: pd.DataFrame, n=20, k=1.5):
    mid = ema(df["close"], n)
    a = atr(df, n)
    return mid, mid + k * a, mid - k * a


def supertrend(df: pd.DataFrame, n=10, mult=3.0) -> pd.Series:
    """Возвращает направление: +1 (бычий) / -1 (медвежий)."""
    hl2 = (df["high"] + df["low"]) / 2
    a = atr(df, n).to_numpy()
    close = df["close"].to_numpy()
    ub = (hl2 + mult * a).to_numpy()
    lb = (hl2 - mult * a).to_numpy()
    mid = hl2.to_numpy()
    fub, flb = ub.copy(), lb.copy()
    direction = np.zeros(len(df))
    for i in range(len(df)):
        if i == 0 or np.isnan(a[i]):
            direction[i] = 1
            continue
        pu, pl = fub[i - 1], flb[i - 1]
        fub[i] = ub[i] if (np.isnan(pu) or ub[i] < pu or close[i - 1] > pu) else pu
        flb[i] = lb[i] if (np.isnan(pl) or lb[i] > pl or close[i - 1] < pl) else pl
        if np.isnan(pu) or np.isnan(pl):
            direction[i] = 1 if close[i] >= mid[i] else -1
            continue
        if direction[i - 1] == -1 and close[i] > fub[i]:
            direction[i] = 1
        elif direction[i - 1] == 1 and close[i] < flb[i]:
            direction[i] = -1
        else:
            direction[i] = direction[i - 1] if direction[i - 1] != 0 else 1
    return pd.Series(direction, index=df.index)


def donchian(df: pd.DataFrame, n=20):
    """Канал по ПРЕДЫДУЩИМ n барам (без текущего) — для пробоя."""
    hi = df["high"].shift(1).rolling(n, min_periods=n).max()
    lo = df["low"].shift(1).rolling(n, min_periods=n).min()
    return hi, lo


def vwap_daily(df: pd.DataFrame) -> pd.Series:
    """VWAP с якорем на начало суток UTC."""
    tp = (df["high"] + df["low"] + df["close"]) / 3
    day = df["time"].dt.floor("D")
    pv = (tp * df["volume"]).groupby(day).cumsum()
    vv = df["volume"].groupby(day).cumsum()
    return pv / vv.replace(0, np.nan)


def obv(df: pd.DataFrame) -> pd.Series:
    sign = np.sign(df["close"].diff().fillna(0))
    return (sign * df["volume"]).cumsum()


def zscore(s: pd.Series, n=50) -> pd.Series:
    m = s.rolling(n, min_periods=n).mean()
    sd = s.rolling(n, min_periods=n).std(ddof=0)
    return (s - m) / sd.replace(0, np.nan)


def stoch_rsi(close: pd.Series, n=14, k=3, d=3):
    r = rsi(close, n)
    lo = r.rolling(n, min_periods=n).min()
    hi = r.rolling(n, min_periods=n).max()
    st = 100 * (r - lo) / (hi - lo).replace(0, np.nan)
    kk = sma(st, k)
    return kk, sma(kk, d)


def pivots_confirmed(s: pd.Series, left=5, right=3, kind="low") -> pd.Series:
    """Пивот (локальный экстремум), ПОДТВЕРЖДЁННЫЙ через `right` баров.

    Значение появляется на баре i только когда пивот на баре i-right
    уже подтверждён — т.е. без заглядывания вперёд.
    Возвращает Series с ценой пивота на баре подтверждения, иначе NaN.
    """
    w = left + right + 1
    if kind == "low":
        ext = s.rolling(w, min_periods=w).min()
    else:
        ext = s.rolling(w, min_periods=w).max()
    cand = s.shift(right)
    is_piv = cand == ext
    return cand.where(is_piv)
