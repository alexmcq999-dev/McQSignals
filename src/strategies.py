"""Модули стратегий.

Каждый модуль — классическая, широко известная стратегия/сетап.
Модуль выдаёт «голос» в диапазоне [-1..+1] на каждом баре:
+1 — сильный лонг, -1 — сильный шорт, 0 — нет мнения.

Семейства (family) используются для взвешивания по режиму рынка:
  trend     — трендследование (работает в тренде)
  momentum  — импульс
  breakout  — пробой волатильности/диапазона
  reversion — возврат к среднему (работает во флэте)
  volume    — объём/поток денег
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import indicators as ta

# key: (название для сообщения, семейство)
MODULES: dict[str, tuple[str, str]] = {
    "ema_trend": ("EMA-тренд 20/50/200", "trend"),
    "supertrend": ("Supertrend (10, 3)", "trend"),
    "pullback": ("Откат к EMA по тренду", "trend"),
    "macd": ("MACD-импульс", "momentum"),
    "rsi_regime": ("RSI-режим (Cardwell)", "momentum"),
    "stoch_rsi": ("StochRSI разворот по тренду", "momentum"),
    "donchian": ("Пробой Дончиана 20 (Turtle)", "breakout"),
    "squeeze": ("Squeeze BB/Keltner (TTM)", "breakout"),
    "bb_reversion": ("Возврат в полосы Боллинджера", "reversion"),
    "rsi_divergence": ("Дивергенция RSI", "reversion"),
    "vwap_obv": ("VWAP + OBV (поток денег)", "volume"),
    # --- контртренд: против толпы и манипуляций (используются в отдельной ветке сигналов)
    "sweep": ("Снятие стопов и возврат (liquidity sweep)", "contrarian"),
    "crowd": ("Перекос толпы на фьючерсах (L/S ratio)", "contrarian"),
    "fng": ("Экстремум Fear & Greed", "contrarian"),
}
CONTRARIAN = ("sweep", "crowd", "fng")


def _sign(x: pd.Series) -> pd.Series:
    return np.sign(x).fillna(0)


def _recent(cond: pd.Series, bars: int) -> pd.Series:
    """True, если условие выполнялось хотя бы раз за последние `bars` баров (включая текущий)."""
    return cond.fillna(False).astype(int).rolling(bars, min_periods=1).max().astype(bool)


def htf_features(h: pd.DataFrame) -> pd.DataFrame:
    """Фичи старшего таймфрейма (1H): тренд-байас."""
    out = pd.DataFrame(index=h.index)
    out["close_time"] = h["close_time"]
    e50, e200 = ta.ema(h["close"], 50), ta.ema(h["close"], 200)
    st = ta.supertrend(h, 10, 3.0)
    adx, _, _ = ta.adx(h, 14)
    score = (_sign(h["close"] - e50) + _sign(e50 - e200) + st) / 3.0
    out["htf_score"] = score.where(e200.notna(), 0.0)
    out["htf_bias"] = np.where(out["htf_score"] > 0.34, 1, np.where(out["htf_score"] < -0.34, -1, 0))
    out["htf_adx"] = adx
    out["htf_rsi"] = ta.rsi(h["close"], 14)
    return out


def merge_htf(df: pd.DataFrame, htf: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    """Присоединяет к 15m барам ПОСЛЕДНЮЮ ЗАКРЫТУЮ 1H свечу (без look-ahead)."""
    left = df.sort_values("close_time")
    right = htf.sort_values("close_time").rename(
        columns={c: prefix + c for c in htf.columns if c != "close_time"}
    )
    merged = pd.merge_asof(left, right, on="close_time", direction="backward")
    merged.index = df.index
    return merged


def build_features(df: pd.DataFrame, h1: pd.DataFrame | None, btc_h1: pd.DataFrame | None = None,
                   crowd: pd.DataFrame | None = None, fng: pd.DataFrame | None = None) -> pd.DataFrame:
    """df — 15m свечи (time, open, high, low, close, volume, close_time)."""
    f = df.copy()
    c = f["close"]
    f["ema20"], f["ema50"], f["ema200"] = ta.ema(c, 20), ta.ema(c, 50), ta.ema(c, 200)
    f["rsi"] = ta.rsi(c, 14)
    f["macd"], f["macd_sig"], f["macd_hist"] = ta.macd(c)
    f["atr"] = ta.atr(f, 14)
    f["adx"], f["pdi"], f["mdi"] = ta.adx(f, 14)
    f["bb_mid"], f["bb_up"], f["bb_lo"] = ta.bollinger(c, 20, 2.0)
    _, f["kc_up"], f["kc_lo"] = ta.keltner(f, 20, 1.5)
    f["st_dir"] = ta.supertrend(f, 10, 3.0)
    f["dc_hi"], f["dc_lo"] = ta.donchian(f, 20)
    f["vwap"] = ta.vwap_daily(f)
    f["obv"] = ta.obv(f)
    f["obv_ema"] = ta.ema(f["obv"], 20)
    f["vol_z"] = ta.zscore(f["volume"], 50)
    f["srsi_k"], f["srsi_d"] = ta.stoch_rsi(c)
    f["swing_lo"] = f["low"].rolling(10, min_periods=1).min()
    f["swing_hi"] = f["high"].rolling(10, min_periods=1).max()
    f["atr_pct"] = f["atr"] / c * 100
    # уровни, за которыми обычно стоят стопы толпы
    f["liq_lo"] = f["low"].shift(1).rolling(20, min_periods=20).min()
    f["liq_hi"] = f["high"].shift(1).rolling(20, min_periods=20).max()

    if h1 is not None and len(h1):
        f = merge_htf(f, htf_features(h1))
    else:
        f["htf_bias"], f["htf_score"], f["htf_adx"], f["htf_rsi"] = 0, 0.0, np.nan, np.nan
    if btc_h1 is not None and len(btc_h1):
        b = htf_features(btc_h1)[["close_time", "htf_bias"]]
        f = merge_htf(f, b, prefix="btc_")
    else:
        f["btc_htf_bias"] = 0
    f["htf_bias"] = f["htf_bias"].fillna(0)
    f["btc_htf_bias"] = f["btc_htf_bias"].fillna(0)
    from .crowd import attach

    return attach(f, crowd, fng)


# ---------------------------------------------------------------- модули

def v_ema_trend(f):
    up_strong = (f.ema20 > f.ema50) & (f.ema50 > f.ema200) & (f.close > f.ema20)
    dn_strong = (f.ema20 < f.ema50) & (f.ema50 < f.ema200) & (f.close < f.ema20)
    up_weak = (f.ema20 > f.ema50) & (f.close > f.ema50)
    dn_weak = (f.ema20 < f.ema50) & (f.close < f.ema50)
    return pd.Series(
        np.select([up_strong, dn_strong, up_weak, dn_weak], [1.0, -1.0, 0.5, -0.5], 0.0), index=f.index
    )


def v_supertrend(f):
    flipped = _recent(f.st_dir != f.st_dir.shift(), 3)
    return f.st_dir * np.where(flipped, 1.0, 0.6)


def v_pullback(f):
    up = (f.ema50 > f.ema200) & (f.htf_bias >= 0)
    dn = (f.ema50 < f.ema200) & (f.htf_bias <= 0)
    touched_dn = _recent(f.low <= f.ema50 * 1.002, 4) & (f.close > f.ema20) & (f.close > f.open)
    touched_up = _recent(f.high >= f.ema50 * 0.998, 4) & (f.close < f.ema20) & (f.close < f.open)
    rsi_up = (f.rsi > f.rsi.shift(1)) & f.rsi.between(40, 62)
    rsi_dn = (f.rsi < f.rsi.shift(1)) & f.rsi.between(38, 60)
    return pd.Series(
        np.select([up & touched_dn & rsi_up, dn & touched_up & rsi_dn], [1.0, -1.0], 0.0), index=f.index
    )


def v_macd(f):
    cross_up = _recent((f.macd > f.macd_sig) & (f.macd.shift() <= f.macd_sig.shift()), 3)
    cross_dn = _recent((f.macd < f.macd_sig) & (f.macd.shift() >= f.macd_sig.shift()), 3)
    rising = f.macd_hist > f.macd_hist.shift()
    return pd.Series(
        np.select(
            [
                cross_up & (f.macd_hist > 0),
                cross_dn & (f.macd_hist < 0),
                (f.macd_hist > 0) & rising,
                (f.macd_hist < 0) & ~rising,
            ],
            [1.0, -1.0, 0.6, -0.6],
            0.0,
        ),
        index=f.index,
    )


def v_rsi_regime(f):
    r, up = f.rsi, f.rsi > f.rsi.shift(2)
    return pd.Series(
        np.select(
            [
                r.between(55, 72) & up,
                r.between(28, 45) & ~up,
                r.between(50, 55) & up,
                r.between(45, 50) & ~up,
            ],
            [1.0, -1.0, 0.4, -0.4],
            0.0,
        ),
        index=f.index,
    )


def v_stoch_rsi(f):
    k, d = f.srsi_k, f.srsi_d
    xu = _recent((k > d) & (k.shift() <= d.shift()) & (k.shift() < 25), 2)
    xd = _recent((k < d) & (k.shift() >= d.shift()) & (k.shift() > 75), 2)
    return pd.Series(
        np.select([xu & (f.ema50 > f.ema200), xd & (f.ema50 < f.ema200)], [1.0, -1.0], 0.0), index=f.index
    )


def v_donchian(f):
    vol = f.vol_z > 1.0
    return pd.Series(
        np.select(
            [(f.close > f.dc_hi) & vol, (f.close < f.dc_lo) & vol, f.close > f.dc_hi, f.close < f.dc_lo],
            [1.0, -1.0, 0.5, -0.5],
            0.0,
        ),
        index=f.index,
    )


def v_squeeze(f):
    on = (f.bb_up < f.kc_up) & (f.bb_lo > f.kc_lo)
    released = _recent(on.shift(1).fillna(False).astype(bool) & ~on, 3)
    return pd.Series(
        np.select(
            [released & (f.close > f.bb_mid) & (f.macd_hist > 0), released & (f.close < f.bb_mid) & (f.macd_hist < 0)],
            [1.0, -1.0],
            0.0,
        ),
        index=f.index,
    )


def v_bb_reversion(f):
    back_in_lo = (f.close.shift() < f.bb_lo.shift()) & (f.close > f.bb_lo)
    back_in_hi = (f.close.shift() > f.bb_up.shift()) & (f.close < f.bb_up)
    oversold = _recent(f.rsi < 32, 4)
    overbought = _recent(f.rsi > 68, 4)
    return pd.Series(
        np.select([back_in_lo & oversold, back_in_hi & overbought], [1.0, -1.0], 0.0), index=f.index
    )


def _divergence(f, kind: str, right: int = 3):
    src = f.low if kind == "low" else f.high
    piv = ta.pivots_confirmed(src, left=5, right=right, kind=kind)
    rsi_at = f.rsi.shift(right).where(piv.notna())
    bar = pd.Series(np.arange(len(f)), index=f.index, dtype=float).where(piv.notna())
    ev = pd.DataFrame({"p": piv, "r": rsi_at, "b": bar}).dropna()
    prev = ev.shift(1)
    last = ev.reindex(f.index).ffill()
    prev = prev.reindex(f.index).ffill()
    idx = pd.Series(np.arange(len(f)), index=f.index)
    fresh = (idx - last["b"]) <= 4
    close_gap = (last["b"] - prev["b"]).between(5, 60)
    if kind == "low":
        cond = (last.p < prev.p) & (last.r > prev.r) & (prev.r < 35) & (f.close > last.p)
    else:
        cond = (last.p > prev.p) & (last.r < prev.r) & (prev.r > 65) & (f.close < last.p)
    return (cond & fresh & close_gap).fillna(False)


def v_rsi_divergence(f):
    bull = _divergence(f, "low")
    bear = _divergence(f, "high")
    return pd.Series(np.select([bull, bear], [1.0, -1.0], 0.0), index=f.index)


def v_vwap_obv(f):
    return (_sign(f.close - f.vwap) + _sign(f.obv - f.obv_ema)) / 2.0


def v_sweep(f):
    """Охота за стопами: цена прокалывает 20-барный экстремум и закрывается обратно
    с длинной тенью на повышенном объёме → торгуем против прокола."""
    rng = (f.high - f.low).replace(0, np.nan)
    lower_wick = (np.minimum(f.open, f.close) - f.low) / rng
    upper_wick = (f.high - np.maximum(f.open, f.close)) / rng
    vol = f.vol_z > 0.5
    bull = (f.low < f.liq_lo) & (f.close > f.liq_lo) & (lower_wick >= 0.5) & vol
    bear = (f.high > f.liq_hi) & (f.close < f.liq_hi) & (upper_wick >= 0.5) & vol
    return pd.Series(np.select([bull, bear], [1.0, -1.0], 0.0), index=f.index)


def v_crowd(f):
    """Против розничной толпы: сильный перекос лонгов по числу счетов → шорт-бias, и наоборот.
    Если топ-трейдеры стоят вместе с толпой — сигнал ослабляем."""
    z = f.get("ls_acc_z", pd.Series(np.nan, index=f.index))
    top = f.get("ls_top_z", pd.Series(np.nan, index=f.index))
    d = -np.sign(z)
    strength = np.where(z.abs() >= 2.0, 1.0, np.where(z.abs() >= 1.5, 0.6, 0.0))
    with_crowd = (np.sign(top) == np.sign(z)) & (top.abs() >= 1.0)
    strength = np.where(with_crowd, strength * 0.5, strength)
    return pd.Series(d * strength, index=f.index).fillna(0.0)


def v_fng(f):
    """Экстремумы Fear & Greed: «покупай страх, продавай жадность»."""
    x = f.get("fng", pd.Series(np.nan, index=f.index))
    return pd.Series(np.select([x <= 20, x <= 30, x >= 80, x >= 70], [1.0, 0.5, -1.0, -0.5], 0.0), index=f.index)


VOTERS = {
    "ema_trend": v_ema_trend,
    "supertrend": v_supertrend,
    "pullback": v_pullback,
    "macd": v_macd,
    "rsi_regime": v_rsi_regime,
    "stoch_rsi": v_stoch_rsi,
    "donchian": v_donchian,
    "squeeze": v_squeeze,
    "bb_reversion": v_bb_reversion,
    "rsi_divergence": v_rsi_divergence,
    "vwap_obv": v_vwap_obv,
    "sweep": v_sweep,
    "crowd": v_crowd,
    "fng": v_fng,
}


def all_votes(f: pd.DataFrame, enabled: list[str] | None = None) -> pd.DataFrame:
    keys = enabled or list(VOTERS)
    return pd.DataFrame({k: VOTERS[k](f).astype(float).fillna(0.0) for k in keys}, index=f.index)
