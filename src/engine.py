"""Движок: режим рынка → взвешенный консенсус модулей → фильтры → уровни.

Идея (как у профессиональных деск-трейдеров): ни одна стратегия не работает
всегда. Трендовые сетапы зарабатывают в тренде, контртрендовые — во флэте.
Поэтому:
  1. Определяем режим (ADX): тренд / флэт / нейтрально.
  2. Каждый модуль голосует, вес голоса зависит от режима.
  3. Сигнал — только при сильном консенсусе, по тренду старшего ТФ,
     с учётом BTC и адекватной волатильности, и только в момент «триггера»
     (сила только что пересекла порог), а не всё время тренда.
  4. Стоп — за структурой (свинг) с ограничением в ATR, цели — в R.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from .strategies import CONTRARIAN, MODULES, all_votes, build_features

REGIME_WEIGHTS = {
    "trend": {"trend": 1.0, "momentum": 1.0, "breakout": 1.0, "reversion": 0.3, "volume": 0.8},
    "range": {"trend": 0.4, "momentum": 0.7, "breakout": 0.7, "reversion": 1.4, "volume": 0.8},
    "neutral": {"trend": 0.8, "momentum": 0.9, "breakout": 0.9, "reversion": 0.8, "volume": 0.8},
}
REGIME_RU = {"trend": "тренд", "range": "флэт", "neutral": "переходный"}


def regime_of(adx: pd.Series) -> pd.Series:
    return pd.Series(np.where(adx >= 23, "trend", np.where(adx < 18, "range", "neutral")), index=adx.index)


def score_frame(f: pd.DataFrame, cfg: dict, is_btc: bool = False) -> pd.DataFrame:
    """Возвращает f + колонки: score, conf, n_agree, regime, signal (+1/-1/0) и голоса модулей."""
    sc = cfg["signals"]
    mods = [m for m in sc.get("modules", list(MODULES)) if m in MODULES and m not in CONTRARIAN]
    votes = all_votes(f, mods)
    regime = regime_of(f["adx"])

    W = pd.DataFrame(index=f.index, columns=mods, dtype=float)
    for reg, wmap in REGIME_WEIGHTS.items():
        mask = regime == reg
        for m in mods:
            W.loc[mask, m] = wmap[MODULES[m][1]]
    raw = (votes * W).sum(axis=1) / W.sum(axis=1) * 100.0

    direction = np.sign(raw)
    n_long = (votes > 0.3).sum(axis=1)
    n_short = (votes < -0.3).sum(axis=1)
    n_agree = np.where(direction > 0, n_long, n_short)

    htf = f["htf_bias"].fillna(0)
    btc = f["btc_htf_bias"].fillna(0) if not is_btc else pd.Series(0, index=f.index)
    bonus = np.where(htf == direction, 5.0, 0.0)
    if sc.get("btc_filter", True) and not is_btc:
        bonus += np.where(btc == direction, 3.0, np.where(btc == -direction, -10.0, 0.0))
    bonus += np.where(f["vol_z"] > 1.5, 3.0, 0.0)
    conf = (raw.abs() + bonus).clip(0, 100)

    ok = (
        (conf >= sc["min_score"])
        & (n_agree >= sc["min_agree"])
        & f["ema200"].notna()
        & f["atr_pct"].between(sc["atr_pct_min"], sc["atr_pct_max"])
        & (f["vol_z"] > -1.0)
    )
    if sc.get("require_htf", True):
        if sc.get("htf_mode", "loose") == "strict":
            ok &= htf == direction  # тренд 1H должен СОВПАДАТЬ с направлением
        else:
            ok &= (htf * direction) >= 0  # достаточно, чтобы не был против
    if not is_btc and sc.get("btc_mode", "penalty") == "block":
        ok &= (btc * direction) >= 0  # альты не торгуем против тренда BTC 1H
    sides = sc.get("sides", "both")
    cond_long = ok & (direction > 0) & (sides != "short_only")
    cond_short = ok & (direction < 0) & (sides != "long_only")
    if sc.get("strict_shorts", False):
        # шорт только когда медвежьи и 1H монеты, и 1H BTC
        cond_short &= (htf == -1) & ((btc == -1) if not is_btc else True)
    # Фильтры «не идти за толпой» для трендовых сигналов
    cr = cfg.get("contrarian", {})
    if cr.get("crowd_filter", False) and "ls_acc_z" in f:
        z = f["ls_acc_z"].fillna(0)
        lim = cr.get("crowd_filter_z", 2.0)
        cond_long &= ~(z >= lim)    # толпа и так перегружена лонгами — не присоединяемся
        cond_short &= ~(z <= -lim)
    if cr.get("fng_filter", False) and "fng" in f:
        g = f["fng"]
        cond_long &= ~(g >= cr.get("fng_greed", 80))
        cond_short &= ~(g <= cr.get("fng_fear", 20))
    # Триггер: условие появилось на ЭТОМ баре (событие, а не состояние)
    trig_long = cond_long & ~cond_long.shift(1, fill_value=False)
    trig_short = cond_short & ~cond_short.shift(1, fill_value=False)

    # ---- Контртрендовая ветка: против толпы/манипуляции
    cvotes = pd.DataFrame({m: all_votes(f, [m])[m] for m in CONTRARIAN}, index=f.index)
    kind = pd.Series("", index=f.index)
    kind[trig_long | trig_short] = "trend"
    if cr.get("enabled", False):
        trig = cvotes[cr.get("trigger", "sweep")]
        d = np.sign(trig)
        confirms = [m for m in cr.get("confirm", ["crowd", "fng"]) if m in cvotes]
        n_conf = sum(((cvotes[m] * d) > 0.3).astype(int) for m in confirms) if confirms else 0
        c_ok = (trig.abs() > 0.3) & (n_conf >= cr.get("min_confirm", 1)) & f["ema200"].notna() \
            & f["atr_pct"].between(sc["atr_pct_min"], sc["atr_pct_max"])
        if not cr.get("ignore_htf", True):
            c_ok &= (htf * d) >= 0
        sides = sc.get("sides", "both")
        c_long = c_ok & (d > 0) & (sides != "short_only") & ~(trig_long | trig_short)
        c_short = c_ok & (d < 0) & (sides != "long_only") & ~(trig_long | trig_short)
        if cr.get("only", False):  # только контртренд (для экспериментов)
            trig_long = trig_long & False
            trig_short = trig_short & False
            kind[:] = ""
        trig_long |= c_long
        trig_short |= c_short
        kind[c_long | c_short] = "contra"
        c_conf = (cr.get("base_conf", 60) + 10 * pd.Series(n_conf, index=f.index)).clip(0, 100)
        conf = conf.where(~(c_long | c_short), c_conf)

    out = f.copy()
    out["regime"] = regime
    out["score"] = raw
    out["conf"] = conf
    out["n_agree"] = n_agree
    out["signal"] = np.where(trig_long, 1, np.where(trig_short, -1, 0))
    out["kind"] = kind
    for m in mods:
        out["v_" + m] = votes[m]
    for m in CONTRARIAN:
        out["v_" + m] = cvotes[m]
    return out


def analyze(df: pd.DataFrame, htf: pd.DataFrame | None, btc_htf: pd.DataFrame | None, cfg: dict, is_btc=False,
            crowd=None, fng=None):
    return score_frame(build_features(df, htf, None if is_btc else btc_htf, crowd, fng), cfg, is_btc=is_btc)


# ---------------------------------------------------------------- уровни и сделка

@dataclass
class Trade:
    symbol: str
    side: int  # +1 long / -1 short
    entry: float
    sl: float
    tp1: float
    tp2: float
    risk: float  # |entry - sl|
    opened: str  # ISO время закрытия сигнальной свечи (UTC)
    conf: float = 0.0
    regime: str = ""
    reasons: list = field(default_factory=list)
    kind: str = "trend"  # trend | contra
    cost_r: float = 0.0  # издержки туда-обратно в R
    atr: float = 0.0     # ATR на входе (для трейлинга)
    peak: float = 0.0    # лучшая цена с момента входа (для трейлинга)
    status: str = "open"  # open | tp1 | closed
    exit_reason: str = ""
    r_gross: float = 0.0
    r_net: float = 0.0
    closed: str = ""
    last_checked: str = ""
    bars_held: int = 0
    id: str = ""
    msg_ids: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def make_trade(row: pd.Series, side: int, symbol: str, cfg: dict, entry: float | None = None) -> Trade:
    rk = cfg["risk"]
    e = float(entry if entry is not None else row["close"])
    a = float(row["atr"])
    if side > 0:
        struct = float(row["swing_lo"]) - rk["sl_swing_buffer_atr"] * a
        dist = e - struct
    else:
        struct = float(row["swing_hi"]) + rk["sl_swing_buffer_atr"] * a
        dist = struct - e
    dist = float(np.clip(dist, rk["sl_min_atr"] * a, rk["sl_max_atr"] * a))
    dist = min(dist, e * rk["sl_max_pct"] / 100)
    dist = max(dist, e * rk.get("sl_min_pct", 0.0) / 100)  # слишком узкий стоп съедается комиссией
    sl = e - side * dist
    kind = str(row.get("kind", "") or "trend")
    pool = CONTRARIAN if kind == "contra" else [m for m in MODULES if m not in CONTRARIAN]
    reasons = [m for m in pool if ("v_" + m) in row and row["v_" + m] * side > 0.3]
    return Trade(
        symbol=symbol,
        side=side,
        entry=e,
        sl=sl,
        tp1=e + side * rk["tp1_r"] * dist,
        tp2=e + side * rk["tp2_r"] * dist,
        risk=dist,
        opened=pd.Timestamp(row["close_time"]).isoformat(),
        conf=round(float(row["conf"]), 1),
        regime=str(row["regime"]),
        reasons=reasons,
        kind=kind,
        cost_r=round(2 * cost_pct(cfg) / 100 * e / dist, 4),
        atr=a,
        peak=e,
    )


def cost_pct(cfg: dict) -> float:
    """Издержки на одну сторону сделки, % (комиссия + проскальзывание)."""
    c = cfg.get("costs") or cfg.get("backtest", {})
    return float(c.get("fee_pct", 0.0)) + float(c.get("slippage_pct", 0.0))


def trade_ok(t: Trade, cfg: dict) -> bool:
    """Отсекает сделки, где комиссии съедают слишком большую долю риска."""
    mx = cfg["risk"].get("max_cost_r", 0) or 0
    return mx <= 0 or t.cost_r <= mx


def update_trade(t: Trade, bar, cfg: dict, fee_pct: float = 0.0) -> list[str]:
    """Прогоняет одну свечу через сделку. События: tp1, tp2, sl, be, trail, timeout.

    exit_mode = fixed: 50% на TP1 (стоп в б/у), остаток на TP2.
    exit_mode = trail: 50% на TP1 (стоп в б/у), остаток ведётся трейлинг-стопом
                       (chandelier: лучшая цена − trail_atr × ATR входа), без потолка прибыли.
    Консервативно: если в одной свече задеты и стоп, и тейк — считаем стоп.
    Трейлинг двигается по свече и начинает действовать со СЛЕДУЮЩЕЙ свечи.
    """
    if t.status == "closed":
        return []
    rk = cfg["risk"]
    frac = rk["tp1_close_frac"]
    trail = rk.get("exit_mode", "fixed") == "trail"
    hi, lo, cl = float(bar["high"]), float(bar["low"]), float(bar["close"])
    now = pd.Timestamp(bar["close_time"])
    t.bars_held += 1
    t.last_checked = now.isoformat()
    ev: list[str] = []

    def hit(level, adverse):
        if t.side > 0:
            return lo <= level if adverse else hi >= level
        return hi >= level if adverse else lo <= level

    if t.status == "open":
        if hit(t.sl, True):
            t.r_gross, t.exit_reason = -1.0, "sl"
            ev.append("sl")
        elif not trail and hit(t.tp2, False):
            t.r_gross, t.exit_reason = frac * rk["tp1_r"] + (1 - frac) * rk["tp2_r"], "tp2"
            ev += ["tp1", "tp2"]
        elif hit(t.tp1, False):
            t.status = "tp1"
            t.r_gross = frac * rk["tp1_r"]
            t.sl = t.entry  # безубыток
            ev.append("tp1")
    elif t.status == "tp1":
        if hit(t.sl, True):
            t.r_gross = frac * rk["tp1_r"] + (1 - frac) * t.side * (t.sl - t.entry) / t.risk
            t.exit_reason = "trail" if abs(t.sl - t.entry) > 1e-9 * t.entry else "be"
            ev.append(t.exit_reason)
        elif not trail and hit(t.tp2, False):
            t.r_gross = frac * rk["tp1_r"] + (1 - frac) * rk["tp2_r"]
            t.exit_reason = "tp2"
            ev.append("tp2")

    if trail and not t.exit_reason:
        t.peak = max(t.peak or t.entry, hi) if t.side > 0 else min(t.peak or t.entry, lo)
        if t.status == "tp1" and t.atr > 0:
            level = t.peak - t.side * rk.get("trail_atr", 3.0) * t.atr
            t.sl = max(t.sl, level) if t.side > 0 else min(t.sl, level)

    if not t.exit_reason and now - pd.Timestamp(t.opened) >= pd.Timedelta(hours=rk["ttl_hours"]):
        remaining = 1.0 if t.status == "open" else (1 - frac)
        t.r_gross += remaining * t.side * (cl - t.entry) / t.risk
        t.exit_reason = "timeout"
        ev.append("timeout")

    if t.exit_reason:
        t.status = "closed"
        t.closed = t.last_checked
        fee_r = 2 * fee_pct / 100 * t.entry / t.risk
        t.r_gross = round(t.r_gross, 3)
        t.r_net = round(t.r_gross - fee_r, 3)
    return ev
