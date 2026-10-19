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

from . import setups
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
    # Макро-режим (DXY/доходности против альтов): см. src/macro.py
    mc = cfg.get("macro", {})
    mmode = mc.get("mode", "off")
    mb = f["macro_bias"].fillna(0) if "macro_bias" in f else pd.Series(0.0, index=f.index)
    if mmode == "penalty":
        conf = (conf - np.where(mb * direction < 0, float(mc.get("penalty", 10)), 0.0)).clip(0, 100)

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
    # v4: не догонять — трендовый вход только если цена не улетела от EMA20 и RSI не на экстремуме
    if sc.get("max_ext_atr"):
        ext = (f["close"] - f["ema20"]) / f["atr"]
        ok &= (ext * direction) <= float(sc["max_ext_atr"])
    if sc.get("rsi_cap"):
        cap = float(sc["rsi_cap"])
        ok &= ~(((direction > 0) & (f["rsi"] > cap)) | ((direction < 0) & (f["rsi"] < 100 - cap)))
    sides = sc.get("sides", "both")
    cond_long = ok & (direction > 0) & (sides != "short_only")
    cond_short = ok & (direction < 0) & (sides != "long_only")
    if sc.get("strict_shorts", False):
        # шорт только когда медвежьи и 1H монеты, и 1H BTC
        cond_short &= (htf == -1) & ((btc == -1) if not is_btc else True)
    # Фильтры «не идти за толпой» для трендовых сигналов
    cr = cfg.get("contrarian", {})
    crowd_long, crowd_short = crowd_ok(f, cfg)
    cond_long &= crowd_long
    cond_short &= crowd_short
    if mmode == "block":
        cond_long &= ~(mb < 0)   # доллар растёт при активной связи — лонги против макро
        cond_short &= ~(mb > 0)
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

    # ---- v4: пробойные сетапы (src/setups.py)
    smode = cfg.get("strategy", {}).get("mode", "trend")
    bo = None
    if smode in ("breakout", "both"):
        bo = setups.detect(f, cfg, is_btc=is_btc)
        if smode == "breakout":
            trig_long = trig_long & False
            trig_short = trig_short & False
            kind[:] = ""
        bsig = bo["bo_signal"]
        b_long = (bsig > 0) & crowd_long & ~(trig_long | trig_short)
        b_short = (bsig < 0) & crowd_short & ~(trig_long | trig_short)
        if mmode == "block":
            b_long &= ~(mb < 0)
            b_short &= ~(mb > 0)
        trig_long = trig_long | b_long
        trig_short = trig_short | b_short
        kind[b_long | b_short] = "breakout"
        conf = conf.where(~(b_long | b_short), bo["bo_score"])

    out = f.copy()
    if bo is not None:
        for col in bo.columns:
            out[col] = bo[col]
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


def crowd_ok(f: pd.DataFrame, cfg: dict) -> tuple[pd.Series, pd.Series]:
    """Фильтры «не идти за толпой»: (можно лонг, можно шорт) на каждую свечу."""
    cr = cfg.get("contrarian", {})
    ok_l = pd.Series(True, index=f.index)
    ok_s = pd.Series(True, index=f.index)
    if cr.get("crowd_filter", False) and "ls_acc_z" in f:
        z = f["ls_acc_z"].fillna(0)
        lim = cr.get("crowd_filter_z", 2.0)
        ok_l &= ~(z >= lim)    # толпа и так перегружена лонгами — не присоединяемся
        ok_s &= ~(z <= -lim)
    if cr.get("fng_filter", False) and "fng" in f:
        g = f["fng"]
        ok_l &= ~(g >= cr.get("fng_greed", 80))
        ok_s &= ~(g <= cr.get("fng_fear", 20))
    return ok_l, ok_s


def analyze(df: pd.DataFrame, htf: pd.DataFrame | None, btc_htf: pd.DataFrame | None, cfg: dict, is_btc=False,
            crowd=None, fng=None, macro=None):
    return score_frame(build_features(df, htf, None if is_btc else btc_htf, crowd, fng, macro), cfg, is_btc=is_btc)


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
    kind: str = "trend"  # trend | contra | breakout
    macro: str = ""      # with | against | neutral — положение сделки относительно макро-режима
    cost_r: float = 0.0  # издержки туда-обратно в R
    atr: float = 0.0     # ATR на входе (для трейлинга)
    peak: float = 0.0    # лучшая цена с момента входа (для трейлинга)
    status: str = "open"  # pending (ждём лимит) | open | tp1 | closed
    expires: str = ""    # pending: до какого времени ждём ретест
    filled: str = ""     # когда исполнился лимит (TTL считаем отсюда)
    level: float = 0.0   # breakout: пробитый уровень
    target: float = 0.0  # breakout: цель по высоте паттерна (measured move)
    notes: str = ""      # breakout: из чего сложилась оценка
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
    kind = str(row.get("kind", "") or "trend")
    limit = row.get("bo_limit") if kind == "breakout" else None
    pending = entry is None and limit is not None and pd.notna(limit)
    e = float(entry if entry is not None else (limit if pending else row["close"]))
    a = float(row["atr"])
    if kind == "breakout" and pd.notna(row.get("bo_stop")):
        dist = side * (e - float(row["bo_stop"]))  # стоп за базой паттерна
    elif side > 0:
        struct = float(row["swing_lo"]) - rk["sl_swing_buffer_atr"] * a
        dist = e - struct
    else:
        struct = float(row["swing_hi"]) + rk["sl_swing_buffer_atr"] * a
        dist = struct - e
    dist = float(np.clip(dist, rk["sl_min_atr"] * a, rk["sl_max_atr"] * a))
    dist = min(dist, e * rk["sl_max_pct"] / 100)
    dist = max(dist, e * rk.get("sl_min_pct", 0.0) / 100)  # слишком узкий стоп съедается комиссией
    sl = e - side * dist
    if kind == "breakout":
        reasons = []
    else:
        pool = CONTRARIAN if kind == "contra" else [m for m in MODULES if m not in CONTRARIAN]
        reasons = [m for m in pool if ("v_" + m) in row and row["v_" + m] * side > 0.3]
    mb = row.get("macro_bias", 0)
    mb = 0.0 if mb is None or pd.isna(mb) else float(mb)
    macro = "" if cfg.get("macro", {}).get("mode", "off") == "off" else \
        ("with" if mb * side > 0 else "against" if mb * side < 0 else "neutral")
    opened = pd.Timestamp(row["close_time"])
    extra = {}
    if kind == "breakout":
        from .config import TF_MINUTES

        bc = setups.bcfg(cfg)
        extra = {"level": float(row["bo_level"]), "target": float(row["bo_target"]),
                 "notes": str(row.get("bo_notes", "") or "")}
        if pending:
            tfm = TF_MINUTES.get(cfg["timeframes"]["entry"], 60)
            extra.update(status="pending",
                         expires=(opened + pd.Timedelta(minutes=tfm * int(bc["retest_bars"]))).isoformat())
    return Trade(
        symbol=symbol,
        side=side,
        entry=e,
        sl=sl,
        tp1=e + side * rk["tp1_r"] * dist,
        tp2=e + side * rk["tp2_r"] * dist,
        risk=dist,
        opened=opened.isoformat(),
        conf=round(float(row["conf"]), 1),
        regime=str(row["regime"]),
        reasons=reasons,
        kind=kind,
        macro=macro,
        cost_r=round(2 * cost_pct(cfg) / 100 * e / dist, 4),
        atr=a,
        peak=e,
        **extra,
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
    """Прогоняет одну свечу через сделку. События: filled, expired, tp1, tp2, sl, be, trail, timeout.

    pending (лимит на ретест уровня): исполняется, когда цена вернулась к лимиту. В свече исполнения
    проверяется только стоп (консервативно: тейк в той же свече не засчитываем). Если до истечения
    цена так и не вернулась или ушла до TP1 без нас — expired (в статистику не идёт).

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
    t.last_checked = now.isoformat()
    ev: list[str] = []

    def hit(level, adverse):
        if t.side > 0:
            return lo <= level if adverse else hi >= level
        return hi >= level if adverse else lo <= level

    if t.status == "pending":
        if hit(t.entry, True):
            t.status, t.filled, t.peak = "open", now.isoformat(), t.entry
            ev.append("filled")
            if hit(t.sl, True):
                t.r_gross, t.exit_reason = -1.0, "sl"
                ev.append("sl")
            else:
                return ev
        elif hit(t.tp1, False) or (t.expires and now >= pd.Timestamp(t.expires)):
            t.status, t.exit_reason, t.closed = "closed", "expired", now.isoformat()
            t.r_gross = t.r_net = 0.0
            return ["expired"]
        else:
            return ev
    t.bars_held += 1

    if t.status == "open" and not t.exit_reason:
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

    if not t.exit_reason and now - pd.Timestamp(t.filled or t.opened) >= pd.Timedelta(hours=rk["ttl_hours"]):
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
