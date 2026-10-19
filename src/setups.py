"""v4: пробойные сетапы — бот ищет паттерн ЗАРАНЕЕ и входит только на подтверждённом пробое.

Почему: трендовый консенсус v3 срабатывал, когда движение уже случилось (все индикаторы «за»
только после сильного импульса) — вход на хаях/лоях, стоп выносило откатом.

Что ищем (лонг; шорт — зеркально):
  • База: последние base_bars свечей, высота (верх − низ) в разумных пределах по ATR.
  • Плоское сопротивление: ≥ min_touches подтверждённых локальных максимумов у верха базы
    (в пределах touch_tol_atr × ATR), первый и последний — с разносом ≥ min_span_bars.
  • Повышающиеся минимумы (восходящий треугольник / флаг): минимум второй половины базы выше первой.
  • Сжатие волатильности: ширина полос Боллинджера в нижних перцентилях за bbw_lookback свечей.
  • Затухание объёма в базе: объём последней трети базы ниже среднего по базе.
  • Контекст: тренд старшего ТФ не против, цена не «улетела» от EMA50 (не догоняем).
Пробой — закрытие 1H свечи за уровнем с объёмом (vol_z ≥ vol_z_min) и сильным закрытием свечи.
Если свеча закрылась далеко за уровнем (> chase_max_atr × ATR) — не догоняем, а ставим лимит
на ретест уровня (действует retest_bars свечей).

Оценка 0–100 складывается из компонентов (см. SCORE). Сигнал — только при оценке ≥ min_score.

Всё каузально: на баре i используются только свечи ≤ i (локальные максимумы считаются
подтверждёнными через 2 свечи).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view as swv

DEFAULT = {
    "base_bars": 36,          # окно базы (1H → 1,5 суток)
    "min_span_bars": 10,      # разнос первого и последнего касания уровня
    "touch_tol_atr": 0.3,     # касание = локальный максимум не дальше 0.3 ATR от верха базы
    "min_touches": 2,
    "min_height_atr": 1.5,    # слишком узкая база — шум
    "max_height_atr": 8.0,    # слишком широкая — это уже не база, а диапазон/разворот
    "bbw_lookback": 120,
    "vol_z_min": 1.0,         # объём свечи пробоя
    "close_pos_min": 0.5,     # закрытие в верхней половине свечи (для шорта — в нижней)
    "close_buffer_atr": 0.05, # закрыться за уровнем хотя бы на 0.05 ATR
    "max_ext_atr": 4.0,       # (цена − EMA50) / ATR на пробое — дальше не входим (перегрев)
    "chase_max_atr": 0.5,     # свеча ушла за уровень дальше — лимит на ретест
    "entry_mode": "auto",     # auto | market | retest
    "retest_bars": 6,
    "retest_offset_atr": 0.1, # лимит чуть выше уровня (для лонга), чтобы его реально наливали
    "stop_buffer_atr": 0.25,
    "min_score": 75,
    "alert_score": 70,        # предупреждение «готовится пробой» (оценка без компонентов пробоя)
    "alert_near_atr": 1.0,    # цена не дальше 1 ATR от уровня
    "require_htf": True,      # тренд старшего ТФ не против
}

# Максимум баллов по компонентам (сумма = 100)
SCORE = {
    "touches": 20,      # 2 касания → 10, 3 → 16, ≥4 → 20
    "compression": 15,  # BB-ширина ≤ 20-го перцентиля → 15, ≤ 35-го → 8
    "higher_lows": 15,  # минимумы растут ≥ 0.5 ATR → 15, почти плоские → 5
    "dryup": 10,        # объём в конце базы ≤ 75% среднего → 10, ≤ 90% → 5
    "duration": 5,      # разнос касаний ≥ 20 свечей → 5
    "bo_volume": 12,    # объём пробоя vol_z ≥ 2 → 12, ≥ 1 → 6
    "bo_close": 5,      # закрытие в верхних 25% свечи
    "htf": 10,          # старший ТФ по направлению → 10, нейтрален → 3
    "btc": 8,           # BTC 4H по направлению → 8, нейтрален → 3, против → −10
}
NOTES_RU = {
    "touches": "{n} {w} уровня",
    "compression": "сжатие волатильности",
    "higher_lows": "повышающиеся минимумы",
    "lower_highs": "понижающиеся максимумы",
    "dryup": "объём затухал в базе",
    "bo_volume": "объём на пробое ×{v:.1f} к норме",
    "htf": "старший ТФ по направлению",
    "btc": "BTC по направлению",
}


def bcfg(cfg: dict) -> dict:
    return {**DEFAULT, **(cfg.get("breakout") or {})}


def _windows(a: np.ndarray, w: int) -> np.ndarray:
    """Строка i — значения a[i-w .. i-1] (только прошлые свечи). Первые w строк — NaN."""
    n = len(a)
    out = np.full((n, w), np.nan)
    if n > w:
        out[w:] = swv(a.astype(float), w)[:n - w]
    return out


def _pivots(x: np.ndarray, kind: str) -> np.ndarray:
    """Локальный экстремум 5 свечей (±2). Флаг стоит на самом пивоте; известен он через 2 свечи."""
    s = pd.Series(x)
    ext = s.rolling(5, center=True, min_periods=5).max() if kind == "high" else \
        s.rolling(5, center=True, min_periods=5).min()
    return (s == ext).to_numpy()


def _side(f: pd.DataFrame, c: dict, side: int) -> dict[str, np.ndarray]:
    """Все компоненты для одной стороны. Для шорта цены зеркалим (−low как high)."""
    W = int(c["base_bars"])
    if side > 0:
        hi, lo = f["high"].to_numpy(float), f["low"].to_numpy(float)
    else:
        hi, lo = -f["low"].to_numpy(float), -f["high"].to_numpy(float)
    op = f["open"].to_numpy(float) * side
    cl = f["close"].to_numpy(float) * side
    atr = f["atr"].to_numpy(float)
    n = len(f)

    H, L = _windows(hi, W), _windows(lo, W)
    with np.errstate(invalid="ignore"):
        top = np.nanmax(H, axis=1) if n > W else np.full(n, np.nan)
        bot = np.nanmin(L, axis=1) if n > W else np.full(n, np.nan)
    top[:W] = np.nan
    bot[:W] = np.nan
    height = (top - bot) / atr

    # касания: подтверждённые пивоты (j ≤ i-3) у верха базы
    piv = _pivots(hi, "high").astype(float)
    P = _windows(piv, W)
    P[:, W - 2:] = 0.0                     # последние 2 свечи окна ещё не подтверждены
    near = H >= (top - c["touch_tol_atr"] * atr)[:, None]
    tmask = (P > 0) & near
    # соседние пивоты на «полке» — одно касание: считаем, только если 2 предыдущие свечи без касания
    p1 = np.zeros_like(tmask)
    p2 = np.zeros_like(tmask)
    p1[:, 1:], p2[:, 2:] = tmask[:, :-1], tmask[:, :-2]
    tmask = tmask & ~p1 & ~p2
    touches = tmask.sum(axis=1)
    idx = np.arange(W)[None, :]
    first = np.where(tmask, idx, W).min(axis=1)
    last = np.where(tmask, idx, -1).max(axis=1)
    span = np.where(touches > 0, last - first, 0)

    # повышающиеся минимумы: минимум второй половины окна против первой
    half = W // 2
    with np.errstate(invalid="ignore"):
        lo1 = np.nanmin(L[:, :half], axis=1) if n > W else np.full(n, np.nan)
        lo2 = np.nanmin(L[:, half:], axis=1) if n > W else np.full(n, np.nan)
        stop_ref = np.nanmin(L[:, -max(4, W // 3):], axis=1) if n > W else np.full(n, np.nan)
    hl = (lo2 - lo1) / atr

    # затухание объёма
    V = _windows(f["volume"].to_numpy(float), W)
    with np.errstate(invalid="ignore", divide="ignore"):
        dry = np.nanmean(V[:, -max(4, W // 3):], axis=1) / np.nanmean(V, axis=1)

    # сжатие: перцентиль ширины Боллинджера на ПРЕДЫДУЩЕЙ свече (до пробоя)
    bbw = ((f["bb_up"] - f["bb_lo"]) / f["bb_mid"]).shift(1)
    lb = int(c["bbw_lookback"])
    bbw_pct = bbw.rolling(lb, min_periods=lb // 2).rank(pct=True).to_numpy()

    rng_ = np.maximum(hi - lo, 1e-12)
    close_pos = (cl - lo) / rng_ if side > 0 else (cl - lo) / rng_  # в зеркале одинаково
    ext = (cl - f["ema50"].to_numpy(float) * side) / atr

    return {"top": top, "bot": bot, "height": height, "touches": touches, "span": span, "hl": hl,
            "dry": dry, "bbw_pct": bbw_pct, "close_pos": close_pos, "ext": ext, "cl": cl, "op": op,
            "stop_ref": stop_ref}


def _pts(cond_vals: list[tuple[np.ndarray, float]], n: int) -> np.ndarray:
    out = np.zeros(n)
    taken = np.zeros(n, dtype=bool)
    for cond, v in cond_vals:
        m = np.nan_to_num(cond, nan=0).astype(bool) & ~taken
        out[m] = v
        taken |= m
    return out


def detect(f: pd.DataFrame, cfg: dict, is_btc: bool = False) -> pd.DataFrame:
    """Колонки сетапа на каждую свечу (см. описание модуля).

    bo_signal  +1/-1 — подтверждённый пробой с оценкой ≥ min_score на этой свече
    bo_score   оценка пробоя 0..100 (для стороны лучшего кандидата)
    bo_armed   +1/-1 — сетап сформирован, цена у уровня, пробоя ещё нет (оценка ≥ alert_score)
    bo_pre     оценка сетапа без компонентов пробоя (для предупреждения), 0..100
    bo_level, bo_base, bo_stop, bo_target, bo_limit (NaN — вход по рынку), bo_touches, bo_notes
    """
    c = bcfg(cfg)
    n = len(f)
    atr = f["atr"].to_numpy(float)
    htf = f["htf_bias"].fillna(0).to_numpy(float)
    btc = np.zeros(n) if is_btc else f["btc_htf_bias"].fillna(0).to_numpy(float)
    vz = f["vol_z"].to_numpy(float)
    sides_cfg = cfg.get("signals", {}).get("sides", "both")
    per = {}
    for side in (1, -1):
        if (side > 0 and sides_cfg == "short_only") or (side < 0 and sides_cfg == "long_only"):
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # пустые окна в начале истории
            s = _side(f, c, side)
        tch, span = s["touches"], s["span"]
        base_ok = ((tch >= c["min_touches"]) & (span >= c["min_span_bars"])
                   & (s["height"] >= c["min_height_atr"]) & (s["height"] <= c["max_height_atr"])
                   & np.isfinite(atr) & (atr > 0) & f["ema200"].notna().to_numpy())
        d_htf, d_btc = htf * side, btc * side
        if c.get("require_htf", True):
            base_ok &= d_htf >= 0

        p_touch = _pts([(tch >= 4, SCORE["touches"]), (tch == 3, 16), (tch == 2, 10)], n)
        p_comp = _pts([(s["bbw_pct"] <= 0.20, SCORE["compression"]), (s["bbw_pct"] <= 0.35, 8)], n)
        p_hl = _pts([(s["hl"] >= 0.5, SCORE["higher_lows"]), (s["hl"] >= -0.25, 5)], n)
        p_dry = _pts([(s["dry"] <= 0.75, SCORE["dryup"]), (s["dry"] <= 0.9, 5)], n)
        p_dur = _pts([(span >= 20, SCORE["duration"])], n)
        p_htf = _pts([(d_htf > 0, SCORE["htf"]), (d_htf == 0, 3)], n)
        p_btc = np.full(n, float(SCORE["btc"])) if is_btc else \
            _pts([(d_btc > 0, SCORE["btc"]), (d_btc == 0, 3), (d_btc < 0, -10)], n)
        pre_raw = p_touch + p_comp + p_hl + p_dry + p_dur + p_htf + p_btc
        pre = np.clip(pre_raw / (100 - SCORE["bo_volume"] - SCORE["bo_close"]) * 100, 0, 100)
        p_vol = _pts([(vz >= 2.0, SCORE["bo_volume"]), (vz >= 1.0, 6)], n)
        p_cls = _pts([(s["close_pos"] >= 0.75, SCORE["bo_close"])], n)
        score = np.clip(pre_raw + p_vol + p_cls, 0, 100)

        level = s["top"]
        not_hot = s["ext"] <= c["max_ext_atr"]
        broke = (s["cl"] > level + c["close_buffer_atr"] * atr) & (s["close_pos"] >= c["close_pos_min"]) \
            & (vz >= c["vol_z_min"]) & not_hot
        sig = base_ok & broke & (score >= c["min_score"])
        dist = (level - s["cl"]) / atr
        armed = base_ok & (dist >= 0) & (dist <= c["alert_near_atr"]) & (pre >= c["alert_score"]) & not_hot

        chase = (s["cl"] - level) / atr
        mode = c.get("entry_mode", "auto")
        use_limit = (mode == "retest") | ((mode == "auto") & (chase > c["chase_max_atr"]))
        lvl_px = level * side
        per[side] = {
            "sig": sig, "armed": armed, "ok": base_ok, "score": score, "pre": np.where(base_ok, pre, -1.0),
            "bo_level": lvl_px, "bo_base": s["bot"] * side,
            "bo_stop": s["stop_ref"] * side - side * c["stop_buffer_atr"] * atr,
            "bo_target": lvl_px + side * (s["top"] - s["bot"]),
            "bo_limit": np.where(use_limit, lvl_px + side * c["retest_offset_atr"] * atr, np.nan),
            "bo_touches": tch.astype(float), "bo_hl": p_hl >= SCORE["higher_lows"],
            "parts": (p_comp, p_hl, p_dry, p_htf, p_btc, tch),
        }

    cols = ("bo_level", "bo_base", "bo_stop", "bo_target", "bo_limit", "bo_touches", "bo_hl")
    res = {k: np.full(n, np.nan) for k in cols + ("bo_score", "bo_pre")}
    res["bo_hl"] = np.zeros(n, dtype=bool)
    res["bo_signal"] = np.zeros(n, dtype=int)
    res["bo_armed"] = np.zeros(n, dtype=int)
    notes = np.empty(n, dtype=object)
    notes[:] = ""
    L_, S_ = per.get(1), per.get(-1)
    zero = np.zeros(n, dtype=bool)
    # какая сторона «владеет» свечой: сигнал > предупреждение > лучшая оценка сетапа
    sig_l = L_["sig"] if L_ else zero
    sig_s = (S_["sig"] if S_ else zero) & ~sig_l
    pre_l = L_["pre"] if L_ else np.full(n, -1.0)
    pre_s = S_["pre"] if S_ else np.full(n, -1.0)
    own_l = sig_l | (~sig_s & (pre_l >= pre_s) & (pre_l >= 0))
    own_s = ~own_l & (sig_s | (pre_s >= 0))
    for side, own, d in ((1, own_l, L_), (-1, own_s, S_)):
        if d is None:
            continue
        for k in cols:
            res[k] = np.where(own, d[k], res[k])
        res["bo_score"] = np.where(own, d["score"], res["bo_score"])
        res["bo_pre"] = np.where(own, d["pre"], res["bo_pre"])
        res["bo_signal"] = np.where(own & d["sig"], side, res["bo_signal"])
        res["bo_armed"] = np.where(own & d["armed"] & ~d["sig"], side, res["bo_armed"])
        p_comp, p_hl, p_dry, p_htf, p_btc, tch = d["parts"]
        for i in np.flatnonzero(own & (d["sig"] | d["armed"])):
            k_ = int(tch[i])
            bits = [NOTES_RU["touches"].format(n=k_, w="касания" if 2 <= k_ % 10 <= 4 and k_ // 10 != 1 else "касаний")]
            if p_comp[i] >= SCORE["compression"]:
                bits.append(NOTES_RU["compression"])
            if p_hl[i] >= SCORE["higher_lows"]:
                bits.append(NOTES_RU["higher_lows" if side > 0 else "lower_highs"])
            if p_dry[i] >= 5:
                bits.append(NOTES_RU["dryup"])
            if d["sig"][i] and np.isfinite(vz[i]) and vz[i] >= 1:
                prev = f["volume"].iloc[max(0, i - 50):i].mean()
                bits.append(NOTES_RU["bo_volume"].format(v=f["volume"].iloc[i] / max(prev, 1e-12)))
            if p_htf[i] >= SCORE["htf"]:
                bits.append(NOTES_RU["htf"])
            if not is_btc and p_btc[i] >= SCORE["btc"]:
                bits.append(NOTES_RU["btc"])
            notes[i] = " · ".join(bits)
    res["bo_pre"] = np.where(res["bo_pre"] < 0, np.nan, res["bo_pre"])

    out = pd.DataFrame(res, index=f.index)
    out["bo_notes"] = notes
    return out


def pattern_name(side: int, hl: bool) -> str:
    if side > 0:
        return "восходящий треугольник" if hl else "пробой диапазона вверх"
    return "нисходящий треугольник" if hl else "пробой диапазона вниз"
