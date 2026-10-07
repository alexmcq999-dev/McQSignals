"""Оценочная карта ликвидаций (как «Liquidation Map» у Coinglass, но на открытых данных).

Это МОДЕЛЬ, а не фактические позиции: биржи не публикуют, у кого какое плечо. Логика:
  1. Открытый интерес Binance USDⓈ-M по часам (data.binance.vision, 5-минутные metrics).
     Рост OI за час = открылись новые позиции примерно по цене этого часа.
     Доля лонгов среди новых позиций — по соотношению агрессивных покупок и продаж (taker buy/sell) за час.
     Падение OI = часть позиций закрыта → все «живые» уровни пропорционально уменьшаются.
  2. Новые позиции раскладываются по типичным плечам (5x…100x). Цена ликвидации:
       лонг  ≈ P × (1 − 1/L + mmr),  шорт ≈ P × (1 + 1/L − mmr).
  3. Если позже цена прошла уровень — эти позиции уже ликвидированы и убираются с карты.
  4. Итог — объём $ потенциальных ликвидаций по ценовым уровням вокруг текущей цены.
     Крупные скопления работают как «магниты»: маркет-мейкерам выгодно до них дотянуть цену.
Ограничение: файлы metrics публикуются на следующий день, поэтому позиции последних ~суток на карте не видны.
"""
from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from . import crowd as cw
from .config import STATE_DIR

log = logging.getLogger("liqmap")

DEFAULT = {
    "symbols": ["BTC", "ETH", "SOL", "XRP", "DOGE", "BNB"],
    "days": 14,
    "range_pct": 12.0,
    "bins": 48,
    "mmr": 0.005,
    "leverage": {5: 0.10, 10: 0.30, 20: 0.25, 50: 0.20, 100: 0.15},
}


def _cfg(cfg: dict) -> dict:
    c = {**DEFAULT, **(cfg.get("liqmap") or {})}
    c["leverage"] = {int(k): float(v) for k, v in c["leverage"].items()}
    return c


# ------------------------------------------------------------------ данные OI (кэш по часам)
def _hourly(df: pd.DataFrame) -> dict[str, list]:
    t = pd.to_datetime(df["create_time"], utc=True, errors="coerce")
    oi = pd.to_numeric(df["sum_open_interest_value"], errors="coerce")
    tk = pd.to_numeric(df.get("sum_taker_long_short_vol_ratio"), errors="coerce")
    x = pd.DataFrame({"t": t, "oi": oi, "tk": tk}).dropna(subset=["t", "oi"]).set_index("t").sort_index()
    h = pd.DataFrame({"oi": x["oi"].resample("1h").last(), "tk": x["tk"].resample("1h").mean()}).dropna(subset=["oi"])
    return {ts.strftime("%Y-%m-%dT%H"): [round(float(r.oi), 0), None if pd.isna(r.tk) else round(float(r.tk), 4)]
            for ts, r in h.iterrows()}


def update_oi(bases: list[str], now: pd.Timestamp, days: int, fsyms: dict[str, str]) -> dict:
    """Докачивает недостающие дни 5-минутного OI и хранит почасовые значения в state/liq_oi.json."""
    path = STATE_DIR / "liq_oi.json"
    try:
        cache = json.loads(path.read_text()) if path.exists() else {}
    except json.JSONDecodeError:
        cache = {}
    want = cw.recent_days(now, days)
    jobs = []
    for b in bases:
        fs = fsyms.get(b) or f"{b}USDT"
        ent = cache.setdefault(b, {"fsym": fs, "days": [], "missing": [], "h": {}})
        ent["fsym"] = fs
        jobs += [(b, fs, d) for d in want if d not in ent["days"] and d not in ent["missing"]]

    def one(j):
        b, fs, d = j
        return b, d, cw.fetch_raw(fs, d)

    if jobs:
        log.info("Карта ликвидаций: докачка %d дневных файлов OI", len(jobs))
        with ThreadPoolExecutor(12) as ex:
            for b, d, df in ex.map(one, jobs):
                if df is None or "sum_open_interest_value" not in df:
                    cache[b]["missing"].append(d)
                    continue
                cache[b]["h"].update(_hourly(df))
                cache[b]["days"].append(d)
    edge = (now - pd.Timedelta(days=days + 1)).strftime("%Y-%m-%dT%H")
    for b, ent in cache.items():
        ent["h"] = {k: v for k, v in sorted(ent["h"].items()) if k >= edge}
        ent["days"] = sorted(d for d in ent["days"] if d in want)
        ent["missing"] = [d for d in ent["missing"] if d in want][-days:]
    cache = {b: v for b, v in cache.items() if b in bases}
    STATE_DIR.mkdir(exist_ok=True)
    path.write_text(json.dumps(cache, separators=(",", ":")))
    return cache


# ------------------------------------------------------------------ модель
def build_levels(oi_h: dict, price_1h: pd.DataFrame, c: dict) -> list[dict]:
    """Живые уровни ликвидаций: [{'price', 'usd', 'side'}]. price_1h — 1H свечи (time, high, low, close)."""
    if not oi_h or price_1h is None or price_1h.empty:
        return []
    px = price_1h.set_index(pd.to_datetime(price_1h["time"], utc=True))[["high", "low", "close"]]
    oi = pd.DataFrame.from_dict(oi_h, orient="index", columns=["oi", "tk"])
    oi.index = pd.to_datetime(oi.index, format="%Y-%m-%dT%H", utc=True)
    oi = oi.sort_index()
    lev, mmr = c["leverage"], c["mmr"]
    levels: list[list] = []  # [liq_price, usd, side]
    prev = None
    for ts, row in oi.iterrows():
        if prev is None:
            prev = row["oi"]
            continue
        d = row["oi"] - prev
        p = px["close"].get(ts)
        if p is None or np.isnan(p):
            prev = row["oi"]
            continue
        if d > 0:
            tk = row["tk"] if row["tk"] and not np.isnan(row["tk"]) else 1.0
            long_frac = tk / (1 + tk)
            for L, w in lev.items():
                levels.append([p * (1 - 1 / L + mmr), d * long_frac * w, 1])
                levels.append([p * (1 + 1 / L - mmr), d * (1 - long_frac) * w, -1])
        elif d < 0 and prev > 0:
            k = max(0.0, 1 + d / prev)
            for lv in levels:
                lv[1] *= k
        prev = row["oi"]
        # ликвидации внутри следующего часа: уровни, которые цена уже прошла, убираем
        bar = px.loc[ts + pd.Timedelta(hours=1):ts + pd.Timedelta(hours=1)]
        if not bar.empty:
            hi, lo = float(bar["high"].iloc[0]), float(bar["low"].iloc[0])
            levels = [lv for lv in levels if not ((lv[2] > 0 and lo <= lv[0]) or (lv[2] < 0 and hi >= lv[0]))]
    # после последнего часа OI — прогоняем все более свежие свечи (позиции последних суток не видны, но
    # уже снятые ликвидации убрать можно)
    last_oi = oi.index[-1] if len(oi) else None
    if last_oi is not None:
        tail = px.loc[last_oi + pd.Timedelta(hours=2):]
        if not tail.empty:
            hi, lo = float(tail["high"].max()), float(tail["low"].min())
            levels = [lv for lv in levels if not ((lv[2] > 0 and lo <= lv[0]) or (lv[2] < 0 and hi >= lv[0]))]
    return [{"price": lv[0], "usd": lv[1], "side": lv[2]} for lv in levels if lv[1] > 0]


def histogram(levels: list[dict], price: float, c: dict, oi_last: float | None = None) -> dict:
    rng, n = c["range_pct"] / 100, int(c["bins"])
    lo, hi = price * (1 - rng), price * (1 + rng)
    edges = np.linspace(lo, hi, n + 1)
    longs = np.zeros(n)
    shorts = np.zeros(n)
    for lv in levels:
        i = np.searchsorted(edges, lv["price"]) - 1
        if 0 <= i < n:
            (longs if lv["side"] > 0 else shorts)[i] += lv["usd"]
    mids = (edges[:-1] + edges[1:]) / 2

    def clusters(arr, side, k=3):
        idx = [i for i in np.argsort(arr)[::-1] if arr[i] > 0][:k]
        return [{"price": float(mids[i]), "usd": float(arr[i]), "dist_pct": round((mids[i] / price - 1) * 100, 2),
                 "side": side} for i in sorted(idx, key=lambda i: abs(mids[i] - price))]

    near = abs(mids / price - 1) <= 0.05
    return {
        "price": price,
        "bins": [round(float(x), 6) for x in mids],
        "long_usd": [round(float(x)) for x in longs],
        "short_usd": [round(float(x)) for x in shorts],
        "clusters_below": clusters(np.where(mids < price, longs, 0), "long"),
        "clusters_above": clusters(np.where(mids > price, shorts, 0), "short"),
        "within5_long": round(float(longs[near & (mids < price)].sum())),
        "within5_short": round(float(shorts[near & (mids > price)].sum())),
        "oi_usd": oi_last,
    }


def build_all(cfg: dict, now: pd.Timestamp, price_1h: dict[str, pd.DataFrame], prices: dict[str, float]) -> dict:
    """Карта для всех символов из конфига. price_1h — 1H свечи (их бот уже скачал для сигналов)."""
    c = _cfg(cfg)
    syms = [s for s in c["symbols"] if s in price_1h]
    crowd_cache = cw._load_cache(STATE_DIR / "crowd.json")
    fsyms = {b: (crowd_cache.get(b) or {}).get("fsym") or f"{b}USDT" for b in syms}
    oi = update_oi(syms, now, int(c["days"]), fsyms)
    out = {}
    for s in syms:
        h = (oi.get(s) or {}).get("h") or {}
        lv = build_levels(h, price_1h[s], c)
        if not lv:
            continue
        last_oi = list(h.values())[-1][0] if h else None
        last_day = max((oi.get(s) or {}).get("days") or [""])
        out[s] = {**histogram(lv, prices.get(s) or float(price_1h[s]["close"].iloc[-1]), c, last_oi),
                  "data_until": last_day}
    return {"updated": now.isoformat(), "model": "OI Binance + плечи 5–100x", "symbols": out}


def fmt_usd(v: float) -> str:
    return f"${v / 1e9:.2f}B" if v >= 1e9 else f"${v / 1e6:.0f}M" if v >= 1e6 else f"${v / 1e3:.0f}K"


def brief_line(m: dict, sym: str = "BTC") -> str:
    s = (m or {}).get("symbols", {}).get(sym)
    if not s:
        return ""
    up = max(s["clusters_above"], key=lambda x: x["usd"], default=None)
    dn = max(s["clusters_below"], key=lambda x: x["usd"], default=None)
    def px(v):
        return f"{v:,.0f}".replace(",", " ") if v >= 1000 else f"{v:.4g}"

    parts = []
    if up:
        parts.append(f"шорты выше {px(up['price'])} ({up['dist_pct']:+.1f}%, ~{fmt_usd(up['usd'])})")
    if dn:
        parts.append(f"лонги ниже {px(dn['price'])} ({dn['dist_pct']:+.1f}%, ~{fmt_usd(dn['usd'])})")
    return f"{sym}: " + "; ".join(parts) if parts else ""
