"""Макро-режим: доллар (DXY) и доходности против альтов.

Идея: рост DXY давит на крипту, но не всегда. Связь включается и выключается
(на 26 неделях корреляция ходит от −0.9 до +0.9). Поэтому фильтр работает как
РЕЖИМ, а не как закон:

  1. Корзина альтов (топ по ликвидности без BTC/ETH, равные веса) — прокси TOTAL3.
  2. Корреляция НЕДЕЛЬНЫХ лог-доходностей драйвера (DXY) и корзины за N недель.
     Считаем по доходностям, а не по ценам: корреляция уровней цены почти всегда
     высокая просто потому, что обе линии в тренде (ложная корреляция).
  3. Тренд драйвера: недельное закрытие против EMA(10 нед.) с мёртвой зоной.
  4. Фильтр активен, только если корреляция ≤ corr_on (по умолчанию −0.5).
     Тогда: драйвер растёт → альтам встречный ветер (лонги против макро),
            драйвер падает → попутный (шорты против макро).
     Если связи нет — фильтр молчит.

Режимы (config.yaml → macro.mode):
  off     — не считать;
  shadow  — считать, показывать в сигнале и сохранять в сделке (macro: with/against),
            но НЕ влиять на сигналы. Так можно накопить статистику, не ломая тест;
  penalty — минус `penalty` к силе сигнала против макро;
  block   — запрет сигналов против макро.

Без заглядывания вперёд: неделя (пн–вс) становится известна только после закрытия
воскресной дневной свечи, т.е. с понедельника 00:00 UTC.

Источники (бесплатно, без ключей): Yahoo Finance (DX-Y.NYB, ^TNX), резерв — FRED
(DTWEXBGS — широкий индекс доллара ФРС, DGS10). Свечи корзины — тот же провайдер,
что и у сканера (Binance vision / KuCoin / OKX).
"""
from __future__ import annotations

import io
import json
import logging
import time

import numpy as np
import pandas as pd

from . import config as _cfg
from .data import HTTP, fetch_many

log = logging.getLogger("macro")

YAHOO_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{t}"
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"}
EXCLUDE_FROM_BASKET = {"BTC", "ETH"}


def enabled_drivers(cfg: dict) -> list[dict]:
    return [d for d in cfg.get("macro", {}).get("drivers", []) if d.get("enabled", True)]


def active(cfg: dict) -> bool:
    return cfg.get("macro", {}).get("mode", "off") != "off" and bool(enabled_drivers(cfg))


def warmup_days(cfg: dict) -> int:
    mc = cfg.get("macro", {})
    ema = max([int(d.get("trend_ema_weeks", 10)) for d in enabled_drivers(cfg)] or [10])
    return (int(mc.get("corr_weeks", 26)) + 3 * ema + 6) * 7


# ------------------------------------------------------------------ загрузка

def _daily_index(ts) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(ts, utc=True)).normalize()


def _yahoo(ticker: str, days: int) -> pd.Series | None:
    end = int(time.time())
    p = {"period1": end - days * 86400, "period2": end, "interval": "1d", "includePrePost": "false"}
    r = HTTP.get(YAHOO_URL.format(t=ticker), params=p, headers=UA, timeout=20)
    r.raise_for_status()
    res = r.json()["chart"]["result"][0]
    ts = res.get("timestamp") or []
    close = res["indicators"]["quote"][0]["close"]
    if not ts:
        return None
    # дневные бары Yahoo помечены временем открытия торгов в Нью-Йорке → берём дату по NY
    idx = pd.to_datetime(ts, unit="s", utc=True).tz_convert("America/New_York").normalize().tz_localize(None)
    s = pd.Series(close, index=pd.DatetimeIndex(idx).tz_localize("UTC"), dtype=float).dropna()
    return s[~s.index.duplicated(keep="last")]


def _fred(series_id: str, days: int) -> pd.Series | None:
    start = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)).strftime("%Y-%m-%d")
    r = HTTP.get(FRED_URL, params={"id": series_id, "cosd": start}, headers=UA, timeout=20)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    s = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    s.index = _daily_index(df.iloc[:, 0])
    return s.dropna()


def fetch_driver(d: dict, days: int) -> pd.Series | None:
    """Дневные закрытия драйвера: Yahoo → FRED. None, если оба недоступны."""
    for src, fn, key in (("Yahoo", _yahoo, "yahoo"), ("FRED", _fred, "fred")):
        if not d.get(key):
            continue
        try:
            s = fn(d[key], days)
            if s is not None and len(s) > 60:
                log.info("Макро %s: %d дней (%s %s)", d["name"], len(s), src, d[key])
                return s
        except Exception as e:  # noqa: BLE001
            log.warning("Макро %s из %s недоступен: %s", d["name"], src, e)
    return None


def basket_bases(bases: list[str], n: int) -> list[str]:
    return [b for b in bases if b not in EXCLUDE_FROM_BASKET][:n]


def basket_index(frames: dict[str, pd.DataFrame], min_coins: int = 3) -> pd.Series | None:
    """Равновзвешенный индекс корзины из дневных свечей (прокси TOTAL3)."""
    rets = {}
    for b, df in frames.items():
        if df is None or len(df) < 30:
            continue
        s = pd.Series(df["close"].astype(float).to_numpy(), index=_daily_index(df["time"]))
        s = s[~s.index.duplicated(keep="last")]
        rets[b] = np.log(s).diff()
    if not rets:
        return None
    R = pd.DataFrame(rets).sort_index()
    r = R.mean(axis=1, skipna=True).where(R.notna().sum(axis=1) >= min_coins)
    r = r.dropna()
    if r.empty:
        return None
    return np.exp(r.cumsum())


# ------------------------------------------------------------------ расчёт

def build_table(drivers: dict[str, pd.Series], basket: pd.Series | None, cfg: dict) -> pd.DataFrame | None:
    """Недельная таблица: usable_time, macro_bias и по каждому драйверу corr / trend / active."""
    if basket is None or not drivers:
        return None
    mc = cfg.get("macro", {})
    n = int(mc.get("corr_weeks", 26))
    thr = float(mc.get("corr_on", -0.5))
    bw = basket.resample("W-SUN").last()
    rb = np.log(bw).diff()
    out = pd.DataFrame(index=bw.index)
    bias = pd.Series(0.0, index=bw.index)
    for d in enabled_drivers(cfg):
        s = drivers.get(d["name"])
        if s is None or s.empty:
            continue
        sw = s.resample("W-SUN").last().reindex(bw.index).ffill(limit=1)
        rs = np.log(sw).diff()
        corr = rs.rolling(n, min_periods=n).corr(rb)
        ema = sw.ewm(span=int(d.get("trend_ema_weeks", 10)), adjust=False).mean()
        dev = sw / ema - 1
        band = float(d.get("trend_band_pct", 0.5)) / 100
        trend = pd.Series(np.where(dev > band, 1, np.where(dev < -band, -1, 0)), index=bw.index).where(sw.notna(), 0)
        on = corr <= thr
        nm = d["name"]
        out[f"macro_{nm}_corr"] = corr
        out[f"macro_{nm}_trend"] = trend
        out[f"macro_{nm}_on"] = on.astype(float)
        out[f"macro_{nm}_dev"] = dev * 100
        bias = bias + np.where(on, -trend, 0)  # отрицательная связь: драйвер ↑ → альтам ↓
    if len(out.columns) == 0:
        return None
    out["macro_bias"] = np.sign(bias)
    # неделя пн–вс известна после закрытия воскресной дневной свечи
    out["usable_time"] = out.index + pd.Timedelta(days=1)
    return out.reset_index(names="week")


MACRO_BASE_COLS = ["macro_bias"]


def attach(f: pd.DataFrame, tbl: pd.DataFrame | None) -> pd.DataFrame:
    """Последние ИЗВЕСТНЫЕ на момент закрытия бара макро-значения."""
    if tbl is None or tbl.empty:
        f = f.copy()
        f["macro_bias"] = 0.0
        return f
    cols = [c for c in tbl.columns if c.startswith("macro_")]
    right = tbl[["usable_time"] + cols].rename(columns={"usable_time": "close_time"}).sort_values("close_time")
    right["close_time"] = right["close_time"].astype(f["close_time"].dtype)
    out = pd.merge_asof(f.sort_values("close_time"), right, on="close_time", direction="backward")
    out.index = f.index
    out["macro_bias"] = out["macro_bias"].fillna(0.0)
    return out


def current(tbl: pd.DataFrame | None, now: pd.Timestamp, cfg: dict) -> dict | None:
    """Состояние на сейчас (последняя закрытая неделя) — для сообщений и мини-приложения."""
    if tbl is None or tbl.empty:
        return None
    known = tbl[tbl["usable_time"] <= now]
    if known.empty:
        return None
    row = known.iloc[-1]
    drv = []
    for d in enabled_drivers(cfg):
        nm = d["name"]
        if f"macro_{nm}_corr" not in row:
            continue
        c = row[f"macro_{nm}_corr"]
        drv.append({"name": nm, "corr": None if pd.isna(c) else round(float(c), 2),
                    "trend": int(row[f"macro_{nm}_trend"]), "on": bool(row[f"macro_{nm}_on"] == 1),
                    "dev_pct": None if pd.isna(row[f"macro_{nm}_dev"]) else round(float(row[f"macro_{nm}_dev"]), 2)})
    return {"week": pd.Timestamp(row["week"]).strftime("%Y-%m-%d"), "bias": int(row["macro_bias"]),
            "mode": cfg.get("macro", {}).get("mode", "off"), "drivers": drv}


# ------------------------------------------------------------------ боевой режим

def _state_path():
    return _cfg.STATE_DIR / "macro.json"


def _cfg_key(cfg: dict) -> str:
    mc = cfg.get("macro", {})
    return json.dumps({k: mc.get(k) for k in ("corr_weeks", "corr_on", "basket_size", "drivers")}, sort_keys=True)


def _to_json(tbl: pd.DataFrame) -> list[dict]:
    t = tbl.tail(80).copy()
    t["week"] = t["week"].dt.strftime("%Y-%m-%d")
    t["usable_time"] = t["usable_time"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return json.loads(t.to_json(orient="records"))


def _from_json(rows: list[dict]) -> pd.DataFrame | None:
    if not rows:
        return None
    t = pd.DataFrame(rows)
    t["week"] = pd.to_datetime(t["week"], utc=True)
    t["usable_time"] = pd.to_datetime(t["usable_time"], utc=True)
    return t


def build_live(cfg: dict, provider, bases: list[str], days: int | None = None) -> pd.DataFrame | None:
    """Скачать драйверы и корзину и посчитать таблицу (используется сканером и бэктестом)."""
    mc = cfg.get("macro", {})
    days = days or warmup_days(cfg) + 14
    drivers = {d["name"]: s for d in enabled_drivers(cfg) if (s := fetch_driver(d, days)) is not None}
    if not drivers:
        return None
    frames = fetch_many(provider, basket_bases(bases, int(mc.get("basket_size", 10))), "1d", days)
    return build_table(drivers, basket_index(frames), cfg)


def load_live(cfg: dict, provider, bases: list[str], now: pd.Timestamp) -> pd.DataFrame | None:
    """Таблица с кэшем в state/macro.json: данные недельные, качать каждые 15 минут незачем."""
    mc = cfg.get("macro", {})
    path = _state_path()
    cache = {}
    if path.exists():
        try:
            cache = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            cache = {}
    fresh = (cache.get("key") == _cfg_key(cfg)
             and time.time() - cache.get("updated_ts", 0) < float(mc.get("refresh_hours", 6)) * 3600)
    if fresh:
        return _from_json(cache.get("table", []))
    tbl = build_live(cfg, provider, bases)
    if tbl is None:
        log.warning("Макро: свежих данных нет, использую кэш")
        return _from_json(cache.get("table", []))
    cache.update({"key": _cfg_key(cfg), "updated_ts": time.time(), "updated": now.isoformat(),
                  "table": _to_json(tbl)})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return tbl


# ------------------------------------------------------------------ тексты

ARR = {1: "↑", -1: "↓", 0: "→"}


def _drivers_txt(items) -> str:
    out = []
    for nm, trend, corr, on in items:
        c = "н/д" if corr is None or (isinstance(corr, float) and np.isnan(corr)) else f"{corr:+.2f}"
        out.append(f"{nm} {ARR.get(int(trend), '→')} · корр. {c}{' ✳️' if on else ''}")
    return "; ".join(out)


def signal_line(row: pd.Series, side: int, cfg: dict) -> str:
    """Строка для сообщения о сигнале."""
    if not active(cfg):
        return ""
    items = []
    for d in enabled_drivers(cfg):
        nm = d["name"]
        if f"macro_{nm}_corr" not in row:
            continue
        c = row[f"macro_{nm}_corr"]
        items.append((nm, row.get(f"macro_{nm}_trend", 0) or 0, None if pd.isna(c) else float(c),
                      row.get(f"macro_{nm}_on", 0) == 1))
    if not items:
        return ""
    b = int(row.get("macro_bias", 0) or 0)
    if b == 0:
        verdict = "связи с альтами нет — не учитывается" if not any(i[3] for i in items) else "нейтрально"
    elif b * side > 0:
        verdict = "✅ по направлению сделки"
    else:
        verdict = "⚠️ против сделки"
    shadow = " <i>(наблюдение)</i>" if cfg["macro"].get("mode") == "shadow" else ""
    return f"🌐 Макро: {_drivers_txt(items)} → {verdict}{shadow}\n"


def change_text(st: dict, cfg: dict) -> str:
    items = [(d["name"], d["trend"], d["corr"], d["on"]) for d in st["drivers"]]
    b = st["bias"]
    what = {1: "🟢 попутный ветер для лонгов по альтам",
            -1: "🔴 встречный ветер для лонгов по альтам",
            0: "⚪️ нейтрально — макро сейчас не влияет на альты"}[b]
    mode = {"shadow": "только наблюдение, сигналы не меняются",
            "penalty": f"сигналы против макро получают −{cfg['macro'].get('penalty', 10)} к силе",
            "block": "сигналы против макро не отправляются"}.get(st["mode"], st["mode"])
    thr = cfg["macro"].get("corr_on", -0.5)
    return (f"🌐 <b>Макро-режим изменился</b> (неделя до {st['week']})\n"
            f"{_drivers_txt(items)}\n{what}\n\n"
            f"<i>✳️ — связь активна: корреляция недельных доходностей с корзиной альтов ≤ {thr}. "
            f"Режим фильтра: {mode}.</i>")
