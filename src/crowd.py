"""Позиционирование толпы и сентимент: данные для контртрендовых модулей.

Источники (бесплатные, без ключей, доступны из GitHub Actions):
  • Binance USDⓈ-M futures metrics — data.binance.vision, дневные файлы с 5-минутными строками:
      count_long_short_ratio            — лонги/шорты по ЧИСЛУ счетов (розница, «толпа»)
      sum_toptrader_long_short_ratio    — лонги/шорты топ-трейдеров по ОБЪЁМУ позиций («умные деньги»)
      sum_taker_long_short_vol_ratio    — агрессивные покупки/продажи
      sum_open_interest_value           — открытый интерес, $
    Файл за день D публикуется на следующий день, поэтому значение за D считается
    доступным только с D+1 + publish_delay_hours. Так же и в бэктесте — без заглядывания вперёд.
  • Fear & Greed Index — api.alternative.me, дневной, вся история.
"""
from __future__ import annotations

import io
import json
import logging
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ROOT, STATE_DIR
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

log = logging.getLogger("crowd")

# отдельная сессия: 404 по отсутствующим дням — норма, долгие ретраи не нужны
HTTP = requests.Session()
HTTP.mount("https://", HTTPAdapter(max_retries=Retry(total=1, backoff_factor=0.5,
                                                     status_forcelist=(429, 500, 502, 503, 504)), pool_maxsize=32))
HTTP.headers["User-Agent"] = "McQSignals/1.0"

METRICS_URL = "https://data.binance.vision/data/futures/um/daily/metrics/{s}/{s}-metrics-{d}.zip"
FNG_URL = "https://api.alternative.me/fng/?limit=0&format=json"
COLS = {
    "count_long_short_ratio": "ls_acc",
    "sum_toptrader_long_short_ratio": "ls_top",
    "sum_taker_long_short_vol_ratio": "taker",
    "sum_open_interest_value": "oi",
}
CACHE_DIR = ROOT / ".cache" / "crowd"


# ------------------------------------------------------------------ Binance metrics

def _fetch_day(fsym: str, day: str) -> dict | None:
    """Агрегат одного дня: средние ratio за день, OI на конец дня. None, если файла нет."""
    try:
        r = HTTP.get(METRICS_URL.format(s=fsym, d=day), timeout=20)
    except Exception as e:  # noqa: BLE001
        log.debug("metrics %s %s: %s", fsym, day, e)
        return None
    if r.status_code != 200:
        return None
    try:
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            raw = z.read(z.namelist()[0]).decode()
        df = pd.read_csv(io.StringIO(raw))
        if "create_time" not in df.columns:  # файл без заголовка
            df = pd.read_csv(io.StringIO(raw), header=None, names=[
                "create_time", "symbol", "sum_open_interest", "sum_open_interest_value",
                "count_toptrader_long_short_ratio", "sum_toptrader_long_short_ratio",
                "count_long_short_ratio", "sum_taker_long_short_vol_ratio"])
        out = {}
        for src, dst in COLS.items():
            v = pd.to_numeric(df[src], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if v.empty:
                continue
            out[dst] = float(v.iloc[-1]) if dst == "oi" else float(v.mean())
        return out or None
    except Exception as e:  # noqa: BLE001
        log.debug("metrics parse %s %s: %s", fsym, day, e)
        return None


def _load_cache(path: Path) -> dict:
    try:
        return json.loads(path.read_text()) if path.exists() else {}
    except json.JSONDecodeError:
        return {}


def fetch_metrics(bases: list[str], days: list[str], cache_path: Path | None = None, workers: int = 16) -> dict:
    """{base: {"fsym": ..., "days": {YYYY-MM-DD: {...}}}} — докачивает только недостающие дни."""
    cache_path = cache_path or (STATE_DIR / "crowd.json")
    cache = _load_cache(cache_path)
    missing_ok = cache.setdefault("_missing", {})  # дни, которых нет на сервере (не перекачиваем)
    jobs = []
    for b in bases:
        ent = cache.setdefault(b, {"fsym": None, "days": {}})
        if ent["fsym"] is None:
            # тикер фьючерса: BTCUSDT, либо 1000PEPEUSDT для «мелких» монет
            for cand in (f"{b}USDT", f"1000{b}USDT", f"1000000{b}USDT"):
                if _fetch_day(cand, days[-1]) is not None or _fetch_day(cand, days[-2]) is not None:
                    ent["fsym"] = cand
                    break
            else:
                ent["fsym"] = ""
        if not ent["fsym"]:
            continue
        miss = set(missing_ok.get(b, []))
        jobs += [(b, ent["fsym"], d) for d in days if d not in ent["days"] and d not in miss]

    def one(j):
        b, fs, d = j
        return b, d, _fetch_day(fs, d)

    if jobs:
        log.info("Binance metrics: докачка %d дневных файлов", len(jobs))
        with ThreadPoolExecutor(workers) as ex:
            for b, d, v in ex.map(one, jobs):
                if v is None:
                    missing_ok.setdefault(b, []).append(d)
                else:
                    cache[b]["days"][d] = v
    # в state держим только нужное окно, в .cache — всё
    keep = set(days)
    if cache_path.parent == STATE_DIR:
        for b, ent in cache.items():
            if b != "_missing":
                ent["days"] = {d: v for d, v in ent["days"].items() if d in keep}
        cache["_missing"] = {b: [d for d in v if d in keep] for b, v in missing_ok.items()}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, separators=(",", ":")))
    return cache


def crowd_table(cache: dict, base: str, cfg: dict) -> pd.DataFrame | None:
    """Дневная таблица для монеты + z-оценки + момент, с которого данные «известны» (usable_time)."""
    ent = cache.get(base)
    if not ent or not ent.get("days"):
        return None
    cc = cfg.get("crowd", {})
    df = pd.DataFrame.from_dict(ent["days"], orient="index").sort_index()
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.asfreq("D")
    w = int(cc.get("z_window_days", 14))
    for c in ("ls_acc", "ls_top", "taker"):
        if c in df:
            x = np.log(df[c].clip(lower=1e-6))
            m = x.rolling(w, min_periods=max(5, w // 2)).mean().shift(1)
            sd = x.rolling(w, min_periods=max(5, w // 2)).std(ddof=0).shift(1)
            df[c + "_z"] = (x - m) / sd.replace(0, np.nan)
    if "oi" in df:
        df["oi_chg"] = df["oi"].pct_change(3) * 100
    df["usable_time"] = df.index + pd.Timedelta(days=1) + pd.Timedelta(hours=cc.get("publish_delay_hours", 8))
    return df.reset_index(drop=True)


# ------------------------------------------------------------------ Fear & Greed

def fetch_fng(cache_path: Path | None = None) -> pd.DataFrame | None:
    cache_path = cache_path or (STATE_DIR / "fng.json")
    try:
        d = HTTP.get(FNG_URL, timeout=20).json()["data"]
        rows = {pd.Timestamp(int(x["timestamp"]), unit="s", tz="UTC").strftime("%Y-%m-%d"): int(x["value"]) for x in d}
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(dict(sorted(rows.items())[-60:] if cache_path.parent == STATE_DIR
                                             else sorted(rows.items()))))
    except Exception as e:  # noqa: BLE001
        log.warning("Fear & Greed недоступен: %s", e)
        rows = _load_cache(cache_path)
    if not rows:
        return None
    df = pd.DataFrame({"fng": list(rows.values())}, index=pd.to_datetime(list(rows.keys()), utc=True)).sort_index()
    # значение за день D считаем известным с конца дня D (консервативно)
    df["usable_time"] = df.index + pd.Timedelta(days=1)
    return df.reset_index(drop=True)


def attach(f: pd.DataFrame, crowd: pd.DataFrame | None, fng: pd.DataFrame | None) -> pd.DataFrame:
    """Присоединяет к барам последние ИЗВЕСТНЫЕ на момент закрытия бара значения."""
    out = f.sort_values("close_time")
    for tbl, cols in ((crowd, ["ls_acc_z", "ls_top_z", "taker_z", "oi_chg"]), (fng, ["fng"])):
        if tbl is None or tbl.empty:
            for c in cols:
                out[c] = np.nan
            continue
        right = tbl[["usable_time"] + [c for c in cols if c in tbl]].dropna(subset=["usable_time"])
        right = right.rename(columns={"usable_time": "close_time"}).sort_values("close_time")
        right["close_time"] = right["close_time"].astype(out["close_time"].dtype)
        out = pd.merge_asof(out, right, on="close_time", direction="backward")
        for c in cols:
            if c not in out:
                out[c] = np.nan
    out.index = f.index
    return out


def recent_days(now: pd.Timestamp, n: int) -> list[str]:
    last = (now.tz_convert("UTC").floor("D") - pd.Timedelta(days=1))
    return [(last - pd.Timedelta(days=i)).strftime("%Y-%m-%d") for i in range(n)][::-1]
