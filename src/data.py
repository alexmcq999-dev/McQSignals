"""Рыночные данные: список активов (капитализация) и свечи.

GitHub Actions работает на серверах в США, а api.binance.com оттуда
заблокирован (HTTP 451). Поэтому:
  • основной источник — data-api.binance.vision (официальное зеркало
    рыночных данных Binance, без гео-блока);
  • резерв — KuCoin и OKX (публичные API без ключей).
Капитализация — CoinGecko (опционально с demo-ключом), резерв — CoinPaprika.
"""
from __future__ import annotations

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import STATE_DIR, TF_MINUTES

log = logging.getLogger("data")

STABLES = {
    "USDT", "USDC", "DAI", "FDUSD", "TUSD", "USDE", "USDD", "PYUSD", "USDS", "BUSD", "USD1", "USDP",
    "GUSD", "FRAX", "LUSD", "EURC", "EURT", "EURI", "AEUR", "USDB", "USDX", "USDY", "USD0", "RLUSD",
    "SUSD", "CRVUSD", "GHO", "USDL", "BFUSD", "USTC", "UST", "USDG", "SUSDE", "SUSDS", "FXUSD", "DOLA",
    "XAUT", "PAXG", "USDA", "USDF", "USDTB", "BUIDL", "OUSG", "USYC", "USDO",
}
WRAPPED = {
    "WBTC", "WETH", "STETH", "WSTETH", "WEETH", "EETH", "CBBTC", "RETH", "METH", "EZETH", "RSETH",
    "SOLVBTC", "LBTC", "BTCB", "JITOSOL", "MSOL", "BNSOL", "WBETH", "CBETH", "OSETH", "SWETH", "PUFETH",
    "TBTC", "WBNB", "WTRX", "WAVAX", "WMATIC", "WPOL", "WSOL", "STSOL", "JUPSOL", "SAVAX", "SFRXETH",
    "FRXETH", "ANKRETH", "LSETH", "RSWETH", "BBTC", "ENZOBTC", "UNIBTC", "PUMPBTC", "CLBTC", "WHYPE",
    "KHYPE", "STHYPE", "BETH", "STX-WRAPPED", "HBTC", "RENBTC", "SUPEROETHB", "WEETHS", "USDT0",
}
NAME_EXCLUDE = ("wrapped", "staked", "bridged", "liquid staking", "restaked", "tokenized", "usd coin")


def _session() -> requests.Session:
    s = requests.Session()
    retry = Retry(total=4, backoff_factor=1.0, status_forcelist=(418, 429, 500, 502, 503, 504),
                  allowed_methods=("GET",), respect_retry_after_header=True)
    s.mount("https://", HTTPAdapter(max_retries=retry, pool_maxsize=16))
    s.headers["User-Agent"] = "McQSignals/1.0"
    return s


HTTP = _session()


def _get(url, params=None, headers=None, timeout=15):
    r = HTTP.get(url, params=params, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.json()


def _frame(rows, tf: str) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["t", "open", "high", "low", "close", "volume"])
    if df.empty:
        return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume", "close_time"])
    df = df.astype({"t": "int64", "open": float, "high": float, "low": float, "close": float, "volume": float})
    df["time"] = pd.to_datetime(df["t"], unit="ms", utc=True)
    df["close_time"] = df["time"] + pd.Timedelta(minutes=TF_MINUTES[tf]) - pd.Timedelta(milliseconds=1)
    df = df.drop(columns="t").drop_duplicates("time").sort_values("time").reset_index(drop=True)
    return df[["time", "open", "high", "low", "close", "volume", "close_time"]]


# ------------------------------------------------------------------ провайдеры

class Binance:
    name = "Binance"
    base = "https://data-api.binance.vision/api/v3"

    def pair(self, b):
        return f"{b}USDT"

    def tickers(self) -> dict[str, float]:
        out = {}
        for t in _get(f"{self.base}/ticker/24hr", timeout=25):
            s = t["symbol"]
            if s.endswith("USDT"):
                out[s[:-4]] = float(t["quoteVolume"])
        return out

    def klines(self, b, tf, limit, end_ms=None):
        rows, end = [], end_ms
        while len(rows) < limit:
            n = min(1000, limit - len(rows))
            p = {"symbol": self.pair(b), "interval": tf, "limit": n}
            if end:
                p["endTime"] = end
            chunk = _get(f"{self.base}/klines", p)
            if not chunk:
                break
            rows = [r[:6] for r in chunk] + rows
            end = int(chunk[0][0]) - 1
            if len(chunk) < n:
                break
        return _frame(rows, tf)


class KuCoin:
    name = "KuCoin"
    base = "https://api.kucoin.com/api/v1"
    TF = {"15m": "15min", "1h": "1hour", "5m": "5min", "30m": "30min", "4h": "4hour", "1d": "1day"}

    def tickers(self):
        d = _get(f"{self.base}/market/allTickers", timeout=25)["data"]["ticker"]
        return {t["symbol"][:-5]: float(t.get("volValue") or 0) for t in d if t["symbol"].endswith("-USDT")}

    def klines(self, b, tf, limit, end_ms=None):
        step = TF_MINUTES[tf] * 60
        end = int((end_ms or time.time() * 1000) / 1000)
        rows = []
        while len(rows) < limit:
            n = min(1500, limit - len(rows))
            p = {"type": self.TF[tf], "symbol": f"{b}-USDT", "startAt": end - n * step, "endAt": end}
            chunk = _get(f"{self.base}/market/candles", p)["data"]
            if not chunk:
                break
            # [time, open, close, high, low, volume, turnover] — от новых к старым
            conv = [[int(r[0]) * 1000, r[1], r[3], r[4], r[2], r[5]] for r in reversed(chunk)]
            rows = conv + rows
            end = int(chunk[-1][0]) - 1
            if len(chunk) < n:
                break
        return _frame(rows, tf).tail(limit).reset_index(drop=True)


class OKX:
    name = "OKX"
    base = "https://www.okx.com/api/v5"
    TF = {"15m": "15m", "1h": "1H", "5m": "5m", "30m": "30m", "4h": "4H", "1d": "1Dutc"}

    def tickers(self):
        d = _get(f"{self.base}/market/tickers", {"instType": "SPOT"}, timeout=25)["data"]
        return {t["instId"][:-5]: float(t.get("volCcy24h") or 0) for t in d if t["instId"].endswith("-USDT")}

    def klines(self, b, tf, limit, end_ms=None):
        rows, after = [], end_ms
        while len(rows) < limit:
            p = {"instId": f"{b}-USDT", "bar": self.TF[tf], "limit": 100}
            if after:
                p["after"] = after
            chunk = _get(f"{self.base}/market/history-candles", p)["data"]
            if not chunk:
                break
            rows = [r[:6] for r in reversed(chunk)] + rows
            after = int(chunk[-1][0])
            time.sleep(0.12)  # лимит OKX 20 req / 2s
        return _frame(rows[-limit:], tf)


PROVIDERS = {"binance": Binance, "kucoin": KuCoin, "okx": OKX}


def pick_provider():
    """Первый доступный провайдер (или заданный через env DATA_PROVIDER)."""
    order = [os.getenv("DATA_PROVIDER", "").lower()] if os.getenv("DATA_PROVIDER") else ["binance", "kucoin", "okx"]
    last = None
    for key in order:
        p = PROVIDERS[key]()
        try:
            tick = p.tickers()
            if len(tick) > 50:
                log.info("Провайдер данных: %s (%d пар USDT)", p.name, len(tick))
                return p, tick
        except Exception as e:  # noqa: BLE001
            last = e
            log.warning("Провайдер %s недоступен: %s", key, e)
    raise RuntimeError(f"Нет доступного источника данных: {last}")


# ------------------------------------------------------------------ вселенная

def _coingecko_caps() -> list[dict]:
    key = os.getenv("COINGECKO_API_KEY", "").strip()
    headers = {"x-cg-demo-api-key": key} if key else None
    out = []
    for page in (1, 2):
        d = _get("https://api.coingecko.com/api/v3/coins/markets",
                 {"vs_currency": "usd", "order": "market_cap_desc", "per_page": 250, "page": page},
                 headers=headers, timeout=25)
        out += [{"symbol": c["symbol"].upper(), "name": c["name"], "mcap": c.get("market_cap") or 0,
                 "vol": c.get("total_volume") or 0} for c in d]
        time.sleep(1.5)
    return out


def _coinpaprika_caps() -> list[dict]:
    d = _get("https://api.coinpaprika.com/v1/tickers", {"quotes": "USD"}, timeout=40)
    d = sorted(d, key=lambda c: c.get("rank") or 10**9)[:600]
    return [{"symbol": c["symbol"].upper(), "name": c["name"], "mcap": c["quotes"]["USD"].get("market_cap") or 0,
             "vol": c["quotes"]["USD"].get("volume_24h") or 0} for c in d]


def _excluded(sym: str, name: str, extra: set[str]) -> bool:
    n = name.lower()
    return (sym in STABLES or sym in WRAPPED or sym in extra or any(w in n for w in NAME_EXCLUDE)
            or (sym.startswith("USD") and len(sym) <= 6) or sym.endswith("USD"))


def build_universe(cfg: dict, provider, tickers: dict[str, float], force=False) -> list[dict]:
    uc = cfg["universe"]
    path = STATE_DIR / "universe.json"
    cache = json.loads(path.read_text()) if path.exists() else None
    if cache and not force and cache.get("provider") == provider.name and cache.get("v") == 2:
        age_h = (time.time() - cache["updated_ts"]) / 3600
        if age_h < uc["refresh_hours"]:
            return cache["assets"]

    caps, src = None, None
    for fn, nm in ((_coingecko_caps, "CoinGecko"), (_coinpaprika_caps, "CoinPaprika")):
        try:
            caps, src = fn(), nm
            break
        except Exception as e:  # noqa: BLE001
            log.warning("%s недоступен: %s", nm, e)
    if caps is None:
        if cache:
            log.warning("Капитализация недоступна — используем старый список монет")
            return cache["assets"]
        log.warning("Капитализация недоступна — фильтр только по ликвидности биржи")

    extra = {s.upper() for s in uc.get("extra_exclude", [])}
    meta: dict[str, dict] = {}
    if caps:
        for c in caps:  # сортировано по капе: первое вхождение тикера = самая крупная монета
            meta.setdefault(c["symbol"], c)

    assets = []
    for base, qv in tickers.items():
        m = meta.get(base)
        name = m["name"] if m else base
        if _excluded(base, name, extra):
            continue
        if caps and (not m or m["mcap"] < uc["min_market_cap_usd"]):
            continue
        if qv < uc["min_quote_volume_24h_usd"]:
            continue
        # Защита от совпадения тикеров: общий объём монеты по всем биржам (CoinGecko) не может быть
        # сильно меньше объёма одной биржи — значит под этим тикером на бирже другая монета.
        if m and m.get("vol") and m["vol"] < 0.2 * qv:
            log.info("Пропуск %s: объём CoinGecko %.0f ≪ биржи %.0f (другая монета с тем же тикером?)", base, m["vol"], qv)
            continue
        assets.append({"base": base, "name": name, "mcap": m["mcap"] if m else None, "qvol": qv})
    assets.sort(key=lambda a: a["qvol"], reverse=True)
    assets = assets[: uc["max_assets"]]
    if not any(a["base"] == "BTC" for a in assets) and "BTC" in tickers:
        assets.insert(0, {"base": "BTC", "name": "Bitcoin", "mcap": meta.get("BTC", {}).get("mcap"), "qvol": tickers["BTC"]})

    STATE_DIR.mkdir(exist_ok=True)
    path.write_text(json.dumps({
        "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "v": 2, "updated_ts": time.time(), "provider": provider.name, "cap_source": src, "assets": assets,
    }, ensure_ascii=False, indent=1))
    log.info("Вселенная: %d монет (капа: %s)", len(assets), src)
    return assets


def fetch_many(provider, bases: list[str], tf: str, limit: int, workers: int = 8) -> dict[str, pd.DataFrame]:
    def one(b):
        try:
            return b, provider.klines(b, tf, limit)
        except Exception as e:  # noqa: BLE001
            log.warning("Свечи %s %s: %s", b, tf, e)
            return b, None

    w = 3 if provider.name == "OKX" else workers
    with ThreadPoolExecutor(w) as ex:
        return {b: df for b, df in ex.map(one, bases) if df is not None and len(df) > 0}


def split_closed(df: pd.DataFrame, now: pd.Timestamp | None = None):
    """Отделяет закрытые свечи от текущей (незакрытой). Возвращает (closed_df, last_price)."""
    now = now or pd.Timestamp.now(tz="UTC")
    closed = df[df["close_time"] < now].reset_index(drop=True)
    return closed, float(df["close"].iloc[-1])
