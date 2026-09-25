"""Новостной фон из бота marketnews999 (его мини-приложение публикует data/app.json).

Используется только в боевом режиме: исторического архива новостей с тональностью нет,
поэтому на бэктесте эти фильтры не проверить — они работают как риск-фильтры:
  • блэкаут вокруг важных макро-событий (High impact): в эти минуты рынок дёргают, стопы выносят;
  • (опционально) не открывать сделку против свежей официальной новости;
  • контекст в сообщении сигнала.
"""
from __future__ import annotations

import logging
import re

import pandas as pd

from .data import HTTP

log = logging.getLogger("news")

CRYPTO_RE = re.compile(r"\b(btc|eth|bitcoin|биткоин|эфир|крипт\w*|crypto\w*|стейбл\w*|altcoin|альткоин\w*)\b", re.I)


def load(cfg: dict) -> dict | None:
    nc = cfg.get("news", {})
    if not nc.get("enabled"):
        return None
    try:
        r = HTTP.get(nc["url"], params={"t": int(pd.Timestamp.now().timestamp())}, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:  # noqa: BLE001
        log.warning("Новостной фон недоступен: %s", e)
        return None


def _events(data: dict) -> list[dict]:
    out = []
    for e in data.get("calendar") or []:
        try:
            out.append({**e, "t": pd.Timestamp(e["ts"]).tz_convert("UTC")})
        except Exception:  # noqa: BLE001
            continue
    return out


def blackout(data: dict | None, now: pd.Timestamp, cfg: dict) -> dict | None:
    """Ближайшее High-impact событие в окне ±N минут, иначе None."""
    if not data:
        return None
    win = pd.Timedelta(minutes=cfg.get("news", {}).get("calendar_blackout_minutes", 45))
    for e in _events(data):
        if str(e.get("impact", "")).lower() == "high" and abs(e["t"] - now) <= win:
            return e
    return None


def context(data: dict | None, now: pd.Timestamp, symbol: str) -> dict:
    """Сводка для сообщения и фильтров: тональность крипто-новостей последнего дайджеста и т.п."""
    ctx = {"bull": 0, "bear": 0, "mixed": 0, "official": [], "fng": None, "fng_ru": "", "next_event": None, "mood": ""}
    if not data:
        return ctx
    sym_re = re.compile(rf"\b{re.escape(symbol)}\b", re.I)
    for n in data.get("news") or []:
        text = f"{n.get('assets', '')} {n.get('title', '')} {n.get('title_en', '')}"
        if not (CRYPTO_RE.search(text) or sym_re.search(text)):
            continue
        b = str(n.get("bias", "")).lower()
        ctx[{"bullish": "bull", "bearish": "bear"}.get(b, "mixed")] += 1
        try:
            age = now - pd.Timestamp(n["ts"]).tz_convert("UTC")
        except Exception:  # noqa: BLE001
            continue
        if n.get("official") and age <= pd.Timedelta(hours=12) and b in ("bullish", "bearish"):
            ctx["official"].append({"title": n.get("title", ""), "bias": b})
    fng = ((data.get("sentiment") or {}).get("fng") or {}).get("crypto") or {}
    ctx["fng"], ctx["fng_ru"] = fng.get("value"), fng.get("rating_ru", "")
    upcoming = [e for e in _events(data) if e["t"] >= now and str(e.get("impact", "")).lower() == "high"]
    if upcoming:
        ctx["next_event"] = min(upcoming, key=lambda e: e["t"])
    ctx["mood"] = data.get("mood") or ""
    return ctx


def against_news(ctx: dict, side: int) -> str | None:
    """Свежая официальная новость против направления сделки → причина пропуска."""
    for o in ctx.get("official", []):
        if (o["bias"] == "bearish" and side > 0) or (o["bias"] == "bullish" and side < 0):
            return o["title"]
    return None
