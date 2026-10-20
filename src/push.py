"""Пуш-уведомления в приложение McQCryptoNews (OneSignal, работает и для PWA, и для будущих сторов).

Нужны: переменная ONESIGNAL_APP_ID и секрет ONESIGNAL_API_KEY (Settings → Secrets and variables → Actions).
Без них функции молча ничего не делают — Telegram продолжает работать как раньше.
Тексты — короткие и без HTML: RU + EN (остальные языки телефона получат английский).
"""
from __future__ import annotations

import logging
import os
import re

import requests

log = logging.getLogger("push")
APP_URL = os.getenv("WEBAPP_URL", "https://alexmcq999-dev.github.io/marketnews999/").rstrip("/") + "/"


def enabled() -> bool:
    return bool(os.getenv("ONESIGNAL_APP_ID") and os.getenv("ONESIGNAL_API_KEY"))


def send(title: dict, body: dict, path: str = "#signals", topic: str = "signals", ttl: int = 4 * 3600) -> None:
    if not enabled():
        return
    try:
        r = requests.post(
            "https://api.onesignal.com/notifications?c=push", timeout=15,
            headers={"Authorization": f"Key {os.environ['ONESIGNAL_API_KEY']}", "Content-Type": "application/json"},
            json={"app_id": os.environ["ONESIGNAL_APP_ID"],
                  "included_segments": [os.getenv("ONESIGNAL_SEGMENT") or "Total Subscriptions"],
                  "headings": {k: v for k, v in title.items() if v}, "contents": {k: v for k, v in body.items() if v},
                  "url": APP_URL + path, "web_push_topic": topic, "ttl": ttl})
        if r.status_code >= 300:
            log.warning("OneSignal %s: %s", r.status_code, r.text[:200])
    except Exception as e:  # noqa: BLE001 — пуш не должен ломать сканер
        log.warning("OneSignal: %s", e)


def _p(x: float) -> str:
    ax = abs(x)
    return f"{x:,.2f}".replace(",", " ") if ax >= 1000 else (f"{x:.4f}".rstrip("0").rstrip(".") if ax >= 1 else f"{x:.6g}")


def plain(html: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", html or "")).strip()


# ---------------------------------------------------------------- готовые сообщения
def new_signal(t) -> None:
    long_ = t.side > 0
    ru_side, en_side = ("🟢 Покупка", "🟢 LONG") if long_ else ("🔴 Продажа", "🔴 SHORT")
    pend_ru = "лимит " if t.status == "pending" else ""
    pend_en = "limit " if t.status == "pending" else ""
    send({"ru": f"{ru_side} {t.symbol}", "en": f"{en_side} {t.symbol}"},
         {"ru": f"Вход {pend_ru}{_p(t.entry)} · стоп {_p(t.sl)} · цель {_p(t.tp1)}",
          "en": f"Entry {pend_en}{_p(t.entry)} · stop {_p(t.sl)} · target {_p(t.tp1)}"})


EV = {
    "tp1": ("✅ Цель взята", "✅ Target hit", "часть прибыли зафиксирована, стоп в безубыток", "partial profit taken, stop to breakeven"),
    "tp2": ("🎯 Цель 2 взята", "🎯 Target 2 hit", "сделка закрыта", "trade closed"),
    "sl": ("❌ Стоп", "❌ Stop hit", "сделка закрыта", "trade closed"),
    "be": ("⚪️ Безубыток", "⚪️ Breakeven", "остаток закрыт в ноль", "rest closed at breakeven"),
    "trail": ("📈 Трейлинг-стоп", "📈 Trailing stop", "прибыль зафиксирована", "profit locked in"),
    "timeout": ("⏱ Закрыто по времени", "⏱ Closed on time", "срок сделки истёк", "trade time limit reached"),
    "filled": ("📥 Лимит исполнен", "📥 Limit filled", "сделка открыта", "trade opened"),
    "expired": ("🚫 Лимит отменён", "🚫 Limit cancelled", "цена не вернулась к уровню", "price didn't return to the level"),
}


def trade_event(t, ev: str) -> None:
    if ev not in EV:
        return
    tr, te, br, be = EV[ev]
    r = "" if ev in ("filled", "expired") else f" · {t.r_gross:+.2f}R"
    send({"ru": f"{tr}: {t.symbol}", "en": f"{te}: {t.symbol}"}, {"ru": br + r, "en": be + r})


def setup_alert(symbol: str, side: int, level: float) -> None:
    send({"ru": f"👀 Готовится пробой {symbol}", "en": f"👀 {symbol} breakout setting up"},
         {"ru": f"Цена у уровня {_p(level)}. Вход — только после закрытия свечи за уровнем.",
          "en": f"Price is at {_p(level)}. Entry only after a candle closes beyond the level."}, ttl=2 * 3600)


def event_alert(text_html: str) -> None:
    first = plain(text_html.split("\n")[0])
    send({"ru": "⚠️ Скоро важное событие", "en": "⚠️ Major event soon"},
         {"ru": first, "en": first}, path="#calendar", topic="events", ttl=3600)


def morning() -> None:
    send({"ru": "☀️ План дня готов", "en": "☀️ Daily plan is ready"},
         {"ru": "События дня, настроение рынка, открытые сигналы и уровни ликвидаций.",
          "en": "Today's events, market sentiment, open signals and liquidation levels."},
         path="#overview", topic="morning", ttl=6 * 3600)
