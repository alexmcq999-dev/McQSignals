"""Минимальный клиент Telegram Bot API + форматирование сообщений."""
from __future__ import annotations

import html
import logging
import os

import pandas as pd
import requests

from .strategies import MODULES
from .engine import REGIME_RU, Trade

log = logging.getLogger("tg")


class TG:
    def __init__(self, token: str | None = None, dry_run: bool = False):
        self.token = "" if dry_run else (token or os.getenv("TELEGRAM_BOT_TOKEN", "")).strip()
        self.fixed = [c.strip() for c in os.getenv("TELEGRAM_CHAT_ID", "").split(",") if c.strip()]
        self.enabled = bool(self.token)
        if not self.enabled:
            log.warning("TELEGRAM_BOT_TOKEN не задан — сообщения печатаются в лог (dry-run)")

    def _call(self, method, **params):
        r = requests.post(f"https://api.telegram.org/bot{self.token}/{method}", json=params, timeout=20)
        data = r.json()
        if not data.get("ok"):
            raise TGError(data.get("error_code"), data.get("description"))
        return data["result"]

    def check(self) -> str:
        """Проверяет токен. Возвращает @username бота или бросает TGError."""
        return "@" + self._call("getMe")["username"]

    def send(self, chat, text, reply_to=None) -> int | None:
        if not self.enabled:
            print(f"\n--- [{chat}] ---\n{text}\n")
            return None
        params = dict(chat_id=chat, text=text, parse_mode="HTML", disable_web_page_preview=True)
        if reply_to:
            params["reply_parameters"] = {"message_id": int(reply_to), "allow_sending_without_reply": True}
        try:
            return self._call("sendMessage", **params)["message_id"]
        except TGError as e:
            log.warning("sendMessage %s: %s", chat, e)
            # аннотация видна прямо на странице запуска в GitHub Actions
            print(f"::error title=Telegram::Не удалось отправить в чат {chat}: {e}", flush=True)
            if e.code == 403:  # пользователь заблокировал бота
                raise
            return None

    def broadcast(self, chats, text, reply_map: dict | None = None, on_blocked=None) -> dict:
        out = {}
        for c in chats:
            try:
                mid = self.send(c, text, (reply_map or {}).get(str(c)))
                if mid:
                    out[str(c)] = mid
            except TGError:
                if on_blocked:
                    on_blocked(c)
        return out

    def updates(self, offset: int):
        if not self.enabled:
            return []
        try:
            return self._call("getUpdates", offset=offset, timeout=0, allowed_updates=["message", "channel_post"])
        except TGError as e:
            log.warning("getUpdates: %s", e)
            return []


class TGError(Exception):
    def __init__(self, code, desc):
        super().__init__(f"{code}: {desc}")
        self.code = code


# ---------------------------------------------------------------- форматирование

def fp(x: float) -> str:
    """Цена с разумной точностью."""
    ax = abs(x)
    if ax >= 1000:
        return f"{x:,.2f}".replace(",", " ")
    if ax >= 1:
        return f"{x:.4f}".rstrip("0").rstrip(".")
    if ax == 0:
        return "0"
    digits = max(4, -int(f"{ax:e}".split("e")[1]) + 3)
    return f"{x:.{digits}f}".rstrip("0").rstrip(".")


def pct(a, b):
    return (b / a - 1) * 100


def bar(score: float) -> str:
    n = int(round(score / 10))
    return "▰" * n + "▱" * (10 - n)


ARROW = {1: "↑ бычий", -1: "↓ медвежий", 0: "→ нейтр."}


def news_line(ctx: dict | None, cfg: dict) -> str:
    if not ctx or not cfg.get("news", {}).get("show_in_signal", True):
        return ""
    parts = []
    if ctx["bull"] or ctx["bear"] or ctx["mixed"]:
        parts.append(f"новости по крипте 🟢{ctx['bull']} 🔴{ctx['bear']} 🟡{ctx['mixed']}")
    if ctx.get("fng") is not None:
        parts.append(f"F&G {ctx['fng']} ({ctx['fng_ru']})")
    ev = ctx.get("next_event")
    if ev is not None:
        tz = cfg.get("telegram", {}).get("timezone", "Europe/Moscow")
        parts.append(f"⚠️ {html.escape(str(ev.get('title', '')))} {ev['t'].tz_convert(tz).strftime('%d.%m %H:%M')}")
    return ("📰 Фон: " + " · ".join(parts) + "\n") if parts else ""


def crowd_line(row: pd.Series) -> str:
    z, g = row.get("ls_acc_z"), row.get("fng")
    parts = []
    if z is not None and pd.notna(z):
        who = "перегружена лонгами" if z >= 1.5 else "перегружена шортами" if z <= -1.5 else "без перекоса"
        parts.append(f"толпа {who} (z {z:+.1f})")
    if g is not None and pd.notna(g):
        parts.append(f"F&G {g:.0f}")
    return ("👥 " + " · ".join(parts) + "\n") if parts else ""


def macro_line(row: pd.Series, side: int, cfg: dict) -> str:
    from .macro import signal_line

    try:
        return signal_line(row, side, cfg)
    except Exception:  # noqa: BLE001 — строка контекста не должна ломать сигнал
        return ""


def signal_text(t: Trade, row: pd.Series, provider: str, cfg: dict, n_modules: int, nctx: dict | None = None) -> str:
    if t.kind == "breakout":
        return breakout_text(t, row, provider, cfg, nctx)
    rk, tf = cfg["risk"], cfg["timeframes"]
    side = "🟢 <b>LONG" if t.side > 0 else "🔴 <b>SHORT"
    sl_pct = pct(t.entry, t.sl)
    size = rk["risk_per_trade_pct"] / abs(sl_pct) * 100
    reasons = "\n".join(f"• {MODULES[m][0]}" for m in t.reasons if m in MODULES)
    tz = cfg.get("telegram", {}).get("timezone", "Europe/Moscow")
    ctime = pd.Timestamp(t.opened).ceil("min").tz_convert(tz).strftime("%d.%m %H:%M")
    tzl = {"Europe/Moscow": "МСК", "UTC": "UTC"}.get(tz, tz)
    btc = "" if t.symbol == "BTC" else f" · BTC: {ARROW[int(row.get('btc_htf_bias', 0))]}"
    if rk.get("exit_mode", "fixed") == "trail":
        rest = (f"Остаток: трейлинг-стоп {rk.get('trail_atr', 3.0):g}×ATR от лучшей цены "
                f"(≈ {fp(rk.get('trail_atr', 3.0) * t.atr)}), без потолка прибыли\n")
    else:
        rest = f"TP2: <code>{fp(t.tp2)}</code> ({pct(t.entry, t.tp2):+.2f}%) · {rk['tp2_r']}R\n"
    test = ("🧪 <b>ТЕСТ</b> — сигнал для проверки стратегии, не для торговли\n\n"
            if cfg.get("telegram", {}).get("test_mode") else "")
    kind = ("🔄 <b>Контртренд</b>: против толпы после снятия стопов\n" if t.kind == "contra" else "")
    if t.status == "pending":
        till = pd.Timestamp(t.expires).ceil("min").tz_convert(tz).strftime("%d.%m %H:%M")
        entry = (f"Вход: <b>лимит</b> <code>{fp(t.entry)}</code> — на откате, закрытие свечи "
                 f"<code>{fp(float(row['close']))}</code> не догоняем\n"
                 f"Действует до {till} {tzl}, иначе отмена\n")
    else:
        entry = f"Вход: <code>{fp(t.entry)}</code>\n"
    return (
        f"{test}{side} · {html.escape(t.symbol)}/USDT</b> · {tf['entry'].upper()}\n{kind}"
        f"Сила: <b>{t.conf:.0f}/100</b> {bar(t.conf)}\n"
        f"Режим: {REGIME_RU.get(t.regime, t.regime)} (ADX {row['adx']:.0f}) · "
        f"{tf['confirm'].upper()}: {ARROW[int(row['htf_bias'])]}{btc}\n\n"
        f"{entry}"
        f"Стоп: <code>{fp(t.sl)}</code> ({sl_pct:+.2f}%)\n"
        f"TP1: <code>{fp(t.tp1)}</code> ({pct(t.entry, t.tp1):+.2f}%) · {rk['tp1_r']}R — закрыть "
        f"{int(rk['tp1_close_frac'] * 100)}%, стоп в б/у\n"
        f"{rest}\n"
        f"✅ Подтверждения {len(t.reasons)}/{n_modules}:\n{reasons}\n\n"
        f"{crowd_line(row)}{macro_line(row, t.side, cfg)}{news_line(nctx, cfg)}"
        f"💼 Риск {rk['risk_per_trade_pct']:g}% депо → позиция ≈ {size:.0f}% депо\n"
        f"⏳ Макс. {rk['ttl_hours']}ч · свеча {ctime} {tzl} · {provider} · #{t.id}\n"
        f"<i>Не финансовый совет.</i>"
    )


def _pattern(side: int, hl) -> str:
    from .setups import pattern_name

    return pattern_name(side, bool(hl) if hl is not None and pd.notna(hl) else False)


def breakout_text(t: Trade, row: pd.Series, provider: str, cfg: dict, nctx: dict | None = None) -> str:
    """Сигнал v4: подтверждённый пробой заранее найденного паттерна."""
    from .setups import bcfg

    rk, tf, bc = cfg["risk"], cfg["timeframes"], bcfg(cfg)
    side = "🟢 <b>LONG" if t.side > 0 else "🔴 <b>SHORT"
    sl_pct = pct(t.entry, t.sl)
    size = rk["risk_per_trade_pct"] / abs(sl_pct) * 100
    tz = cfg.get("telegram", {}).get("timezone", "Europe/Moscow")
    tzl = {"Europe/Moscow": "МСК", "UTC": "UTC"}.get(tz, tz)
    ctime = pd.Timestamp(t.opened).ceil("min").tz_convert(tz).strftime("%d.%m %H:%M")
    btc = "" if t.symbol == "BTC" else f" · BTC: {ARROW[int(row.get('btc_htf_bias', 0))]}"
    word = "сопротивления" if t.side > 0 else "поддержки"
    if t.status == "pending":
        till = pd.Timestamp(t.expires).ceil("min").tz_convert(tz).strftime("%d.%m %H:%M")
        entry = (f"Вход: <b>лимит</b> <code>{fp(t.entry)}</code> — ретест пробитого уровня "
                 f"(свеча ушла далеко, не догоняем)\nДействует до {till} {tzl}, иначе отмена\n")
    else:
        entry = f"Вход: <code>{fp(t.entry)}</code> (по рынку, у уровня)\n"
    tgt = (f"🎯 Цель по высоте паттерна: <code>{fp(t.target)}</code> ({pct(t.entry, t.target):+.2f}%)\n"
           if t.target else "")
    test = ("🧪 <b>ТЕСТ</b> — сигнал для проверки стратегии, не для торговли\n\n"
            if cfg.get("telegram", {}).get("test_mode") else "")
    notes = "\n".join(f"• {html.escape(x)}" for x in t.notes.split(" · ") if x)
    return (
        f"{test}{side} · {html.escape(t.symbol)}/USDT</b> · {tf['entry'].upper()}\n"
        f"📐 <b>Пробой</b>: {_pattern(t.side, row.get('bo_hl'))}\n"
        f"Сила сетапа: <b>{t.conf:.0f}/100</b> {bar(t.conf)} (порог {bc['min_score']})\n"
        f"{tf['confirm'].upper()}: {ARROW[int(row['htf_bias'])]}{btc}\n\n"
        f"Уровень {word}: <code>{fp(t.level)}</code> — свеча закрылась за ним с объёмом\n"
        f"{entry}"
        f"Стоп: <code>{fp(t.sl)}</code> ({sl_pct:+.2f}%) — за базой паттерна\n"
        f"TP1: <code>{fp(t.tp1)}</code> ({pct(t.entry, t.tp1):+.2f}%) · {rk['tp1_r']}R — закрыть "
        f"{int(rk['tp1_close_frac'] * 100)}%, стоп в б/у\n"
        f"Остаток: трейлинг {rk.get('trail_atr', 3.0):g}×ATR (≈ {fp(rk.get('trail_atr', 3.0) * t.atr)})\n"
        f"{tgt}\n"
        f"✅ Сетап:\n{notes}\n\n"
        f"{crowd_line(row)}{macro_line(row, t.side, cfg)}{news_line(nctx, cfg)}"
        f"💼 Риск {rk['risk_per_trade_pct']:g}% депо → позиция ≈ {size:.0f}% депо\n"
        f"⏳ Макс. {rk['ttl_hours']}ч · свеча {ctime} {tzl} · {provider} · #{t.id}\n"
        f"<i>Не финансовый совет.</i>"
    )


def setup_alert_text(symbol: str, row: pd.Series, price: float, cfg: dict) -> str:
    """Предупреждение «готовится пробой»: паттерн сформирован, цена у уровня, входа ещё нет."""
    side = int(row["bo_armed"])
    lvl = float(row["bo_level"])
    word = "сопротивление" if side > 0 else "поддержку"
    dist = pct(price, lvl)
    test = "🧪 ТЕСТ · " if cfg.get("telegram", {}).get("test_mode") else ""
    notes = "\n".join(f"• {html.escape(x)}" for x in str(row.get("bo_notes", "")).split(" · ") if x)
    return (
        f"👀 {test}<b>Готовится пробой</b> · {'🟢' if side > 0 else '🔴'} {html.escape(symbol)}/USDT · "
        f"{cfg['timeframes']['entry'].upper()}\n"
        f"{_pattern(side, row.get('bo_hl')).capitalize()}: цена жмёт {word} <code>{fp(lvl)}</code> "
        f"({dist:+.2f}% от цены)\n"
        f"Оценка сетапа: <b>{float(row['bo_pre']):.0f}/100</b>\n{notes}\n\n"
        f"Сигнал на вход придёт ТОЛЬКО если свеча {cfg['timeframes']['entry'].upper()} закроется "
        f"{'выше' if side > 0 else 'ниже'} уровня с объёмом. Заранее не входим.\n"
        f"<i>Не финансовый совет.</i>"
    )


EVENT_TEXT = {
    "filled": "📥 <b>Лимит исполнен</b> — цена вернулась к уровню, сделка открыта",
    "expired": "🚫 <b>Лимит отменён</b> — ретеста не было, сделка не открылась (в статистику не идёт)",
    "tp1": "✅ <b>TP1 взят</b> ({r:+.2f}R) — фиксируем часть, стоп в безубыток",
    "tp2": "🎯 <b>TP2 взят</b> — сделка закрыта: <b>{r:+.2f}R</b>",
    "sl": "❌ <b>Стоп</b> — сделка закрыта: <b>{r:+.2f}R</b>",
    "be": "⚪️ <b>Безубыток</b> — остаток закрыт в б/у, итог <b>{r:+.2f}R</b>",
    "trail": "📈 <b>Трейлинг-стоп</b> — остаток закрыт, итог <b>{r:+.2f}R</b>",
    "timeout": "⏱ <b>Закрыто по времени</b> — итог <b>{r:+.2f}R</b>",
}


def event_text(t: Trade, ev: str) -> str:
    side = "LONG" if t.side > 0 else "SHORT"
    return f"{t.symbol}/USDT {side} #{t.id}\n" + EVENT_TEXT[ev].format(r=t.r_gross)


def stats_text(trades: list[dict], title: str) -> str:
    from .stats import summarize

    s = summarize(trades)
    if not s["n"]:
        return f"📊 <b>{title}</b>\nЗакрытых сигналов пока нет."
    return (
        f"📊 <b>{title}</b>\n"
        f"Сделок: {s['n']} · Winrate: <b>{s['winrate']:.0f}%</b>\n"
        f"Итого: <b>{s['total_r']:+.2f}R</b> · средн. {s['avg_r']:+.2f}R\n"
        f"Profit factor: {s['pf']:.2f} · макс. просадка {s['max_dd']:.1f}R\n"
        f"TP2: {s['by_exit'].get('tp2', 0)} · TP1→б/у: {s['by_exit'].get('be', 0)} · "
        f"трейлинг: {s['by_exit'].get('trail', 0)} · стоп: {s['by_exit'].get('sl', 0)} · "
        f"время: {s['by_exit'].get('timeout', 0)}\n"
        f"<i>R — в единицах риска, без комиссий.</i>"
    )
