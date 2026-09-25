"""Один проход сканера. Запускается GitHub Actions каждые 15 минут.

  python scan.py            — боевой режим (нужен TELEGRAM_BOT_TOKEN)
  python scan.py --dry-run  — без отправки в Telegram, сообщения в консоль
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import timedelta

import pandas as pd

from src import crowd as cw
from src import news as nw
from src import store
from src.config import TF_MINUTES, load_config
from src.data import build_universe, fetch_many, pick_provider, split_closed
from src.engine import Trade, analyze, update_trade
from src.engine import make_trade, trade_ok
from src.strategies import MODULES
from src.telegram import TG, event_text, signal_text, stats_text, fp

log = logging.getLogger("scan")

HELP = (
    "🤖 <b>McQ Signals</b> — сигналы ({etf} + фильтр {ctf}) по ликвидным монетам с капой ≥ $100M.\n\n"
    "Каждый сигнал — консенсус классических стратегий (тренд, импульс, пробой, возврат к среднему, "
    "объём, позиционирование толпы) с учётом режима рынка, тренда {ctf} и BTC.\n\n"
    "Команды:\n"
    "/stats — результаты сигналов (7д / 30д / всё время)\n"
    "/open — открытые сигналы\n"
    "/coins — какие монеты сканируются\n"
    "/stop — отписаться\n\n"
    "⚠️ Бот работает через GitHub Actions: ответ на команду приходит при следующем запуске (до ~15 мин).\n"
    "<i>Не финансовый совет. Всегда используй стоп-лосс.</i>"
)


def help_text(cfg) -> str:
    tf = cfg["timeframes"]
    text = HELP.format(etf=tf["entry"].upper(), ctf=tf["confirm"].upper())
    if cfg.get("telegram", {}).get("test_mode"):
        text = "🧪 <b>ТЕСТОВЫЙ РЕЖИМ</b>: сигналы для проверки стратегии, не для торговли.\n\n" + text
    return text


def _closed_since(hist: dict, days: int | None):
    if days is None:
        return hist["closed"]
    edge = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)
    return [t for t in hist["closed"] if t.get("closed") and pd.Timestamp(t["closed"]) >= edge]


def handle_commands(tg: TG, cfg, subs, sig, hist):
    allow = cfg.get("telegram", {}).get("allow_public_subscribe", True)
    for u in tg.updates(subs["offset"]):
        subs["offset"] = u["update_id"] + 1
        m = u.get("message") or u.get("channel_post")
        if not m or not m.get("text"):
            continue
        chat = str(m["chat"]["id"])
        cmd = m["text"].strip().split()[0].split("@")[0].lower()
        known = chat in subs["chats"] or chat in tg.fixed
        if cmd == "/start":
            if allow or chat in tg.fixed:
                if chat not in subs["chats"] and chat not in tg.fixed:
                    subs["chats"].append(chat)
                tg.send(chat, "✅ Подписка активна. Сигналы будут приходить сюда.\n\n" + help_text(cfg))
            else:
                tg.send(chat, "Бот приватный.")
            continue
        if not known:
            continue
        if cmd == "/stop":
            if chat in subs["chats"]:
                subs["chats"].remove(chat)
            tg.send(chat, "Отписка выполнена. /start — подписаться снова.")
        elif cmd == "/stats":
            tg.send(chat, "\n\n".join([
                stats_text(_closed_since(hist, 7), "7 дней"),
                stats_text(_closed_since(hist, 30), "30 дней"),
                stats_text(_closed_since(hist, None), "Всё время"),
            ]))
        elif cmd == "/open":
            rows = [Trade.from_dict(d) for d in sig["open"]]
            if not rows:
                tg.send(chat, "Открытых сигналов нет.")
            else:
                lines = [f"{'🟢' if t.side > 0 else '🔴'} {t.symbol} вход {fp(t.entry)} · стоп {fp(t.sl)} · "
                         f"{'TP1 ✅' if t.status == 'tp1' else 'в работе'} · #{t.id}" for t in rows]
                tg.send(chat, "📂 <b>Открытые сигналы</b>\n" + "\n".join(lines))
        elif cmd == "/coins":
            u_ = store.STATE_DIR / "universe.json"
            import json
            assets = json.loads(u_.read_text())["assets"] if u_.exists() else []
            tg.send(chat, f"🪙 <b>Сканируется {len(assets)} монет</b>\n" + ", ".join(a["base"] for a in assets))
        elif cmd.startswith("/"):
            tg.send(chat, help_text(cfg))


def _pub(t: Trade, price: float | None = None) -> dict:
    d = {k: getattr(t, k) for k in ("id", "symbol", "side", "kind", "entry", "sl", "tp1", "risk", "opened", "conf",
                                    "regime", "reasons", "status", "exit_reason", "r_gross", "closed")}
    d["side"] = "LONG" if t.side > 0 else "SHORT"
    d["reasons"] = [MODULES[m][0] for m in t.reasons if m in MODULES]
    if price is not None and t.status != "closed":
        frac = 0.5 if t.status == "tp1" else 0.0
        d["price"] = price
        d["r_open"] = round(t.r_gross * (1 if frac else 0) + (1 - frac) * t.side * (price - t.entry) / t.risk, 2)
    return d


def export_app(cfg, now, still_open, hist, prices, provider, n_coins, blackout_ev):
    """state/app_signals.json — данные для вкладки «Сигналы» в мини-приложении marketnews999."""
    import json

    from src.stats import summarize

    def st(days):
        s_ = summarize(_closed_since(hist, days), "r_gross")
        return {k: (None if isinstance(v, float) and v == float("inf") else v) for k, v in s_.items()}

    exp = store.STATE_DIR / "experiments.json"
    data = {
        "updated": now.isoformat(),
        "test_mode": bool(cfg.get("telegram", {}).get("test_mode")),
        "provider": provider,
        "coins": n_coins,
        "timeframes": {"entry": cfg["timeframes"]["entry"], "confirm": cfg["timeframes"]["confirm"]},
        "exit": {"mode": cfg["risk"].get("exit_mode"), "tp1_r": cfg["risk"]["tp1_r"],
                 "trail_atr": cfg["risk"].get("trail_atr")},
        "blackout": ({"title": blackout_ev.get("title"), "ts": blackout_ev["t"].isoformat()} if blackout_ev else None),
        "open": [_pub(t, prices.get(t.symbol)) for t in still_open],
        "closed": [_pub(Trade.from_dict(d)) for d in hist["closed"][-50:]][::-1],
        "stats": {"d7": st(7), "d30": st(30), "all": st(None)},
        "backtest": json.loads(exp.read_text()) if exp.exists() else None,
    }
    (store.STATE_DIR / "app_signals.json").write_text(json.dumps(data, ensure_ascii=False, default=str), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--now", help="ISO-время UTC для воспроизведения прогона (тесты/отладка)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg = load_config()
    tg = TG(dry_run=args.dry_run)
    sig, hist, subs = store.load("signals"), store.load("history"), store.load("subscribers")
    now = pd.Timestamp(args.now) if args.now else pd.Timestamp.now(tz="UTC")
    if now.tzinfo is None:
        now = now.tz_localize("UTC")

    def chats():
        return list(dict.fromkeys(tg.fixed + [str(c) for c in subs["chats"]])) or ["dry-run"]

    def drop_chat(c):
        if str(c) in subs["chats"]:
            subs["chats"].remove(str(c))

    if tg.enabled:
        try:
            log.info("Telegram: бот %s, фиксированных чатов: %d", tg.check(), len(tg.fixed))
        except Exception as e:  # noqa: BLE001
            print(f"::error title=Telegram::Токен не работает ({e}). Проверь секрет TELEGRAM_BOT_TOKEN", flush=True)
    elif not args.dry_run:
        print("::error title=Telegram::Секрет TELEGRAM_BOT_TOKEN не задан — сигналы не отправляются", flush=True)
    if tg.enabled and not tg.fixed:
        print("::warning title=Telegram::TELEGRAM_CHAT_ID не задан — сигналы получат только те, кто написал боту /start",
              flush=True)

    try:
        handle_commands(tg, cfg, subs, sig, hist)
    except Exception as e:  # noqa: BLE001
        log.warning("Команды: %s", e)
    store.save("subscribers", subs)

    provider, tickers = pick_provider()
    universe = build_universe(cfg, provider, tickers)
    bases = [a["base"] for a in universe]
    open_trades = [Trade.from_dict(d) for d in sig["open"]]
    tf = cfg["timeframes"]
    etf, ctf, ttf = tf["entry"], tf["confirm"], tf.get("track", tf["entry"])
    kE = fetch_many(provider, bases, etf, tf["bars_entry"])
    kC = fetch_many(provider, list(dict.fromkeys(bases + ["BTC"])), ctf, tf["bars_confirm"])
    track_syms = [t.symbol for t in open_trades]
    n_track = min(1500, int(cfg["risk"]["ttl_hours"] * 60 / TF_MINUTES[ttf]) + 50)
    kT = fetch_many(provider, track_syms, ttf, n_track) if track_syms else {}
    log.info("Свечи: %s=%d, %s=%d монет, сопровождение %s=%d", etf, len(kE), ctf, len(kC), ttf, len(kT))

    # ---------------------------------------------- 1. сопровождение открытых сигналов
    still_open = []
    for t in open_trades:
        df = kT.get(t.symbol)
        if df is None:
            still_open.append(t)
            continue
        closed, _ = split_closed(df, now)
        new = closed[closed["close_time"] > pd.Timestamp(t.last_checked or t.opened)]
        for _, b in new.iterrows():
            evs = update_trade(t, b, cfg)
            for ev in evs:
                if ev == "tp1" and "tp2" in evs:
                    continue
                tg.broadcast(chats(), event_text(t, ev), reply_map=t.msg_ids, on_blocked=drop_chat)
            if t.status == "closed":
                break
        if t.status == "closed":
            hist["closed"].append(t.to_dict())
            log.info("Закрыт %s #%s: %s %+.2fR", t.symbol, t.id, t.exit_reason, t.r_gross)
        else:
            still_open.append(t)
    hist["closed"] = hist["closed"][-3000:]

    # ---------------------------------------------- 2. поиск новых сигналов
    sc = cfg["signals"]
    cr = cfg.get("contrarian", {})
    crowd_tbls, fng = {}, None
    if cr.get("enabled") or cr.get("crowd_filter") or cr.get("fng_filter"):
        try:
            cache = cw.fetch_metrics(bases, cw.recent_days(now, cfg.get("crowd", {}).get("days", 20)))
            crowd_tbls = {b: cw.crowd_table(cache, b, cfg) for b in bases}
            fng = cw.fetch_fng()
        except Exception as e:  # noqa: BLE001
            log.warning("Данные толпы: %s", e)
    news_data = nw.load(cfg)
    bo = nw.blackout(news_data, now, cfg)
    if bo:
        log.info("Блэкаут: %s (%s) — новые сигналы не отправляем", bo.get("title"), bo["t"])
    btc_h1 = split_closed(kC["BTC"], now)[0] if "BTC" in kC else None
    max_age = pd.Timedelta(minutes=sc.get("max_signal_age_minutes", 75))
    open_syms = {t.symbol for t in still_open}
    candidates = []
    for b in bases:
        df, h = kE.get(b), kC.get(b)
        if df is None or h is None or len(df) < 260:
            continue
        closed, last_price = split_closed(df, now)
        h_closed, _ = split_closed(h, now)
        out = analyze(closed, h_closed, btc_h1, cfg, is_btc=(b == "BTC"), crowd=crowd_tbls.get(b), fng=fng)
        prev = sig["last_bar"].get(b)
        last_seen = pd.Timestamp(prev) if prev else out["close_time"].iloc[-sc["lookback_bars_on_run"] - 1]
        tail = out.tail(sc["lookback_bars_on_run"])
        fresh = tail[(tail["signal"] != 0) & (tail["close_time"] > last_seen)
                     & (now - tail["close_time"] <= max_age)]
        sig["last_bar"][b] = out["close_time"].iloc[-1].isoformat()
        if fresh.empty or b in open_syms or bo:
            continue
        cd = sig["cooldown"].get(b)
        if cd and now < pd.Timestamp(cd):
            continue
        row = fresh.iloc[-1]
        t = make_trade(row, int(row["signal"]), b, cfg)
        if not trade_ok(t, cfg):
            log.info("%s: стоп слишком узкий, комиссии %.2fR — пропуск", b, t.cost_r)
            continue
        drift = t.side * (last_price - t.entry) / t.risk
        if not -0.5 < drift < 0.5:
            log.info("%s: цена уже ушла (%.2fR) — пропуск", b, drift)
            continue
        nctx = nw.context(news_data, now, b)
        if cfg.get("news", {}).get("against_news_filter"):
            why = nw.against_news(nctx, t.side)
            if why:
                log.info("%s: против свежей официальной новости «%s» — пропуск", b, why)
                continue
        candidates.append((float(row["conf"]), t, row, nctx))

    candidates.sort(key=lambda x: x[0], reverse=True)
    slots = max(0, min(sc["max_signals_per_run"], sc["max_open_signals"] - len(still_open)))
    for _, t, row, nctx in candidates[:slots]:
        sig["seq"] += 1
        t.id = f"{sig['seq']:04d}"
        t.last_checked = t.opened
        n_mod = len(cr.get("confirm", [])) + 1 if t.kind == "contra" else len(sc.get("modules", MODULES))
        text = signal_text(t, row, provider.name, cfg, n_mod, nctx)
        t.msg_ids = tg.broadcast(chats(), text, on_blocked=drop_chat)
        still_open.append(t)
        sig["cooldown"][t.symbol] = (now + timedelta(minutes=sc["cooldown_minutes"])).isoformat()
        log.info("СИГНАЛ %s %s conf=%.0f", t.symbol, "LONG" if t.side > 0 else "SHORT", t.conf)
    log.info("Кандидатов: %d, отправлено: %d, открыто: %d", len(candidates), min(len(candidates), slots), len(still_open))

    # ---------------------------------------------- 3. ежедневный отчёт
    hour = cfg.get("telegram", {}).get("daily_report_hour_utc", 6)
    today = now.strftime("%Y-%m-%d")
    if now.hour == hour and sig.get("last_daily") != today:
        sig["last_daily"] = today
        text = stats_text(_closed_since(hist, 1), "Итоги 24ч") + "\n\n" + stats_text(_closed_since(hist, 7), "7 дней")
        tg.broadcast(chats(), text, on_blocked=drop_chat)

    sig["open"] = [t.to_dict() for t in still_open]
    sig["cooldown"] = {k: v for k, v in sig["cooldown"].items() if pd.Timestamp(v) > now}
    store.save("signals", sig)
    try:
        prices = {}
        for src_ in (kE, kT):
            for b_, df_ in src_.items():
                prices[b_] = float(df_["close"].iloc[-1])
        export_app(cfg, now, still_open, hist, prices, provider.name, len(bases), bo)
    except Exception as e:  # noqa: BLE001
        log.warning("Экспорт для мини-приложения: %s", e)
    store.save("history", hist)
    store.save("subscribers", subs)


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001
        logging.exception("Сканер упал")
        sys.exit(1)
