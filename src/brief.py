"""Алерты перед событиями, сводка «толпы» и утренний «План дня».

Всё строится из данных, которые бот уже собирает:
  • календарь и итоги событий — из мини-приложения marketnews999 (data/app.json);
  • позиционирование толпы — Binance futures metrics (src/crowd.py);
  • сигналы и статистика — state/.
"""
from __future__ import annotations

import html

import numpy as np
import pandas as pd

FLAG = {"USD": "🇺🇸", "EUR": "🇪🇺", "GBP": "🇬🇧", "JPY": "🇯🇵", "CNY": "🇨🇳", "ALL": "🌐"}
VS = {"above": "выше прогноза", "below": "ниже прогноза", "inline": "в рамках прогноза"}
TONE = {"hawkish": "ястребино", "dovish": "голубино", "neutral": "нейтрально"}
WD = ["пн", "вт", "ср", "чт", "пт", "сб", "вс"]


def _events(news_data: dict | None) -> list[dict]:
    out = []
    for e in (news_data or {}).get("calendar") or []:
        try:
            out.append({**e, "t": pd.Timestamp(e["ts"]).tz_convert("UTC")})
        except Exception:  # noqa: BLE001
            continue
    return out


def _ekey(e: dict) -> str:
    return f"{e.get('ts')}|{e.get('country')}|{e.get('title')}"


def _tz(cfg: dict) -> str:
    return cfg.get("telegram", {}).get("timezone", "Europe/Moscow")


# ------------------------------------------------------------------ 1. алерт перед событием
def event_alerts(news_data: dict | None, now: pd.Timestamp, sig: dict, cfg: dict, open_trades: list) -> list[str]:
    """Тексты алертов о High-событиях, до которых осталось ≤ alert_minutes. Помечает отправленные в sig."""
    nc = cfg.get("news", {})
    lead = pd.Timedelta(minutes=nc.get("alert_minutes", 40))
    sent = sig.setdefault("event_alerts", {})
    out = []
    for e in _events(news_data):
        if str(e.get("impact", "")).lower() != "high":
            continue
        dt = e["t"] - now
        k = _ekey(e)
        if not (pd.Timedelta(0) < dt <= lead) or k in sent:
            continue
        sent[k] = now.isoformat()
        mins = max(1, int(dt.total_seconds() // 60))
        local = e["t"].tz_convert(_tz(cfg)).strftime("%H:%M")
        nums = " · ".join(x for x in (f"прогноз {e['forecast']}" if e.get("forecast") else "",
                                      f"пред. {e['previous']}" if e.get("previous") else "") if x)
        hist = typical_move(news_data, e.get("title", ""))
        lines = [f"⚠️ <b>Через {mins} мин</b> ({local}): {FLAG.get(e.get('country'), '')} "
                 f"<b>{html.escape(e.get('country', ''))}</b> {html.escape(e.get('title', ''))}"]
        if nums:
            lines.append(html.escape(nums))
        if hist:
            lines.append(hist)
        bo = nc.get("calendar_blackout_minutes", 45)
        lines.append(f"В эти минуты часто выносят стопы. Новые сигналы бот не даёт ±{bo} мин.")
        if open_trades:
            syms = ", ".join(f"{t.symbol} {'L' if t.side > 0 else 'S'}" for t in open_trades[:8])
            lines.append(f"📂 Открыто сигналов: {len(open_trades)} ({syms}) — проверь стопы.")
        out.append("\n".join(lines))
    # чистим старые отметки
    edge = now - pd.Timedelta(days=8)
    sig["event_alerts"] = {k: v for k, v in sent.items() if pd.Timestamp(v) > edge}
    return out


def typical_move(news_data: dict | None, title: str) -> str:
    """Средняя реакция BTC за час после событий с тем же названием (из накопленных итогов)."""
    moves = [abs(x["value"]) for e in _events(news_data) if e.get("title") == title and e.get("result")
             for x in e["result"].get("reaction", []) if x.get("asset") == "BTC"]
    if len(moves) < 2:
        return ""
    return f"Обычно BTC за час после него: ±{np.mean(moves):.1f}% (по {len(moves)} прошлым)."


# ------------------------------------------------------------------ 2. толпа
def crowd_rows(crowd_tbls: dict, bases: list[str]) -> list[dict]:
    rows = []
    for b in bases:
        t = crowd_tbls.get(b)
        if t is None or t.empty:
            continue
        last = t.dropna(subset=["ls_acc"]).tail(1) if "ls_acc" in t else t.tail(0)
        if last.empty:
            continue
        r = last.iloc[0]

        def f(c, nd=2):
            v = r.get(c)
            return None if v is None or pd.isna(v) else round(float(v), nd)

        z = f("ls_acc_z")
        label = ("перегрев лонгов" if z is not None and z >= 2 else "перегрев шортов" if z is not None and z <= -2
                 else "много лонгов" if z is not None and z >= 1.5 else "много шортов" if z is not None and z <= -1.5
                 else "норма")
        day = (pd.Timestamp(r["usable_time"]) - pd.Timedelta(days=1)).strftime("%Y-%m-%d") if "usable_time" in r else ""
        rows.append({"symbol": b, "ls": f("ls_acc"), "z": z, "top_z": f("ls_top_z"), "oi_chg": f("oi_chg", 1),
                     "label": label, "day": day})
    rows.sort(key=lambda x: -abs(x["z"] or 0))
    return rows


def crowd_line(rows: list[dict], n: int = 3) -> str:
    lo = [r for r in rows if (r["z"] or 0) >= 1.5][:n]
    sh = [r for r in rows if (r["z"] or 0) <= -1.5][:n]
    parts = []
    if lo:
        parts.append("лонгами: " + ", ".join(f"{r['symbol']} (z {r['z']:+.1f})" for r in lo))
    if sh:
        parts.append("шортами: " + ", ".join(f"{r['symbol']} (z {r['z']:+.1f})" for r in sh))
    return "перегружены " + "; ".join(parts) if parts else "сильных перекосов нет"


# ------------------------------------------------------------------ 3. план дня
def morning_due(now: pd.Timestamp, sig: dict, cfg: dict) -> bool:
    """Первый скан после brief_hour по местному времени (Actions может опаздывать) и до полудня."""
    local = now.tz_convert(_tz(cfg))
    hour = cfg.get("telegram", {}).get("brief_hour_local", 8)
    return hour <= local.hour < 12 and sig.get("last_daily") != local.strftime("%Y-%m-%d")


def morning_text(now: pd.Timestamp, cfg: dict, news_data: dict | None, crowd: list[dict], open_pub: list[dict],
                 stats_24h: dict, stats_7d: dict, macro_state: dict | None = None) -> str:
    tz = _tz(cfg)
    local = now.tz_convert(tz)
    nd = news_data or {}
    ev = _events(nd)
    p = [f"☀️ <b>План дня</b> — {local:%d.%m} ({WD[local.weekday()]})"]
    if cfg.get("telegram", {}).get("test_mode"):
        p[0] += " · 🧪 ТЕСТ"

    # фон
    bg = []
    fng = ((nd.get("sentiment") or {}).get("fng") or {}).get("crypto") or {}
    if fng.get("value") is not None:
        bg.append(f"F&G крипта {fng['value']} ({fng.get('rating_ru', '')})")
    if macro_state:
        bg.append({1: "макро DXY: 🟢 попутно альтам", -1: "макро DXY: 🔴 против альтов",
                   0: "макро DXY: ⚪️ нейтрально"}.get(macro_state.get("bias"), ""))
    if bg:
        p.append("🌡 " + " · ".join(x for x in bg if x))
    if nd.get("mood"):
        mood = nd["mood"] if len(nd["mood"]) < 320 else nd["mood"][:317].rsplit(" ", 1)[0] + "…"
        p.append(f"🧭 {html.escape(mood)}")

    # события сегодня
    day_end = local.normalize() + pd.Timedelta(days=1, hours=2)
    today = [e for e in ev if str(e.get("impact", "")).lower() == "high"
             and now - pd.Timedelta(minutes=30) <= e["t"] <= day_end.tz_convert("UTC")]
    if today:
        rows = []
        for e in sorted(today, key=lambda x: x["t"]):
            extra = f" <i>(прогноз {html.escape(e['forecast'])})</i>" if e.get("forecast") else ""
            hm = typical_move(nd, e.get("title", ""))
            rows.append(f"• {e['t'].tz_convert(tz):%H:%M} {FLAG.get(e.get('country'), '')} "
                        f"{html.escape(e.get('title', ''))}{extra}" + (f"\n   <i>{hm}</i>" if hm else ""))
        p.append("📅 <b>Сегодня важно</b>\n" + "\n".join(rows))
    else:
        p.append("📅 Важных макро-событий сегодня нет")

    # итоги вчера
    done = [e for e in ev if e.get("result") and str(e.get("impact", "")).lower() == "high"
            and now - pd.Timedelta(hours=24) <= e["t"] <= now]
    if done:
        rows = []
        for e in sorted(done, key=lambda x: x["t"])[-4:]:
            r = e["result"]
            bits = []
            if r.get("actual"):
                bits.append(f"{html.escape(r['actual'])}" + (f" vs {html.escape(e['forecast'])}" if e.get("forecast") else "")
                            + (f" — {VS[r['vs_forecast']]}" if r.get("vs_forecast") in VS else ""))
            if r.get("tone") in TONE:
                bits.append(TONE[r["tone"]])
            btc = next((x["value"] for x in r.get("reaction", []) if x["asset"] == "BTC"), None)
            if btc is not None:
                bits.append(f"BTC {btc:+.1f}% за час")
            rows.append(f"• {html.escape(e.get('title', ''))}: " + ("; ".join(bits) if bits else "итог уточняется"))
        p.append("📌 <b>Итоги за сутки</b>\n" + "\n".join(rows))

    # толпа
    if crowd:
        day = crowd[0].get("day", "")
        p.append(f"👥 <b>Толпа на фьючерсах</b> ({day}): {crowd_line(crowd)}")

    # сигналы
    if open_pub:
        rows = [f"{'🟢' if o['side'] == 'LONG' else '🔴'} {o['symbol']} {o.get('r_open', 0):+.2f}R"
                + (" · TP1 ✓" if o.get("status") == "tp1" else "") for o in open_pub[:10]]
        p.append(f"📂 <b>Открытые сигналы ({len(open_pub)})</b>\n" + "\n".join(rows))
    else:
        p.append("📂 Открытых сигналов нет")

    def st(s, name):
        if not s.get("n"):
            return f"{name}: сделок нет"
        return f"{name}: {s['n']} сд., {s['total_r']:+.2f}R, winrate {s['winrate']:.0f}%"
    p.append("📊 " + " · ".join([st(stats_24h, "24ч"), st(stats_7d, "7д")]))
    p.append("<i>Не финансовый совет.</i>")
    return "\n\n".join(p)
