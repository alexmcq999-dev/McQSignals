"""Бэктест той же логики, что и в боевом сканере.

  python backtest.py                    — по config.yaml (60 дней, топ-25 монет)
  python backtest.py --days 90 --symbols 40
  python backtest.py --grid             — перебор порогов min_score / min_agree
  python backtest.py --synthetic        — офлайн-проверка на синтетике (без сети)

Честность симуляции:
  • индикаторы каузальные, 1H берётся только ЗАКРЫТАЯ свеча;
  • вход по закрытию сигнальной свечи, выходы — со следующей свечи;
  • если в одной свече задеты и стоп, и тейк — считаем СТОП;
  • комиссия + проскальзывание на обе стороны вычитаются из R;
  • одна позиция на монету + кулдаун, как в боевом режиме.
"""
from __future__ import annotations

import argparse
import logging
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.config import REPORTS_DIR, load_config
from src.engine import analyze, make_trade, update_trade
from src.stats import summarize
from src.strategies import MODULES

log = logging.getLogger("backtest")


def simulate(out: pd.DataFrame, symbol: str, cfg: dict, start_time) -> list[dict]:
    bt, sc = cfg["backtest"], cfg["signals"]
    cost = bt["fee_pct"] + bt["slippage_pct"]
    cooldown = pd.Timedelta(minutes=sc["cooldown_minutes"])
    trades, t, next_ok = [], None, None
    sigs = out["signal"].to_numpy()
    ct = out["close_time"]
    cols = ["high", "low", "close", "close_time"]
    bars = out[cols].to_dict("records")
    for i in range(len(out)):
        if t is not None:
            update_trade(t, bars[i], cfg, fee_pct=cost)
            if t.status == "closed":
                trades.append(t.to_dict())
                t = None
            continue  # в баре закрытия новую сделку не открываем
        if sigs[i] == 0 or ct.iloc[i] < start_time:
            continue
        if next_ok is not None and ct.iloc[i] < next_ok:
            continue
        t = make_trade(out.iloc[i], int(sigs[i]), symbol, cfg)
        next_ok = ct.iloc[i] + cooldown
    return trades


def run(data: dict, btc_h1, cfg: dict, start_time) -> list[dict]:
    trades = []
    for sym, (d15, d1h) in data.items():
        out = analyze(d15, d1h, btc_h1, cfg, is_btc=(sym == "BTC"))
        trades += simulate(out, sym, cfg, start_time)
    trades.sort(key=lambda x: x["opened"])
    return trades


# ------------------------------------------------------------------ отчёт

def _row(name, tr):
    s = summarize(tr, "r_net")
    if not s["n"]:
        return f"| {name} | 0 | – | – | – | – |"
    pf = "∞" if s["pf"] == float("inf") else f"{s['pf']:.2f}"
    return f"| {name} | {s['n']} | {s['winrate']:.0f}% | {s['avg_r']:+.3f} | {s['total_r']:+.1f} | {pf} |"


HDR = "| Срез | Сделок | Winrate | Ср. R | Сумма R | PF |\n|---|---|---|---|---|---|"


def report(trades: list[dict], cfg: dict, meta: dict) -> str:
    s = summarize(trades, "r_net")
    lines = [f"# Бэктест McQ Signals — {meta['date']}", "",
             f"Период: {meta['days']} дн. · монет: {meta['n_symbols']} · источник: {meta['provider']} · "
             f"TF 15m + 1H · комиссия+проскальзывание {cfg['backtest']['fee_pct'] + cfg['backtest']['slippage_pct']:.3f}%/сторона",
             f"Порог силы: {cfg['signals']['min_score']} · мин. модулей: {cfg['signals']['min_agree']}", ""]
    if not s["n"]:
        return "\n".join(lines + ["Сделок нет — снизь min_score / min_agree."])
    lines += ["## Итог (R после комиссий)", "",
              f"- Сделок: **{s['n']}** ({s['n'] / meta['days']:.1f} в день)",
              f"- Winrate: **{s['winrate']:.1f}%**",
              f"- Матожидание: **{s['avg_r']:+.3f}R** на сделку",
              f"- Сумма: **{s['total_r']:+.1f}R** · Profit factor: **{s['pf']:.2f}** · Макс. просадка: **{s['max_dd']:.1f}R**",
              f"- Выходы: {s['by_exit']}", "",
              "## Срезы", "", HDR]
    half = len(trades) // 2
    lines += [_row("Первая половина периода", trades[:half]), _row("Вторая половина периода", trades[half:])]
    lines += [_row("LONG", [t for t in trades if t["side"] > 0]), _row("SHORT", [t for t in trades if t["side"] < 0])]
    for reg in ("trend", "neutral", "range"):
        lines.append(_row(f"Режим: {reg}", [t for t in trades if t["regime"] == reg]))
    for lo, hi in ((0, 70), (70, 80), (80, 101)):
        lines.append(_row(f"Сила {lo}–{min(hi, 100)}", [t for t in trades if lo <= t["conf"] < hi]))
    lines += ["", "## Вклад модулей (сделки, где модуль голосовал «за»)", "", HDR]
    for m, (nm, fam) in MODULES.items():
        lines.append(_row(f"{nm} [{fam}]", [t for t in trades if m in t["reasons"]]))
    by_sym = {}
    for t in trades:
        by_sym.setdefault(t["symbol"], []).append(t)
    ranked = sorted(by_sym.items(), key=lambda kv: sum(x["r_net"] for x in kv[1]), reverse=True)
    lines += ["", "## Монеты", "", HDR] + [_row(k, v) for k, v in ranked]
    lines += ["", "> Бэктест не гарантирует будущий результат. Смотри на стабильность между половинами "
              "периода и на PF > 1.2 после комиссий, а не на максимальную доходность."]
    return "\n".join(lines)


def tg_summary(trades, cfg, meta) -> str:
    s = summarize(trades, "r_net")
    if not s["n"]:
        return "🧪 Бэктест: сделок нет."
    half = len(trades) // 2
    a, b = summarize(trades[:half], "r_net"), summarize(trades[half:], "r_net")
    return (f"🧪 <b>Бэктест {meta['days']}д · {meta['n_symbols']} монет</b>\n"
            f"Сделок: {s['n']} ({s['n'] / meta['days']:.1f}/день)\n"
            f"Winrate: <b>{s['winrate']:.1f}%</b> · PF: <b>{s['pf']:.2f}</b>\n"
            f"Матожидание: <b>{s['avg_r']:+.3f}R</b> · Сумма: {s['total_r']:+.1f}R\n"
            f"Макс. просадка: {s['max_dd']:.1f}R\n"
            f"1-я половина: {a.get('avg_r', 0):+.3f}R · 2-я: {b.get('avg_r', 0):+.3f}R\n"
            f"<i>После комиссий и проскальзывания. Полный отчёт — в reports/ репозитория.</i>")


# ------------------------------------------------------------------ загрузка

def load_live(cfg, days, n_symbols):
    from src.data import build_universe, fetch_many, pick_provider, split_closed

    provider, tickers = pick_provider()
    uni = build_universe(cfg, provider, tickers)[:n_symbols]
    bases = [a["base"] for a in uni]
    n15, n1h = days * 96 + 400, days * 24 + 260
    k15 = fetch_many(provider, bases, "15m", n15)
    k1h = fetch_many(provider, list(dict.fromkeys(bases + ["BTC"])), "1h", n1h)
    now = pd.Timestamp.now(tz="UTC")
    data = {b: (split_closed(k15[b], now)[0], split_closed(k1h[b], now)[0]) for b in bases if b in k15 and b in k1h}
    btc = split_closed(k1h["BTC"], now)[0] if "BTC" in k1h else None
    return data, btc, provider.name


def load_synthetic(days, n_symbols):
    import sys
    sys.path.insert(0, "tests")
    from synth import resample, synth

    n = days * 96 + 400
    btc15 = synth(n, seed=999)
    data = {}
    for i in range(n_symbols):
        d = synth(n, seed=i)
        data[f"SYN{i}"] = (d, resample(d))
    return data, resample(btc15), "synthetic"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int)
    ap.add_argument("--symbols", type=int)
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--no-telegram", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg = load_config()
    days = a.days or cfg["backtest"]["days"]
    nsym = a.symbols or cfg["backtest"]["symbols"]
    data, btc, provider = load_synthetic(days, nsym) if a.synthetic else load_live(cfg, days, nsym)
    last = max(d[0]["close_time"].iloc[-1] for d in data.values())
    start = last - pd.Timedelta(days=days)
    meta = {"date": datetime.now(timezone.utc).strftime("%Y-%m-%d"), "days": days,
            "n_symbols": len(data), "provider": provider}

    REPORTS_DIR.mkdir(exist_ok=True)
    if a.grid:
        rows = ["| min_score | min_agree | Сделок | Winrate | Ср. R | Сумма R | PF | 1-я пол. | 2-я пол. |",
                "|---|---|---|---|---|---|---|---|---|"]
        for ms in (58, 62, 66, 70, 74):
            for ma in (4, 5, 6):
                c = load_config(overrides={"signals": {"min_score": ms, "min_agree": ma}})
                tr = run(data, btc, c, start)
                s = summarize(tr, "r_net")
                if not s["n"]:
                    rows.append(f"| {ms} | {ma} | 0 | | | | | | |")
                    continue
                h = len(tr) // 2
                a1, a2 = summarize(tr[:h], "r_net"), summarize(tr[h:], "r_net")
                rows.append(f"| {ms} | {ma} | {s['n']} | {s['winrate']:.0f}% | {s['avg_r']:+.3f} | "
                            f"{s['total_r']:+.1f} | {s['pf']:.2f} | {a1.get('avg_r', 0):+.3f} | {a2.get('avg_r', 0):+.3f} |")
                log.info("grid %s/%s: n=%d avgR=%+.3f", ms, ma, s["n"], s["avg_r"])
        text = f"# Сетка порогов — {meta['date']} ({days}д, {len(data)} монет, {provider})\n\n" + "\n".join(rows) + \
               "\n\n> Выбирай не максимум, а устойчивую зону, где обе половины периода положительны."
        (REPORTS_DIR / "grid_latest.md").write_text(text, encoding="utf-8")
        print(text)
        return

    trades = run(data, btc, cfg, start)
    md = report(trades, cfg, meta)
    (REPORTS_DIR / "backtest_latest.md").write_text(md, encoding="utf-8")
    if not a.synthetic:
        (REPORTS_DIR / f"backtest_{meta['date']}.md").write_text(md, encoding="utf-8")
    pd.DataFrame(trades).to_csv(REPORTS_DIR / "backtest_trades.csv", index=False)
    print(md)

    if not a.no_telegram and not a.synthetic and os.getenv("TELEGRAM_BOT_TOKEN"):
        from src import store
        from src.telegram import TG

        tg = TG()
        subs = store.load("subscribers")
        chats = list(dict.fromkeys(tg.fixed + [str(c) for c in subs["chats"]]))
        tg.broadcast(chats, tg_summary(trades, cfg, meta))


if __name__ == "__main__":
    main()
