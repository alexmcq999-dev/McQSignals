"""Бэктест той же логики, что и в боевом сканере.

  python backtest.py                    — по config.yaml (60 дней, топ-25 монет)
  python backtest.py --days 90 --symbols 40
  python backtest.py --grid             — перебор порогов min_score / min_agree
  python backtest.py --experiments      — сравнение вариантов из config.yaml на 2 периодах по 60 дней
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
from src.engine import cost_pct, make_trade, score_frame, trade_ok, update_trade
from src.strategies import build_features
from src.stats import summarize
from src.strategies import MODULES

log = logging.getLogger("backtest")


def simulate(out: pd.DataFrame, symbol: str, cfg: dict, start_time, end_time=None) -> list[dict]:
    sc = cfg["signals"]
    cost = cost_pct(cfg)
    cooldown = pd.Timedelta(minutes=sc["cooldown_minutes"])
    trades, t, next_ok = [], None, None
    sigs = out["signal"].to_numpy()
    start64 = np.datetime64(pd.Timestamp(start_time).tz_convert(None))
    end64 = np.datetime64(pd.Timestamp(end_time).tz_convert(None)) if end_time is not None else None
    ct = pd.to_datetime(out["close_time"]).dt.tz_convert(None).to_numpy()
    bars = out[["high", "low", "close", "close_time"]].to_dict("records")
    for i in range(len(out)):
        if t is not None:
            update_trade(t, bars[i], cfg, fee_pct=cost)
            if t.status == "closed":
                trades.append(t.to_dict())
                t = None
            continue  # в баре закрытия новую сделку не открываем
        if sigs[i] == 0 or ct[i] < start64 or (end64 is not None and ct[i] >= end64):
            continue
        if next_ok is not None and ct[i] < next_ok:
            continue
        cand = make_trade(out.iloc[i], int(sigs[i]), symbol, cfg)
        if not trade_ok(cand, cfg):
            continue
        t = cand
        next_ok = ct[i] + np.timedelta64(int(cooldown.total_seconds()), "s")
    return trades


def prepare(data: dict, btc_h1) -> dict:
    """Фичи не зависят от порогов — считаем один раз на монету."""
    return {sym: build_features(d15, d1h, None if sym == "BTC" else btc_h1) for sym, (d15, d1h) in data.items()}


def run(feats: dict, cfg: dict, start_time, end_time=None) -> list[dict]:
    trades = []
    for sym, f in feats.items():
        out = score_frame(f, cfg, is_btc=(sym == "BTC"))
        trades += simulate(out, sym, cfg, start_time, end_time)
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
             f"TF 15m + 1H · комиссия+проскальзывание {cost_pct(cfg):.3f}%/сторона",
             f"Порог силы: {cfg['signals']['min_score']} · мин. модулей: {cfg['signals']['min_agree']}", ""]
    if not s["n"]:
        return "\n".join(lines + ["Сделок нет — снизь min_score / min_agree."])
    g = summarize(trades, "r_gross")
    lines += ["## Итог (R после комиссий)", "",
              f"- Сделок: **{s['n']}** ({s['n'] / meta['days']:.1f} в день)",
              f"- Winrate: **{s['winrate']:.1f}%**",
              f"- Матожидание: **{s['avg_r']:+.3f}R** на сделку",
              f"- Сумма: **{s['total_r']:+.1f}R** · Profit factor: **{s['pf']:.2f}** · Макс. просадка: **{s['max_dd']:.1f}R**",
              f"- До комиссий: {g['avg_r']:+.3f}R на сделку (PF {g['pf']:.2f}) · комиссии съели "
              f"{g['total_r'] - s['total_r']:.1f}R (в среднем {np.mean([t['cost_r'] for t in trades]):.2f}R на сделку)",
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


def _cell(tr):
    s = summarize(tr, "r_net")
    if not s["n"]:
        return "0 | – | – | –"
    pf = "∞" if s["pf"] == float("inf") else f"{s['pf']:.2f}"
    return f"{s['n']} | {s['winrate']:.0f}% | {s['avg_r']:+.3f} | {pf}"


def experiments(feats, cfg, first, mid, last, meta) -> tuple[str, list]:
    """Все варианты на одних данных. Период A (старый) и B (свежий) считаются раздельно."""
    rows, summary = [], []
    for ex in cfg.get("experiments", []):
        c = load_config(overrides=ex.get("set") or {})
        a = run(feats, c, first, mid)
        b = run(feats, c, mid, last + pd.Timedelta(minutes=1))
        sa, sb = summarize(a, "r_net"), summarize(b, "r_net")
        rows.append(f"| {ex['name']} | {_cell(a)} | {_cell(b)} |")
        summary.append((ex["name"], sa, sb))
        log.info("exp %-40s A: n=%d avg=%+.3f | B: n=%d avg=%+.3f", ex["name"], sa.get("n", 0), sa.get("avg_r", 0),
                 sb.get("n", 0), sb.get("avg_r", 0))
    fmt = "%d.%m"
    text = "\n".join([
        f"# Эксперименты McQ Signals — {meta['date']}", "",
        f"Монет: {meta['n_symbols']} · источник: {meta['provider']} · 15m + 1H · R после комиссий", "",
        f"- **Период A** {first.strftime(fmt)}–{mid.strftime(fmt)} — старые данные, при настройке v2 их НЕ смотрели (честная проверка)",
        f"- **Период B** {mid.strftime(fmt)}–{last.strftime(fmt)} — по нему делали выводы после первого бэктеста", "",
        "| Вариант | A: сделок | A: WR | A: ср. R | A: PF | B: сделок | B: WR | B: ср. R | B: PF |",
        "|---|---|---|---|---|---|---|---|---|", *rows, "",
        "> Рабочий вариант — тот, что в плюсе в ОБОИХ периодах (особенно в A) и даёт хотя бы 1–2 сделки в день. "
        "Улучшение только в B при провале в A — это подгонка под историю.",
    ])
    return text, summary


def tg_experiments(summary, meta) -> str:
    lines = [f"🧪 <b>Эксперименты · {meta['n_symbols']} монет · 2×60д</b>", "A = старые 60д (честная проверка) · B = свежие 60д", ""]
    for name, a, b in summary:
        ok = "✅" if a.get("avg_r", -1) > 0 and b.get("avg_r", -1) > 0 else "▫️"
        lines.append(f"{ok} {name}\n   A: {a.get('avg_r', 0):+.3f}R ×{a.get('n', 0)} · B: {b.get('avg_r', 0):+.3f}R ×{b.get('n', 0)}")
    lines.append("\n<i>Ср. R на сделку после комиссий. Отчёт: reports/experiments_latest.md</i>")
    return "\n".join(lines)


def _broadcast(text):
    if not os.getenv("TELEGRAM_BOT_TOKEN"):
        return
    from src import store
    from src.telegram import TG

    tg = TG()
    subs = store.load("subscribers")
    tg.broadcast(list(dict.fromkeys(tg.fixed + [str(c) for c in subs["chats"]])), text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int)
    ap.add_argument("--symbols", type=int)
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--experiments", action="store_true")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--no-telegram", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg = load_config()
    days = a.days or (cfg["backtest"].get("experiments_days", 120) if a.experiments else cfg["backtest"]["days"])
    nsym = a.symbols or cfg["backtest"]["symbols"]
    data, btc, provider = load_synthetic(days, nsym) if a.synthetic else load_live(cfg, days, nsym)
    last = max(d[0]["close_time"].iloc[-1] for d in data.values())
    start = last - pd.Timedelta(days=days)
    meta = {"date": datetime.now(timezone.utc).strftime("%Y-%m-%d"), "days": days,
            "n_symbols": len(data), "provider": provider}
    feats = prepare(data, btc)
    send = not a.no_telegram and not a.synthetic
    REPORTS_DIR.mkdir(exist_ok=True)

    if a.experiments:
        mid = last - pd.Timedelta(days=days / 2)
        text, summary = experiments(feats, cfg, start, mid, last, meta)
        (REPORTS_DIR / "experiments_latest.md").write_text(text, encoding="utf-8")
        print(text)
        if send:
            _broadcast(tg_experiments(summary, meta))
        return

    if a.grid:
        rows = ["| min_score | min_agree | Сделок | Winrate | Ср. R | Сумма R | PF | 1-я пол. | 2-я пол. |",
                "|---|---|---|---|---|---|---|---|---|"]
        for ms in (62, 66, 70, 74, 78):
            for ma in (4, 5, 6):
                c = load_config(overrides={"signals": {"min_score": ms, "min_agree": ma}})
                tr = run(feats, c, start)
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

    trades = run(feats, cfg, start)
    md = report(trades, cfg, meta)
    (REPORTS_DIR / "backtest_latest.md").write_text(md, encoding="utf-8")
    if not a.synthetic:
        (REPORTS_DIR / f"backtest_{meta['date']}.md").write_text(md, encoding="utf-8")
    pd.DataFrame(trades).to_csv(REPORTS_DIR / "backtest_trades.csv", index=False)
    print(md)
    if send:
        _broadcast(tg_summary(trades, cfg, meta))


if __name__ == "__main__":
    main()
