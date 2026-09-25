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


def prepare(data: dict, btc_htf, fng=None) -> dict:
    """Фичи не зависят от порогов — считаем один раз на монету."""
    return {sym: build_features(dE, dC, None if sym == "BTC" else btc_htf, crowd, fng)
            for sym, (dE, dC, crowd) in data.items()}


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
             f"TF {cfg['timeframes']['entry'].upper()} + {cfg['timeframes']['confirm'].upper()} · "
             f"выход: {cfg['risk'].get('exit_mode', 'fixed')} · комиссия+проскальзывание {cost_pct(cfg):.3f}%/сторона",
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
    lines += [_row("Трендовые сигналы", [t for t in trades if t.get("kind", "trend") == "trend"]),
              _row("Контртрендовые (против толпы)", [t for t in trades if t.get("kind") == "contra"])]
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

RULE = {"15m": "15min", "30m": "30min", "1h": "1h", "4h": "4h", "1d": "1D"}


def load_live(cfg, days, n_symbols):
    from src import crowd as cw
    from src.config import ROOT, TF_MINUTES
    from src.data import build_universe, fetch_many, pick_provider, split_closed

    tf = cfg["timeframes"]
    provider, tickers = pick_provider()
    uni = build_universe(cfg, provider, tickers)[:n_symbols]
    bases = [a["base"] for a in uni]
    nE = int(days * 1440 / TF_MINUTES[tf["entry"]]) + 400
    nC = int(days * 1440 / TF_MINUTES[tf["confirm"]]) + 260
    kE = fetch_many(provider, bases, tf["entry"], nE)
    kC = fetch_many(provider, list(dict.fromkeys(bases + ["BTC"])), tf["confirm"], nC)
    now = pd.Timestamp.now(tz="UTC")

    crowd_tbls, fng = {}, None
    cr = cfg.get("contrarian", {})
    if cr.get("enabled") or cr.get("crowd_filter") or cr.get("fng_filter") or any(
            (e.get("set") or {}).get("contrarian") for e in cfg.get("experiments", [])):
        ndays = days + cfg.get("crowd", {}).get("z_window_days", 14) + 3
        cache = cw.fetch_metrics(bases, cw.recent_days(now, ndays), ROOT / ".cache" / "crowd" / "metrics.json")
        crowd_tbls = {b: cw.crowd_table(cache, b, cfg) for b in bases}
        fng = cw.fetch_fng(ROOT / ".cache" / "crowd" / "fng.json")
        log.info("Толпа: данные есть по %d/%d монетам; F&G: %s", sum(v is not None for v in crowd_tbls.values()),
                 len(bases), "да" if fng is not None else "нет")

    data = {b: (split_closed(kE[b], now)[0], split_closed(kC[b], now)[0], crowd_tbls.get(b))
            for b in bases if b in kE and b in kC}
    btc = split_closed(kC["BTC"], now)[0] if "BTC" in kC else None
    return data, btc, fng, provider.name


def load_synthetic(cfg, days, n_symbols):
    import sys
    sys.path.insert(0, "tests")
    from synth import resample, synth

    from src.config import TF_MINUTES

    tf = cfg["timeframes"]
    m = TF_MINUTES[tf["entry"]]
    n = int(days * 1440 / m) + 400
    rule = RULE[tf["confirm"]]
    btc = synth(n, seed=999, tf_min=m)
    data = {}
    for i in range(n_symbols):
        d = synth(n, seed=i, tf_min=m)
        data[f"SYN{i}"] = (d, resample(d, rule), None)
    return data, resample(btc, rule), None, "synthetic"


def _cell(tr):
    s = summarize(tr, "r_net")
    if not s["n"]:
        return "0 | – | – | –"
    pf = "∞" if s["pf"] == float("inf") else f"{s['pf']:.2f}"
    return f"{s['n']} | {s['winrate']:.0f}% | {s['avg_r']:+.3f} | {pf}"


def experiments(feats, cfg, first, last, meta) -> tuple[str, list]:
    """Все варианты на одних данных, период разбит на k фолдов — каждый считается отдельно."""
    k = int(cfg["backtest"].get("folds", 3))
    edges = [first + (last - first) * i / k for i in range(k + 1)]
    edges[-1] = last + pd.Timedelta(minutes=1)
    rows, summary = [], []
    for ex in cfg.get("experiments", []):
        c = load_config(overrides=ex.get("set") or {})
        folds = [run(feats, c, edges[i], edges[i + 1]) for i in range(k)]
        ss = [summarize(tr, "r_net") for tr in folds]
        allr = summarize(sum(folds, []), "r_net")
        ok = all(x.get("n", 0) >= 10 and x.get("avg_r", -1) > 0 for x in ss)
        cells = " | ".join(f"{x.get('n', 0)} / {x.get('avg_r', 0):+.3f}" for x in ss)
        pf = allr.get("pf", 0)
        rows.append(f"| {'✅' if ok else '▫️'} {ex['name']} | {cells} | {allr.get('n', 0)} | "
                    f"{allr.get('winrate', 0):.0f}% | {allr.get('avg_r', 0):+.3f} | "
                    f"{'∞' if pf == float('inf') else f'{pf:.2f}'} | {allr.get('max_dd', 0):.1f} |")
        summary.append((ex["name"], ss, allr, ok))
        log.info("exp %-45s %s", ex["name"], cells)
    fmt = "%d.%m.%y"
    heads = " | ".join(f"П{i + 1} {edges[i].strftime(fmt)}–{edges[i + 1].strftime(fmt)}" for i in range(k))
    text = "\n".join([
        f"# Эксперименты McQ Signals — {meta['date']}", "",
        f"Монет: {meta['n_symbols']} · источник: {meta['provider']} · "
        f"{cfg['timeframes']['entry'].upper()} + {cfg['timeframes']['confirm'].upper()} · R после комиссий", "",
        "В ячейках периодов: `сделок / средний R на сделку`. ✅ — в плюсе во ВСЕХ периодах (и ≥10 сделок в каждом).", "",
        f"| Вариант | {heads} | Всего сделок | WR | Ср. R | PF | Макс. просадка R |",
        "|---" * (k + 6) + "|", *rows, "",
        "> Рабочий вариант — ✅ и с разумным числом сделок. Лучший результат только в одном периоде — это "
        "подгонка под конкретный рынок, а не преимущество.",
    ])
    return text, summary


def tg_experiments(summary, meta) -> str:
    lines = [f"🧪 <b>Эксперименты · {meta['n_symbols']} монет · {meta['days']}д</b>",
             "Ср. R на сделку по периодам (после комиссий):", ""]
    for name, ss, allr, ok in summary:
        per = " · ".join(f"{x.get('avg_r', 0):+.2f}" for x in ss)
        lines.append(f"{'✅' if ok else '▫️'} {name}\n   {per} (всего {allr.get('n', 0)})")
    lines.append("\n✅ — в плюсе во всех периодах. Отчёт: reports/experiments_latest.md")
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
    data, btc, fng, provider = load_synthetic(cfg, days, nsym) if a.synthetic else load_live(cfg, days, nsym)
    last = max(d[0]["close_time"].iloc[-1] for d in data.values())
    start = last - pd.Timedelta(days=days)
    meta = {"date": datetime.now(timezone.utc).strftime("%Y-%m-%d"), "days": days,
            "n_symbols": len(data), "provider": provider}
    feats = prepare(data, btc, fng)
    send = not a.no_telegram and not a.synthetic
    REPORTS_DIR.mkdir(exist_ok=True)

    if a.experiments:
        text, summary = experiments(feats, cfg, start, last, meta)
        import json

        from src.config import STATE_DIR
        (STATE_DIR / "experiments.json").write_text(json.dumps({
            "date": meta["date"], "days": days, "coins": meta["n_symbols"],
            "rows": [{"name": n, "ok": ok, "folds": [round(x.get("avg_r", 0), 3) for x in ss],
                      "n": allr.get("n", 0), "avg_r": round(allr.get("avg_r", 0), 3),
                      "pf": (None if allr.get("pf") == float("inf") else round(allr.get("pf", 0), 2))}
                     for n, ss, allr, ok in summary]}, ensure_ascii=False), encoding="utf-8")
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
