"""Бэктест той же логики, что и в боевом сканере.

  python backtest.py                    — по config.yaml (60 дней, топ-25 монет)
  python backtest.py --days 90 --symbols 40
  python backtest.py --grid             — перебор порогов min_score / min_agree
  python backtest.py --experiments      — сравнение вариантов из config.yaml на 2 периодах по 60 дней
  python backtest.py --synthetic        — офлайн-проверка на синтетике (без сети)
  python backtest.py --replay           — сверка живых сигналов (state/history.json) с бэктестом того же окна

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
                if t.exit_reason != "expired":  # неисполненный лимит — не сделка
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


def prepare(data: dict, btc_htf, fng=None, macro=None) -> dict:
    """Фичи не зависят от порогов — считаем один раз на монету."""
    return {sym: build_features(dE, dC, None if sym == "BTC" else btc_htf, crowd, fng, macro)
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
              _row("Пробойные сетапы", [t for t in trades if t.get("kind") == "breakout"]),
              _row("Пробой: вход по рынку", [t for t in trades if t.get("kind") == "breakout" and not t.get("filled")]),
              _row("Пробой: лимит на ретест", [t for t in trades if t.get("kind") == "breakout" and t.get("filled")]),
              _row("Контртрендовые (против толпы)", [t for t in trades if t.get("kind") == "contra"])]
    if any(t.get("macro") for t in trades):
        lines += [_row("Макро: по направлению сделки", [t for t in trades if t.get("macro") == "with"]),
                  _row("Макро: против сделки", [t for t in trades if t.get("macro") == "against"]),
                  _row("Макро: нейтрально / связи нет", [t for t in trades if t.get("macro") == "neutral"])]
    for reg in ("trend", "neutral", "range"):
        lines.append(_row(f"Режим: {reg}", [t for t in trades if t["regime"] == reg]))
    for lo, hi in ((0, 70), (70, 75), (75, 80), (80, 85), (85, 101)):
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

MACRO_NOTE = [""]  # строка о макро-данных для отчёта экспериментов

RULE = {"15m": "15min", "30m": "30min", "1h": "1h", "4h": "4h", "1d": "1D"}


def load_live(cfg, days, n_symbols, bases=None):
    from src import crowd as cw
    from src import macro as mc
    from src.config import ROOT, TF_MINUTES
    from src.data import build_universe, fetch_many, pick_provider, split_closed

    tf = cfg["timeframes"]
    provider, tickers = pick_provider()
    if bases is None:
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

    macro_tbl = None
    if mc.active(cfg) or any((e.get("set") or {}).get("macro") for e in cfg.get("experiments", [])):
        mcfg = cfg if mc.active(cfg) else load_config(overrides={"macro": {"mode": "shadow"}})
        try:
            macro_tbl = mc.build_live(mcfg, provider, bases, days=days + mc.warmup_days(mcfg) + 14)
        except Exception as e:  # noqa: BLE001
            log.warning("Макро: %s", e)
        log.info("Макро: %s · %s", f"{len(macro_tbl)} недель" if macro_tbl is not None else "нет данных", mc.DIAG)
        if macro_tbl is not None:
            mstat = mc.summary(macro_tbl, mcfg, pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days))
            src = ", ".join(f"{k}: {v.get('source')}" for k, v in mc.DIAG.get("drivers", {}).items())
            per = "; ".join(f"{d['name']} корр. {v['corr_min']:+.2f}…{v['corr_max']:+.2f}, связь активна "
                            f"{v['on_pct']:.0f}% недель" for d in mc.enabled_drivers(mcfg)
                            if isinstance(v := mstat.get(d["name"]), dict))
            MACRO_NOTE[0] = (f"Макро-данные: {src} · корзина {len(mc.DIAG.get('basket', {}).get('coins', []))} монет · "
                             f"{per} · фильтр действовал {mstat['bias_on_pct']:.0f}% недель периода")
        else:
            errs = "; ".join(f"{k}: {' | '.join(v.get('errors', []))}" for k, v in mc.DIAG.get("drivers", {}).items())
            MACRO_NOTE[0] = f"⚠️ Макро-данных НЕТ — варианты с макро совпадут с базовым. {errs}"

    data = {b: (split_closed(kE[b], now)[0], split_closed(kC[b], now)[0], crowd_tbls.get(b))
            for b in bases if b in kE and b in kC}
    btc = split_closed(kC["BTC"], now)[0] if "BTC" in kC else None
    return data, btc, fng, provider.name, macro_tbl


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
    return data, resample(btc, rule), None, "synthetic", None


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
        *([MACRO_NOTE[0], ""] if MACRO_NOTE[0] else []),
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


# ------------------------------------------------------------------ сверка с живой работой

def _entry_features(f: pd.DataFrame, opened, side: int) -> dict:
    """Где находилась цена в момент живого сигнала: насколько «догоняли»."""
    ct = pd.to_datetime(f["close_time"])
    ts = pd.Timestamp(opened)
    idx = ct.searchsorted(ts, side="right") - 1
    if idx < 50:
        return {}
    r = f.iloc[idx]
    w = f.iloc[idx - 49:idx + 1]
    hi, lo, a = float(w["high"].max()), float(w["low"].min()), float(r["atr"])
    pos = (float(r["close"]) - lo) / max(hi - lo, 1e-12)

    def g(c, nd=2):
        v = r.get(c)
        return None if v is None or pd.isna(v) else round(float(v), nd)

    return {
        "bar": pd.Timestamp(r["close_time"]).isoformat(),
        "ext_ema20_atr": round(side * (float(r["close"]) - float(r["ema20"])) / a, 2),
        "ext_ema50_atr": round(side * (float(r["close"]) - float(r["ema50"])) / a, 2),
        "rsi": g("rsi", 1),
        "range_pos": round(pos if side > 0 else 1 - pos, 2),  # 1.0 = у самого экстремума 50 свечей по направлению
        "move_10_atr": round(side * (float(r["close"]) - float(f["close"].iloc[idx - 10])) / a, 2),
        "vol_z": g("vol_z"), "adx": g("adx", 0), "atr_pct": g("atr_pct"), "crowd_z": g("ls_acc_z"), "fng": g("fng", 0),
    }


def replay_report(live: list[dict], feats: dict, cfg: dict, tolerance_h: float = 2) -> tuple[str, list[dict]]:
    """Прогоняет бэктест на окне живых сделок и сопоставляет сделку к сделке."""
    first = min(pd.Timestamp(t["opened"]) for t in live) - pd.Timedelta(hours=1)
    sub = {s: f for s, f in feats.items() if s in {t["symbol"] for t in live}}
    bt = run(sub, cfg, first)
    tol = pd.Timedelta(hours=tolerance_h)
    rows, used = [], set()
    for t in sorted(live, key=lambda x: x["opened"]):
        side = t["side"] if isinstance(t["side"], int) else (1 if t["side"] == "LONG" else -1)
        best = None
        for j, b in enumerate(bt):
            if j in used or b["symbol"] != t["symbol"] or b["side"] != side:
                continue
            dt = abs(pd.Timestamp(b["opened"]) - pd.Timestamp(t["opened"]))
            if dt <= tol and (best is None or dt < best[1]):
                best = (j, dt)
        b = bt[best[0]] if best else None
        if best:
            used.add(best[0])
        fe = _entry_features(sub[t["symbol"]], t["opened"], side) if t["symbol"] in sub else {}
        rows.append({"id": t.get("id", ""), "symbol": t["symbol"], "side": side, "opened": t["opened"],
                     "live_exit": t.get("exit_reason", ""), "live_r": t.get("r_gross"), "matched": bool(b),
                     "bt_exit": b["exit_reason"] if b else "", "bt_r": b["r_gross"] if b else None,
                     "stop_atr": round(t["risk"] / t["atr"], 2) if t.get("atr") else None,
                     "stop_pct": round(t["risk"] / t["entry"] * 100, 2), **fe})
    live_r = [r["live_r"] for r in rows if r["live_r"] is not None]
    m = [r for r in rows if r["matched"]]
    extra = [b for j, b in enumerate(bt) if j not in used]
    bts = summarize(bt, "r_gross")

    def avg(key, rs):
        v = [r[key] for r in rs if r.get(key) is not None]
        return f"{np.mean(v):+.2f}" if v else "–"

    win = [r for r in rows if (r["live_r"] or 0) > 0]
    loss = [r for r in rows if (r["live_r"] or 0) <= 0]
    btf = [_entry_features(sub[b["symbol"]], b["opened"], b["side"]) for b in bt]
    keys = [("ext_ema20_atr", "От EMA20, ATR"), ("ext_ema50_atr", "От EMA50, ATR"), ("rsi", "RSI"),
            ("range_pos", "Позиция в диапазоне 50 свечей (1 = экстремум)"), ("move_10_atr", "Ход за 10 свечей, ATR"),
            ("stop_atr", "Стоп, ATR"), ("stop_pct", "Стоп, %"), ("crowd_z", "Толпа z"), ("fng", "F&G")]
    lines = [
        f"# Сверка живых сигналов с бэктестом — {datetime.now(timezone.utc):%Y-%m-%d}", "",
        f"Окно: с {first:%d.%m.%Y %H:%M} UTC · монет: {len(sub)} · режим бэктеста: "
        f"{cfg.get('strategy', {}).get('mode', 'trend')} · 4H {cfg['signals'].get('htf_mode')} · "
        f"BTC {cfg['signals'].get('btc_mode')} · вход {cfg['signals'].get('entry_mode', 'market')}", "",
        "## Итог", "",
        f"- Живых сделок: **{len(rows)}**, сумма **{sum(live_r):+.2f}R**, в плюс {len(win)}",
        f"- Бэктест на тех же монетах и окне: **{bts.get('n', 0)}** сделок, сумма **{bts.get('total_r', 0):+.2f}R**, "
        f"winrate {bts.get('winrate', 0):.0f}%",
        f"- Совпало сделка-в-сделку (±{tolerance_h:g}ч): **{len(m)}/{len(rows)}**; бэктест дал ещё {len(extra)} сделок, "
        f"которых не было вживую",
        f"- По совпавшим: живые {sum(r['live_r'] or 0 for r in m):+.2f}R против бэктеста "
        f"{sum(r['bt_r'] or 0 for r in m):+.2f}R (разница = исполнение/сопровождение, а не стратегия)", "",
        "## Где были входы", "",
        "| Признак | Живые: прибыльные | Живые: убыточные | Все сделки бэктеста в окне |", "|---|---|---|---|",
        *[f"| {nm} | {avg(k, win)} | {avg(k, loss)} | {avg(k, btf) if k not in ('stop_atr', 'stop_pct') else '–'} |"
          for k, nm in keys], "",
        "## Сделки", "",
        "| # | Монета | Сторона | Свеча | Живой итог | Бэктест | От EMA20 ATR | RSI | Позиция | Ход 10 св. | Стоп % |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        bt_s = f"{r['bt_exit']} {r['bt_r']:+.2f}R" if r["matched"] else "нет сделки"
        lines.append(f"| {r['id']} | {r['symbol']} | {'L' if r['side'] > 0 else 'S'} | {r['opened'][:16]} | "
                     f"{r['live_exit']} {(r['live_r'] or 0):+.2f}R | {bt_s} | {r.get('ext_ema20_atr', '–')} | "
                     f"{r.get('rsi', '–')} | {r.get('range_pos', '–')} | {r.get('move_10_atr', '–')} | {r['stop_pct']} |")
    lines += ["", "> Позиция 0.9–1.0 и ход за 10 свечей > 3 ATR = вход после того, как движение уже случилось "
              "(«на хаях/лоях»). Если совпавшие сделки и в бэктесте убыточны — проблема в логике входа, "
              "если в бэктесте плюс — в исполнении/сопровождении."]
    return "\n".join(lines), rows


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
    ap.add_argument("--replay", action="store_true")
    ap.add_argument("--since", default="2026-09-25T10:14:00+00:00", help="--replay: живые сделки с этого времени")
    ap.add_argument("--replay-mode", default="trend", help="--replay: режим стратегии для бэктеста (как был вживую)")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if a.replay:
        import json

        from src import store
        hist = store.load("history")
        since = pd.Timestamp(a.since)
        live = [t for t in hist["closed"] if pd.Timestamp(t["opened"]) >= since]
        if not live:
            print("Живых сделок в окне нет")
            return
        cfg = load_config(overrides={"strategy": {"mode": a.replay_mode}})
        bases = sorted({t["symbol"] for t in live})
        days = (pd.Timestamp.now(tz="UTC") - since).days + 3
        data, btc, fng, provider, macro_tbl = load_live(cfg, days, len(bases), bases=bases)
        feats = prepare(data, btc, fng, macro_tbl)
        md, rows = replay_report(live, feats, cfg)
        REPORTS_DIR.mkdir(exist_ok=True)
        (REPORTS_DIR / "replay_latest.md").write_text(md, encoding="utf-8")
        (REPORTS_DIR / "replay_latest.json").write_text(json.dumps(rows, ensure_ascii=False, default=str),
                                                        encoding="utf-8")
        print(md)
        return

    cfg = load_config()
    days = a.days or (cfg["backtest"].get("experiments_days", 120) if a.experiments else cfg["backtest"]["days"])
    nsym = a.symbols or cfg["backtest"]["symbols"]
    data, btc, fng, provider, macro_tbl = (load_synthetic(cfg, days, nsym) if a.synthetic
                                           else load_live(cfg, days, nsym))
    last = max(d[0]["close_time"].iloc[-1] for d in data.values())
    start = last - pd.Timedelta(days=days)
    meta = {"date": datetime.now(timezone.utc).strftime("%Y-%m-%d"), "days": days,
            "n_symbols": len(data), "provider": provider}
    feats = prepare(data, btc, fng, macro_tbl)
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
