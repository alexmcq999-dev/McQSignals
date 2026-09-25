"""Офлайн-тесты: python -m pytest -q"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from synth import resample, synth  # noqa: E402

from src import indicators as ta  # noqa: E402
from src.config import load_config  # noqa: E402
from src.engine import Trade, analyze, make_trade, update_trade  # noqa: E402


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def test_no_lookahead(cfg):
    """Сигнал на баре i не должен меняться, если добавить будущие бары."""
    btc = resample(synth(2000, seed=999))
    d = synth(2000, seed=3)
    h = resample(d)
    full = analyze(d, h, btc, cfg)
    cols = ["signal", "conf", "score"] + [c for c in full.columns if c.startswith("v_")]
    for cut in range(600, 2000, 131):
        dd = d.iloc[:cut]
        ct = dd["close_time"].iloc[-1]
        part = analyze(dd, h[h.close_time <= ct], btc[btc.close_time <= ct], cfg)
        a, b = part[cols].iloc[-1].astype(float), full[cols].iloc[cut - 1].astype(float)
        assert np.allclose(a.fillna(0), b.fillna(0), atol=1e-6), f"look-ahead на баре {cut}"


def test_supertrend_flips():
    st = ta.supertrend(synth(3000, seed=1))
    assert set(st.unique()) == {1.0, -1.0}
    assert (st != st.shift()).sum() > 10


def test_signals_both_sides(cfg):
    btc = resample(synth(3000, seed=999))
    L = S = 0
    for s in range(10):
        d = synth(3000, seed=s)
        o = analyze(d, resample(d), btc, cfg)
        L += (o.signal > 0).sum()
        S += (o.signal < 0).sum()
    assert L > 0 and S > 0


def _bar(h, l, c, t="2026-01-01T00:15:00+00:00"):
    return {"high": h, "low": l, "close": c, "close_time": pd.Timestamp(t)}


def _trade(side=1):
    e = 100.0
    return Trade(symbol="X", side=side, entry=e, sl=e - side * 2, tp1=e + side * 3, tp2=e + side * 6,
                 risk=2.0, opened="2026-01-01T00:00:00+00:00")


def test_trade_sl_first(cfg):
    t = _trade()
    ev = update_trade(t, _bar(107, 97, 100), cfg)  # и стоп, и TP2 в одной свече → стоп
    assert ev == ["sl"] and t.r_gross == -1.0


def test_trade_tp1_then_be(cfg):
    t = _trade()
    assert update_trade(t, _bar(103.5, 99.5, 103), cfg) == ["tp1"]
    assert t.sl == t.entry
    assert update_trade(t, _bar(101, 99.9, 100), cfg) == ["be"]
    assert t.r_gross == pytest.approx(0.75)


def test_trade_short_tp2(cfg):
    t = _trade(-1)
    assert update_trade(t, _bar(100.5, 93.5, 94), cfg) == ["tp1", "tp2"]
    assert t.r_gross == pytest.approx(0.5 * 1.5 + 0.5 * 3.0)


def test_levels_sane(cfg):
    btc = resample(synth(3000, seed=999))
    d = synth(3000, seed=2)
    o = analyze(d, resample(d), btc, cfg)
    for i in np.flatnonzero(o.signal.to_numpy() != 0)[:20]:
        row = o.iloc[i]
        t = make_trade(row, int(row.signal), "X", cfg)
        assert (t.side > 0 and t.sl < t.entry < t.tp1 < t.tp2) or (t.side < 0 and t.sl > t.entry > t.tp1 > t.tp2)
        assert t.risk <= t.entry * cfg["risk"]["sl_max_pct"] / 100 + 1e-9
        assert len(t.reasons) >= cfg["signals"]["min_agree"]


def test_scan_dry_run(monkeypatch, tmp_path, capsys):
    """Полный проход scan.py на синтетике: сигналы → сопровождение → состояние."""
    import scan
    from src import config, data, store

    monkeypatch.setattr(config, "STATE_DIR", tmp_path)
    monkeypatch.setattr(store, "STATE_DIR", tmp_path)
    monkeypatch.setattr(data, "STATE_DIR", tmp_path)

    # фиксированное время → детерминированный тест (не зависит от времени суток запуска)
    now = pd.Timestamp("2026-06-10 12:03", tz="UTC")
    n = 900
    start = now.floor("15min") - pd.Timedelta(minutes=15 * (n - 1))
    frames = {f"C{i}": synth(n, seed=i, start=str(start.tz_convert(None))) for i in range(40)}
    frames["BTC"] = synth(n, seed=999, start=str(start.tz_convert(None)))

    class Fake:
        name = "Fake"

        def tickers(self):
            return {k: 5e7 for k in frames}

        def klines(self, b, tf, limit, end_ms=None):
            return (frames[b] if tf == "15m" else resample(frames[b])).tail(limit).reset_index(drop=True)

    monkeypatch.setattr(scan, "pick_provider", lambda: (Fake(), Fake().tickers()))
    monkeypatch.setattr(data, "_coingecko_caps", lambda: [{"symbol": k, "name": k, "mcap": 2e8} for k in frames])
    monkeypatch.setattr(sys, "argv", ["scan.py", "--dry-run", "--now", now.isoformat()])
    # лояльный порог, чтобы гарантированно увидеть сигнал на последних свечах
    orig = scan.load_config
    monkeypatch.setattr(scan, "load_config", lambda: orig(overrides={
        "signals": {"min_score": 30, "min_agree": 3, "lookback_bars_on_run": 40, "htf_mode": "loose",
                    "btc_mode": "penalty", "strict_shorts": False},
        "risk": {"max_cost_r": 0}}))
    scan.main()
    st = store.load("signals")
    assert (tmp_path / "universe.json").exists()
    assert len(st["last_bar"]) >= 40
    out = capsys.readouterr().out
    assert len(st["open"]) > 0 and ("LONG" in out or "SHORT" in out)
    # второй прогон с тем же временем не должен дублировать сигналы
    before = len(st["open"])
    scan.main()
    assert len(store.load("signals")["open"]) == before


def _fake_series(n, step_ms=900_000, t0=1_780_000_000_000):
    return [(t0 + i * step_ms, 100 + i, 101 + i, 99 + i, 100.5 + i, 10 + i) for i in range(n)]


def test_providers_parsing(monkeypatch):
    """Парсинг и пагинация свечей всех провайдеров (на фейковых ответах API)."""
    from src import data

    rows = _fake_series(2500)

    def fake_get(url, params=None, headers=None, timeout=15):
        if "binance" in url:
            end = params.get("endTime", 10**15)
            sel = [r for r in rows if r[0] <= end][-params["limit"]:]
            return [[r[0], str(r[1]), str(r[2]), str(r[3]), str(r[4]), str(r[5]), r[0] + 899_999] for r in sel]
        if "kucoin" in url:
            sel = [r for r in rows if params["startAt"] * 1000 <= r[0] <= params["endAt"] * 1000]
            # time(s), open, close, high, low, volume, turnover — от новых к старым
            return {"data": [[str(r[0] // 1000), r[1], r[4], r[2], r[3], r[5], 0] for r in reversed(sel)][:1500]}
        if "okx" in url:
            after = params.get("after", 10**15)
            sel = [r for r in rows if r[0] < after][-params["limit"]:]
            return {"data": [[str(r[0]), r[1], r[2], r[3], r[4], r[5]] for r in reversed(sel)]}
        raise AssertionError(url)

    monkeypatch.setattr(data, "_get", fake_get)
    monkeypatch.setattr(data.time, "sleep", lambda s: None)
    end_ms = rows[-1][0] + 1
    for P in (data.Binance, data.KuCoin, data.OKX):
        df = P().klines("BTC", "15m", 1200, end_ms=end_ms)
        assert len(df) == 1200, P.name
        assert df["time"].is_monotonic_increasing and df["time"].is_unique, P.name
        last = df.iloc[-1]
        assert last["open"] == rows[-1][1] and last["close"] == rows[-1][4] and last["high"] == rows[-1][2], P.name
        assert (df["high"] >= df["low"]).all()


def test_v2_filters(cfg):
    """Строгие режимы только сокращают сигналы и не создают новых."""
    btc = resample(synth(3000, seed=999))
    d = synth(3000, seed=5)
    loose = load_config(overrides={"signals": {"htf_mode": "loose", "btc_mode": "penalty", "strict_shorts": False}})
    a = analyze(d, resample(d), btc, loose)
    b = analyze(d, resample(d), btc, cfg)
    lo = load_config(overrides={"signals": {"sides": "long_only"}})
    c = analyze(d, resample(d), btc, lo)
    assert (c.signal < 0).sum() == 0
    s_b = b[b.signal < 0]
    assert ((s_b.htf_bias == -1) & (s_b.btc_htf_bias == -1)).all()
    assert (b[b.signal > 0].htf_bias == 1).all()


def test_cost_filter_and_min_stop(cfg):
    from src.engine import trade_ok

    btc = resample(synth(3000, seed=999))
    d = synth(3000, seed=2)
    o = analyze(d, resample(d), btc, load_config(overrides={"signals": {"min_score": 40, "min_agree": 3}}))
    row = o.iloc[np.flatnonzero(o.signal.to_numpy() != 0)[0]]
    t = make_trade(row, int(row.signal), "X", cfg)
    assert t.cost_r == pytest.approx(2 * 0.105 / 100 * t.entry / t.risk, rel=1e-3)
    assert trade_ok(t, cfg) == (t.cost_r <= cfg["risk"]["max_cost_r"])
    wide = load_config(overrides={"risk": {"sl_min_pct": 3.0}})
    t2 = make_trade(row, int(row.signal), "X", wide)
    assert t2.risk >= t2.entry * 0.03 - 1e-9 and t2.cost_r < t.cost_r + 1e-12


def test_experiments_synthetic(tmp_path, monkeypatch):
    import backtest
    from src import config

    monkeypatch.setattr(backtest, "REPORTS_DIR", tmp_path)
    monkeypatch.setattr(sys, "argv", ["backtest.py", "--synthetic", "--experiments", "--days", "40", "--symbols", "4"])
    backtest.main()
    txt = (tmp_path / "experiments_latest.md").read_text()
    assert txt.count("\n| v") >= 5 and "Период A" in txt
