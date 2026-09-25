from __future__ import annotations

import numpy as np


def summarize(trades: list[dict], key: str = "r_gross") -> dict:
    r = np.array([t[key] for t in trades], dtype=float)
    if r.size == 0:
        return {"n": 0}
    wins, losses = r[r > 0], r[r < 0]
    eq = np.cumsum(r)
    dd = float(np.max(np.maximum.accumulate(np.r_[0, eq]) - np.r_[0, eq]))
    by_exit: dict[str, int] = {}
    for t in trades:
        by_exit[t["exit_reason"]] = by_exit.get(t["exit_reason"], 0) + 1
    return {
        "n": int(r.size),
        "winrate": float((r > 0).mean() * 100),
        "total_r": float(r.sum()),
        "avg_r": float(r.mean()),
        "pf": float(wins.sum() / -losses.sum()) if losses.size else float("inf"),
        "max_dd": dd,
        "by_exit": by_exit,
    }
