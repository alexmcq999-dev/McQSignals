from __future__ import annotations

import copy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = ROOT / "state"
REPORTS_DIR = ROOT / "reports"

TF_MINUTES = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240, "1d": 1440}


def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


DEFAULTS = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8")) if (ROOT / "config.yaml").exists() else {}


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    p = Path(path) if path else ROOT / "config.yaml"
    cfg = yaml.safe_load(p.read_text(encoding="utf-8")) if p.exists() else {}
    cfg = _deep_merge(DEFAULTS, cfg)
    return _deep_merge(cfg, overrides or {})
