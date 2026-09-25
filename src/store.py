"""Состояние бота в JSON-файлах (коммитятся обратно в репозиторий из Actions)."""
from __future__ import annotations

import json
from pathlib import Path

from .config import STATE_DIR

FILES = {
    "signals": {"open": [], "last_bar": {}, "cooldown": {}, "seq": 0, "last_daily": ""},
    "history": {"closed": []},
    "subscribers": {"chats": [], "offset": 0},
}


def _p(name) -> Path:
    return STATE_DIR / f"{name}.json"


def load(name: str) -> dict:
    p = _p(name)
    base = json.loads(json.dumps(FILES[name]))
    if p.exists():
        try:
            base.update(json.loads(p.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            pass
    return base


def save(name: str, data: dict):
    STATE_DIR.mkdir(exist_ok=True)
    tmp = _p(name).with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    tmp.replace(_p(name))
