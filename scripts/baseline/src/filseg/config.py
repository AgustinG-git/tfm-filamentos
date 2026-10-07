"""Carga de configuraciones YAML con herencia (clave ``hereda``)."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path) -> dict:
    """Lee un YAML y lo fusiona con el que hereda (recursivo)."""
    path = Path(path)
    cfg = yaml.safe_load(path.read_text()) or {}
    parent = cfg.pop("hereda", None)
    if parent:
        cfg = _merge(load_config(path.parent / parent), cfg)
    return cfg
