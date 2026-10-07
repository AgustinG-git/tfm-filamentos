"""Salidas de una ejecución según el contrato v1.

Una ejecución es una carpeta ``runs/<run_id>/`` con:

- ``run_config.json``    configuración resuelta, commit, split, selección
- ``val_predictions.csv`` instancias por imagen de validación (RLE oficial)
- ``val_logits.npz``     logits a 512 de cada imagen de validación
- ``history.csv``        métricas por época
- ``checkpoint.pth``     pesos del checkpoint elegido
- ``submission.csv``     (opcional) predicciones de test

El evaluador y el análisis solo leen estas salidas: no saben cómo se
generaron las predicciones.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils

from . import contract
from .contract import RUN_FILES, ContractError
from .metrics import PRED_ID, check_ids, find_overlaps, image_of
from .split import file_sha256, stems_of


# ── Identificación ───────────────────────────────────────────────────
def git_commit(repo_dir: str | Path = ".") -> str:
    """Commit actual; en Kaggle sin .git, la variable FILSEG_COMMIT."""
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return os.environ.get("FILSEG_COMMIT", "unknown")


def make_run_id(name: str, commit: str, when: datetime | None = None) -> str:
    when = when or datetime.now(timezone.utc)
    return f"{when:%Y%m%d-%H%M}_{name}_{commit[:7]}"


def make_run_config(
    *,
    run_id: str,
    config: dict,
    split_path: str | Path,
    git_commit: str,
    selection: dict,
    postproc: dict,
    metrics_val_A: dict,
    **extra,
) -> dict:
    return {
        "run_id": run_id,
        "contract_version": contract.CONTRACT_VERSION,
        "split_version": contract.SPLIT_VERSION,
        "split_sha256": file_sha256(split_path),
        "git_commit": git_commit,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": config,
        "selection": selection,
        "postproc": postproc,
        "metrics_val_A": metrics_val_A,
        **extra,
    }


# ── Predicciones ─────────────────────────────────────────────────────
def labels_to_rles(labels: np.ndarray, n: int) -> list[str]:
    """Mapa de etiquetas (0 = fondo) -> RLE oficial por instancia."""
    out = []
    for i in range(1, n + 1):
        rle = mask_utils.encode(np.asfortranarray((labels == i).astype(np.uint8)))
        out.append(rle["counts"].decode("utf-8"))
    return out


def predictions_frame(preds: Mapping[str, Sequence[str]]) -> pd.DataFrame:
    """``{stem: [rle, ...]}`` -> DataFrame oficial (``<stem>_<n>``, n desde 1)."""
    rows = [
        {"filament_id": f"{stem}_{j}", "segmentation_rle": rle}
        for stem, rles in preds.items()
        for j, rle in enumerate(rles, 1)
    ]
    return pd.DataFrame(rows, columns=list(contract.PRED_COLUMNS))


# ── Logits ───────────────────────────────────────────────────────────
def save_logits(path: str | Path, stems: Sequence[str], logits: np.ndarray) -> None:
    logits = np.asarray(logits, dtype=contract.LOGITS_DTYPE)
    if logits.shape != (len(stems), contract.TRAIN_SIZE, contract.TRAIN_SIZE):
        raise ContractError(f"Logits con forma {logits.shape}")
    np.savez_compressed(path, stems=np.asarray(stems, dtype=str), logits=logits)


def load_logits(path: str | Path) -> tuple[list[str], np.ndarray]:
    with np.load(path) as z:
        return z["stems"].tolist(), z["logits"]


# ── Ejecución completa ───────────────────────────────────────────────
def write_run(
    run_dir: str | Path,
    *,
    run_config: dict,
    val_predictions: pd.DataFrame,
    history: pd.DataFrame,
    val_logits: tuple[Sequence[str], np.ndarray],
) -> Path:
    """Escribe las salidas (el checkpoint lo guarda el entrenamiento)."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / RUN_FILES["config"]).write_text(
        json.dumps(run_config, indent=1, ensure_ascii=False, default=str) + "\n"
    )
    val_predictions.to_csv(run_dir / RUN_FILES["val_predictions"], index=False)
    history.to_csv(run_dir / RUN_FILES["history"], index=False)
    save_logits(run_dir / RUN_FILES["val_logits"], *val_logits)
    return run_dir


@dataclass
class Run:
    path: Path
    config: dict
    val_predictions: pd.DataFrame
    history: pd.DataFrame

    @property
    def run_id(self) -> str:
        return self.config["run_id"]

    def logits(self) -> tuple[list[str], np.ndarray]:
        return load_logits(self.path / RUN_FILES["val_logits"])


def read_run(run_dir: str | Path) -> Run:
    run_dir = Path(run_dir)
    return Run(
        path=run_dir,
        config=json.loads((run_dir / RUN_FILES["config"]).read_text()),
        val_predictions=pd.read_csv(
            run_dir / RUN_FILES["val_predictions"], dtype=str, keep_default_na=False,
        ),
        history=pd.read_csv(run_dir / RUN_FILES["history"]),
    )


def validate_run(
    run_dir: str | Path,
    split: dict,
    split_path: str | Path | None = None,
    check_overlaps: bool = True,
) -> list[str]:
    """Lista de incumplimientos del contrato (vacía = válida)."""
    run_dir = Path(run_dir)
    problems = [f"falta {f}" for f in RUN_FILES.values() if not (run_dir / f).exists()]
    if problems:
        return problems

    run = read_run(run_dir)
    cfg = run.config
    problems += [f"run_config sin '{k}'" for k in contract.REQUIRED_CONFIG_KEYS if k not in cfg]
    if cfg.get("contract_version") != contract.CONTRACT_VERSION:
        problems.append(f"contrato {cfg.get('contract_version')} ≠ {contract.CONTRACT_VERSION}")
    if cfg.get("split_version") != contract.SPLIT_VERSION:
        problems.append(f"split {cfg.get('split_version')} ≠ {contract.SPLIT_VERSION}")
    if split_path is not None and cfg.get("split_sha256") != file_sha256(split_path):
        problems.append("split_sha256 no coincide con el split cargado")
    if cfg.get("git_commit", "unknown") == "unknown":
        problems.append("git_commit desconocido: la ejecución no es trazable")
    if cfg.get("selection", {}).get("subset") not in (None, contract.TUNE_SUBSET):
        problems.append("checkpoint elegido fuera de val_A")

    preds = run.val_predictions
    if tuple(preds.columns) != contract.PRED_COLUMNS:
        problems.append(f"columnas {tuple(preds.columns)} ≠ {contract.PRED_COLUMNS}")
        return problems
    try:
        check_ids(preds, PRED_ID, "Predicciones")
    except ContractError as e:
        problems.append(str(e))
    val_stems = stems_of(split, "val_A", "val_B")
    outside = set(image_of(preds["filament_id"])) - val_stems
    if outside:
        problems.append(f"{len(outside)} imágenes predichas fuera de val: {sorted(outside)[:3]}")
    if check_overlaps and len(preds):
        overlapping = find_overlaps(preds)
        if overlapping:
            problems.append(f"{len(overlapping)} imágenes con predicciones solapadas")

    stems, logits = run.logits()
    if set(stems) != val_stems:
        problems.append("val_logits no cubre exactamente las imágenes de val")
    if logits.dtype != np.dtype(contract.LOGITS_DTYPE):
        problems.append(f"logits en {logits.dtype}, se espera {contract.LOGITS_DTYPE}")
    return problems


def assert_valid_run(run_dir, split, split_path=None) -> None:
    problems = validate_run(run_dir, split, split_path)
    if problems:
        raise ContractError(f"{run_dir} no cumple el contrato:\n- " + "\n- ".join(problems))
