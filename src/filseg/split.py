"""Split congelado train / val_A / val_B, agrupado por fecha.

Reproduce exactamente el split de la baseline original:

1. ``GroupShuffleSplit(test_size=0.2, random_state=42)`` por fecha sobre
   todos los pares anotador-imagen -> train / val.
2. ``GroupShuffleSplit(test_size=0.5, random_state=42)`` por fecha sobre
   val -> val_A (la parte "train" del segundo corte) / val_B.

``GroupShuffleSplit`` reparte días, no filas, y su resultado solo depende
del conjunto de fechas; por eso es independiente del orden de las filas.

El split se genera UNA vez, se guarda en ``splits/split_v1.json`` y se
versiona en git. Ninguna ejecución lo recalcula: todas lo cargan.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import sklearn
from sklearn.model_selection import GroupShuffleSplit

from . import contract
from .contract import ContractError


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _summary(meta: pd.DataFrame, ids: list[str]) -> dict:
    part = meta[meta["annotator_image"].isin(ids)]
    return {
        "annotations": len(part),
        "images": part["stem"].nunique(),
        "days": part["date"].nunique(),
        "filaments": int(part["n_filaments"].sum()),
    }


def make_split(meta: pd.DataFrame, annotations_path: str | Path | None = None) -> dict:
    """Crea el split v1 a partir de ``gt.build_meta``."""
    gss = GroupShuffleSplit(
        n_splits=1, test_size=contract.VAL_SIZE, random_state=contract.SPLIT_SEED,
    )
    tr, va = next(gss.split(meta, groups=meta["date"]))
    val = meta.iloc[va].reset_index(drop=True)

    gss_half = GroupShuffleSplit(
        n_splits=1, test_size=contract.HALF_SIZE, random_state=contract.SPLIT_SEED,
    )
    a, b = next(gss_half.split(val, groups=val["date"]))

    subsets = {
        "train": meta.iloc[tr]["annotator_image"].tolist(),
        "val_A": val.iloc[a]["annotator_image"].tolist(),
        "val_B": val.iloc[b]["annotator_image"].tolist(),
    }
    source = {}
    if annotations_path is not None:
        source = {
            "annotations_file": Path(annotations_path).name,
            "sha256": file_sha256(annotations_path),
        }
    split = {
        "contract_version": contract.CONTRACT_VERSION,
        "split_version": contract.SPLIT_VERSION,
        "params": {
            "method": "sklearn.model_selection.GroupShuffleSplit",
            "groups": "date = stem[:8]",
            "val_size": contract.VAL_SIZE,
            "half_size": contract.HALF_SIZE,
            "seed": contract.SPLIT_SEED,
            "sklearn_version": sklearn.__version__,
        },
        "source": source,
        "summary": {k: _summary(meta, v) for k, v in subsets.items()},
        "subsets": subsets,
    }
    check_split(split, meta)
    return split


def check_split(split: dict, meta: pd.DataFrame | None = None) -> None:
    """Sin solapes de pares, imágenes ni días entre subconjuntos."""
    subsets = split["subsets"]
    if tuple(subsets) != contract.SUBSETS:
        raise ContractError(f"Subconjuntos {tuple(subsets)} ≠ {contract.SUBSETS}")
    owner = {}
    for name, ids in subsets.items():
        for ai in ids:
            if ai in owner:
                raise ContractError(f"{ai} está en {owner[ai]} y en {name}")
            owner[ai] = name

    stems = {n: {ai.split("-", 1)[1] for ai in ids} for n, ids in subsets.items()}
    for x, y in [("train", "val_A"), ("train", "val_B"), ("val_A", "val_B")]:
        shared = stems[x] & stems[y]
        if shared:
            raise ContractError(f"Imágenes en {x} y {y}: {sorted(shared)[:3]}")
        days = {s[:8] for s in stems[x]} & {s[:8] for s in stems[y]}
        if days:
            raise ContractError(f"Días en {x} y {y}: {sorted(days)[:3]}")

    if meta is not None:
        missing = set(meta["annotator_image"]) - set(owner)
        extra = set(owner) - set(meta["annotator_image"])
        if missing or extra:
            raise ContractError(
                f"El split no cubre el JSON: faltan {len(missing)}, sobran {len(extra)}"
            )


def save_split(split: dict, path: str | Path = contract.SPLIT_PATH) -> Path:
    path = Path(path)
    if path.exists():
        raise ContractError(f"{path} ya existe: el split v1 no se sobrescribe")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split, indent=1, ensure_ascii=False) + "\n")
    return path


def load_split(
    path: str | Path = contract.SPLIT_PATH,
    annotations_path: str | Path | None = None,
) -> dict:
    """Carga el split y, si se da el JSON, verifica que es el mismo."""
    split = json.loads(Path(path).read_text())
    if split.get("split_version") != contract.SPLIT_VERSION:
        raise ContractError(f"Split {split.get('split_version')} ≠ {contract.SPLIT_VERSION}")
    check_split(split)
    if annotations_path is not None and split.get("source"):
        sha = file_sha256(annotations_path)
        if sha != split["source"]["sha256"]:
            raise ContractError("El JSON de anotaciones no es el que generó el split")
    return split


def subset_map(split: dict) -> dict[str, str]:
    """``annotator_image`` -> nombre del subconjunto."""
    return {ai: name for name, ids in split["subsets"].items() for ai in ids}


def stems_of(split: dict, *names: str) -> set[str]:
    """Imágenes (stems) de uno o varios subconjuntos."""
    return {ai.split("-", 1)[1] for n in names for ai in split["subsets"][n]}
