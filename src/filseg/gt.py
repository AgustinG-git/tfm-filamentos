"""GT de evaluación en el formato oficial, construido desde el JSON COCO.

En el JSON de MAGFiLO cada entrada de ``images`` es un par anotador-imagen:
su ``id`` es ``<anotador>-<imagen>`` y varias entradas comparten fichero.
Ese par es la unidad de evaluación de la competición.

El GT de evaluación es siempre el de este módulo, sea cual sea la
estrategia de etiquetas con la que se entrene (contrato v1, punto 3).
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils

from .contract import FULL_SIZE, ContractError


def load_coco(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)


def build_meta(coco: dict) -> pd.DataFrame:
    """Una fila por par anotador-imagen, en el orden del JSON."""
    n_by_image = Counter(str(a["image_id"]) for a in coco["annotations"])
    meta = (
        pd.DataFrame(coco["images"])
        .rename(columns={"id": "image_id"})
        .assign(
            image_id=lambda d: d["image_id"].astype(str),
            annotator_image=lambda d: d["image_id"],
            stem=lambda d: d["file_name"].map(lambda f: Path(f).stem),
            date=lambda d: d["stem"].str[:8],
            annotator=lambda d: d["image_id"].str.split("-", n=1).str[0],
            n_filaments=lambda d: d["image_id"].map(n_by_image).fillna(0).astype(int),
        )
    )
    validate_meta(meta)
    return meta


def validate_meta(meta: pd.DataFrame) -> None:
    """Comprueba que los ids se parsean igual que en el evaluador oficial.

    El evaluador oficial obtiene la imagen con ``split("-", 1)[1]`` sobre
    el id anotador-imagen y con ``split("_", 1)[0]`` sobre el
    ``filament_id``. Si algún id no encaja, el emparejamiento falla sin
    avisar, así que aquí se exige explícitamente.
    """
    ids = meta["annotator_image"]
    problems = {
        "id sin '-'": ~ids.str.contains("-", regex=False),
        "id ≠ <anotador>-<stem>": ids.str.split("-", n=1).str[1] != meta["stem"],
        "'_' en el id": ids.str.contains("_", regex=False),
        "id duplicado": ids.duplicated(keep=False),
    }
    sizes = meta[["height", "width"]].apply(tuple, axis=1)
    problems["tamaño ≠ 2048×2048"] = sizes != FULL_SIZE
    found = {k: ids[m].head(3).tolist() for k, m in problems.items() if m.any()}
    if found:
        raise ContractError(f"Ids o tamaños incompatibles con el evaluador oficial: {found}")


def ann_to_mask(ann: dict, h: int, w: int) -> np.ndarray:
    """Polígonos, RLE sin comprimir o RLE comprimido -> máscara (h, w)."""
    segm = ann["segmentation"]
    if isinstance(segm, list):  # polígonos
        polys = [p for p in segm if len(p) >= 6]
        if not polys:
            return np.zeros((h, w), np.uint8)
        rle = mask_utils.merge(mask_utils.frPyObjects(polys, h, w))
    elif isinstance(segm["counts"], list):  # RLE sin comprimir
        rle = mask_utils.frPyObjects(segm, h, w)
    else:
        rle = segm
    return mask_utils.decode(rle)


def mask_to_rle(mask: np.ndarray) -> str:
    """Máscara binaria -> RLE COCO comprimido (texto)."""
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    return rle["counts"].decode("utf-8")


def build_gt_df(coco: dict, annotator_images=None) -> pd.DataFrame:
    """GT en formato oficial: ``filament_id`` = ``<anotador>-<imagen>_<n>``.

    Args:
        coco: JSON de anotaciones cargado.
        annotator_images: si se da, solo esos pares (p. ej. val_B).

    Returns:
        DataFrame con ``filament_id`` y ``segmentation_rle`` (las dos
        columnas oficiales) más ``annotator_image`` y ``area`` para
        diagnóstico. Las máscaras vacías se conservan: el evaluador
        oficial las contaría como FN.
    """
    meta = build_meta(coco).set_index("annotator_image")
    keep = None if annotator_images is None else set(map(str, annotator_images))
    rows, counter = [], Counter()
    for ann in coco["annotations"]:
        ai = str(ann["image_id"])
        if keep is not None and ai not in keep:
            continue
        h, w = meta.loc[ai, ["height", "width"]]
        mask = ann_to_mask(ann, int(h), int(w))
        counter[ai] += 1
        rows.append({
            "filament_id": f"{ai}_{counter[ai]}",
            "segmentation_rle": mask_to_rle(mask),
            "annotator_image": ai,
            "area": int(mask.sum()),
        })
    return pd.DataFrame(rows, columns=["filament_id", "segmentation_rle", "annotator_image", "area"])
