"""Evaluador estándar del contrato v1.

Envuelve la métrica oficial (``official_metric``) sin modificarla:

- comprueba ids y solapes antes de evaluar;
- calcula PQ, SQ, RQ y recuentos con el mismo bucle que ``get_pq_score``
  y verifica que el PQ coincide con el oficial;
- genera los tres gráficos de la rúbrica con las funciones oficiales.

Todas las cifras del TFM salen de ``evaluate``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils

from .contract import FULL_SIZE, IOU_THR, ContractError
from .official_metric import (
    get_overlap_df,
    get_pq_score,
    plot_distribution,
    plot_m2n_counts,
)

GT_ID = r"[^_\-]+-[^_]+_\d+"   # <anotador>-<imagen>_<n>
PRED_ID = r"[^_]+_\d+"          # <imagen>_<n>


def image_of(filament_ids: pd.Series) -> pd.Series:
    """Parte previa al primer '_' (como el oficial)."""
    return filament_ids.str.split("_", n=1).str[0]


def check_ids(df: pd.DataFrame, pattern: str, name: str) -> None:
    ids = df["filament_id"].astype(str)
    bad = ids[~ids.str.fullmatch(pattern)]
    dup = ids[ids.duplicated()]
    if len(bad) or len(dup):
        raise ContractError(
            f"{name}: {len(bad)} ids con formato inválido {bad.head(3).tolist()}, "
            f"{len(dup)} duplicados {dup.head(3).tolist()}"
        )


def find_overlaps(pred_df: pd.DataFrame, size=FULL_SIZE) -> list[str]:
    """Imágenes con predicciones que se solapan entre sí.

    Sin decodificar: hay solape si el área de la unión es menor que la
    suma de áreas. El evaluador oficial contaría cada copia como un TP.
    """
    out = []
    for img, grp in pred_df.groupby(image_of(pred_df["filament_id"])):
        if len(grp) < 2:
            continue
        rles = [{"size": list(size), "counts": c.encode()} for c in grp["segmentation_rle"]]
        union = mask_utils.area(mask_utils.merge(rles, intersect=False))
        if union < mask_utils.area(rles).sum():
            out.append(img)
    return out


def pq_components(overlap_df: pd.DataFrame) -> dict:
    """Mismo bucle que ``get_pq_score``, devolviendo también los recuentos."""
    sum_iou, tp, fp, fn = 0.0, 0, 0, 0
    for row in overlap_df.itertuples(index=False):
        if row.n_gt == 0:
            fp += row.n_pred
            continue
        if row.n_pred == 0:
            fn += row.n_gt
            continue
        iou = np.asarray(row.iou_matrix, dtype=float)
        hit = iou > IOU_THR
        sum_iou += float(iou[hit].sum())
        tp += int(hit.sum())
        fp += int((hit.sum(axis=0) == 0).sum())
        fn += int((hit.sum(axis=1) == 0).sum())
    den = tp + 0.5 * fp + 0.5 * fn
    return {
        "pq": sum_iou / den if den > 0 else 0.0,
        "sq": sum_iou / tp if tp > 0 else 0.0,
        "rq": tp / den if den > 0 else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "sum_iou": sum_iou,
    }


@dataclass
class EvalResult:
    metrics: dict
    overlap_df: pd.DataFrame


def evaluate(
    gt_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    *,
    check_overlaps: bool = True,
) -> EvalResult:
    """Evaluación oficial + comprobaciones del contrato.

    Args:
        gt_df: GT en formato oficial (``gt.build_gt_df``), ya filtrado al
            subconjunto que se quiere medir.
        pred_df: predicciones por imagen (``filament_id`` = ``<imagen>_<n>``).
        check_overlaps: rechazar predicciones solapadas (no desactivar
            salvo para demostrar el agujero de la métrica).

    Returns:
        ``EvalResult`` con métricas y la tabla de solapes oficial, que es
        la entrada de los gráficos de la rúbrica.
    """
    pred_df = pred_df.astype({"filament_id": str, "segmentation_rle": str})
    check_ids(gt_df, GT_ID, "GT")
    check_ids(pred_df, PRED_ID, "Predicciones")
    if check_overlaps:
        overlapping = find_overlaps(pred_df)
        if overlapping:
            raise ContractError(
                f"{len(overlapping)} imágenes con predicciones solapadas: {overlapping[:3]}"
            )

    overlap_df = get_overlap_df(gt_df, pred_df)
    metrics = pq_components(overlap_df)
    official = get_pq_score(overlap_df)
    if not np.isclose(metrics["pq"], official, rtol=0, atol=1e-9):
        raise AssertionError(f"PQ propio {metrics['pq']} ≠ oficial {official}")

    gt_images = set(overlap_df["annotator_image"].str.split("-", n=1).str[1])
    pred_images = image_of(pred_df["filament_id"])
    metrics.update({
        "n_annotator_images": len(overlap_df),
        "n_images": len(gt_images),
        "n_gt": int(overlap_df["n_gt"].sum()),
        "n_pred_ignored": int((~pred_images.isin(gt_images)).sum()),
    })
    return EvalResult(metrics, overlap_df)


def filter_gt(gt_df: pd.DataFrame, annotator_images) -> pd.DataFrame:
    """GT restringido a un subconjunto de pares anotador-imagen."""
    keep = set(map(str, annotator_images))
    ai = image_of(gt_df["filament_id"])
    return gt_df[ai.isin(keep)].reset_index(drop=True)


def plot_official(overlap_df: pd.DataFrame, out_dir: str | Path | None = None, tag: str = ""):
    """Los tres gráficos de la rúbrica, con las funciones oficiales."""
    import matplotlib.pyplot as plt

    specs = [
        ("iou", lambda: plot_distribution(overlap_df["iou_matrix"], f"IoU Distribution {tag}", "IoU")),
        ("dice", lambda: plot_distribution(overlap_df["dice_matrix"], f"Dice Distribution {tag}", "Dice")),
        ("m2n", lambda: plot_m2n_counts(overlap_df, f"Many to One and One to Many Distribution {tag}")),
    ]
    figs = {}
    for name, draw in specs:
        fig = plt.figure(figsize=(7, 4))
        draw()
        fig.tight_layout()
        if out_dir is not None:
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            fig.savefig(Path(out_dir) / f"{name}{'_' + tag if tag else ''}.png", dpi=120)
        figs[name] = fig
    return figs
