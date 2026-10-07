"""Post-proceso: probabilidades -> instancias (mapa de etiquetas)."""

from __future__ import annotations

import cv2
import numpy as np

from .contract import FULL_SIZE, TRAIN_SIZE


def umbral_cc(prob, umbral=0.5, area_min=0, conectividad=8):
    """Umbral + componentes conexas + filtro de área.

    Devuelve ``(labels, n)``: 0 = fondo, 1..n = instancias.
    """
    binary = (prob > umbral).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=conectividad)
    keep = np.where(stats[1:, cv2.CC_STAT_AREA] >= area_min)[0] + 1
    lut = np.zeros(n, np.int32)
    lut[keep] = np.arange(1, len(keep) + 1)
    return lut[labels], len(keep)


POSTPROC = {"umbral_cc": umbral_cc}


def apply(prob512, post: dict, full: bool = True):
    """Aplica el post-proceso de la config, a 2048 o a 512.

    A 2048 se reescalan las probabilidades (bilineal) y el área
    mínima, que en la config está en píxeles a 512.
    """
    if post["metodo"] not in POSTPROC:
        raise KeyError(f"Post-proceso desconocido: {post['metodo']}")
    fn = POSTPROC[post["metodo"]]
    prob = prob512.astype(np.float32)
    area = post.get("area_min_512", 0)
    if full:
        h, w = FULL_SIZE
        prob = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)
        area = area * (h * w) / TRAIN_SIZE**2
    return fn(prob, umbral=post.get("umbral", 0.5), area_min=area,
              conectividad=post.get("conectividad", 8))
