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


def persistencia_512(prob, umbral=0.5, tau=0.0, delta=0.0):
    """E2 a 512: filtro por persistencia y fusión por collado.

    Trabaja sobre la filtración de superniveles de ``prob``. Una
    componente de ``X_t = {prob > umbral}`` se conserva si contiene un
    máximo con persistencia ``b - d >= tau`` (los h-máximos de altura
    ``tau`` son esos puntos del diagrama H0). Después, las componentes
    conservadas que están unidas en ``X_{t-delta}`` forman una sola
    instancia. Ambas con 8-conectividad. Devuelve ``(labels, n)``.
    """
    from skimage.morphology import h_maxima      # solo si se usa E2
    xt = (prob > umbral).astype(np.uint8)
    n_t, lt = cv2.connectedComponents(xt, connectivity=8)
    if n_t <= 1:
        return np.zeros(prob.shape, np.int32), 0
    if tau > 0:
        hmax = h_maxima(prob.astype(np.float64), tau, footprint=np.ones((3, 3))) > 0
        keep = np.unique(lt[hmax & (xt > 0)])
        keep = keep[keep > 0]
    else:
        keep = np.arange(1, n_t)
    vivo = np.isin(lt, keep)
    if delta > 0:                                # fusión por collado
        _, ld = cv2.connectedComponents(
            (prob > umbral - delta).astype(np.uint8), connectivity=8)
        lab = np.where(vivo, ld, 0)
    else:
        lab = np.where(vivo, lt, 0)
    ids = np.unique(lab)
    lut = np.zeros(lab.max() + 1, np.int32)
    lut[ids[ids > 0]] = np.arange(1, (ids > 0).sum() + 1)
    return lut[lab], int((ids > 0).sum())


def persistencia_full(prob512, prob_full, umbral=0.5, tau=0.0, delta=0.0):
    """E2 a resolución completa: las decisiones se toman a 512.

    Cada componente de ``{prob_full > umbral}`` toma la instancia de
    512 que más píxeles le aporta (etiquetas de 512 ampliadas por
    vecino más próximo); si no le corresponde ninguna, se descarta.
    """
    lab512, n = persistencia_512(prob512, umbral, tau, delta)
    if n == 0:
        return np.zeros(prob_full.shape, np.int32), 0
    k = prob_full.shape[0] // prob512.shape[0]
    up = lab512.repeat(k, 0).repeat(k, 1)
    n_c, comp = cv2.connectedComponents((prob_full > umbral).astype(np.uint8), connectivity=8)
    sel = (comp > 0) & (up > 0)
    c, l = comp[sel].astype(np.int64), up[sel].astype(np.int64)
    key, cnt = np.unique(c * (n + 1) + l, return_counts=True)
    kc, kl = key // (n + 1), key % (n + 1)
    orden = np.lexsort((-cnt, kc))               # por comp., más votos primero
    kc, kl = kc[orden], kl[orden]
    primero = np.r_[True, kc[1:] != kc[:-1]]
    mapa = np.zeros(n_c, np.int32)
    mapa[kc[primero]] = kl[primero]
    out = mapa[comp]
    ids = np.unique(out)
    lut = np.zeros(n + 1, np.int32)
    lut[ids[ids > 0]] = np.arange(1, (ids > 0).sum() + 1)
    return lut[out], int((ids > 0).sum())


POSTPROC = {"umbral_cc": umbral_cc, "persistencia": persistencia_512}


def apply(prob512, post: dict, full: bool = True):
    """Aplica el post-proceso de la config, a 2048 o a 512.

    A 2048 se reescalan las probabilidades (bilineal) y el área
    mínima, que en la config está en píxeles a 512.
    """
    if post["metodo"] not in POSTPROC:
        raise KeyError(f"Post-proceso desconocido: {post['metodo']}")
    if post["metodo"] == "persistencia":
        args = {k: post.get(k, 0.0) for k in ("umbral", "tau", "delta")}
        p512 = prob512.astype(np.float32)
        if not full:
            return persistencia_512(p512, **args)
        h, w = FULL_SIZE
        pf = cv2.resize(p512, (w, h), interpolation=cv2.INTER_LINEAR)
        return persistencia_full(p512, pf, **args)
    fn = POSTPROC[post["metodo"]]
    prob = prob512.astype(np.float32)
    area = post.get("area_min_512", 0)
    if full:
        h, w = FULL_SIZE
        prob = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)
        area = area * (h * w) / TRAIN_SIZE**2
    return fn(prob, umbral=post.get("umbral", 0.5), area_min=area,
              conectividad=post.get("conectividad", 8))
