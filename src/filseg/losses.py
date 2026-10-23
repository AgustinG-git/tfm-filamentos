"""Pérdidas disponibles por nombre. Todas reciben logits.

Cada experimento cambia una sola cosa respecto a la baseline:

- ``bce_dice``:        0,5·BCE + 0,5·Dice (baseline)
- ``bce_iou``:         0,5·BCE + 0,5·IoU
- ``bce_dice_cldice``: 0,5·BCE + 0,25·Dice + 0,25·clDice
- ``bce_cldice``:      0,5·BCE + 0,5·clDice
- ``bce_iou_cldice``:  0,5·BCE + 0,25·IoU + 0,25·clDice
- ``*_topo``:          cada una de las anteriores + α·Topograph

Topograph se activa tras ``calentamiento`` épocas (α = 0 antes): con
un modelo sin entrenar hay cientos de regiones críticas por imagen.

clDice: Shit et al., CVPR 2021, con su esqueleto blando iterativo.
Topograph: Lux et al., ICLR 2025, código oficial (extensión C++).
"""

import os
import sys

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEDice(nn.Module):
    """Baseline. Se deja exactamente como se entrenó."""

    def __init__(self, peso_bce=0.5, peso_dice=0.5):
        super().__init__()
        self.wb, self.wd = peso_bce, peso_dice
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = smp.losses.DiceLoss(mode="binary", from_logits=True)

    def forward(self, logits, target):
        return self.wb * self.bce(logits, target) + self.wd * self.dice(logits, target)


# ── clDice ───────────────────────────────────────────────────────────
def soft_erode(x):
    p1 = -F.max_pool2d(-x, (3, 1), 1, (1, 0))
    p2 = -F.max_pool2d(-x, (1, 3), 1, (0, 1))
    return torch.min(p1, p2)


def soft_dilate(x):
    return F.max_pool2d(x, 3, 1, 1)


def soft_open(x):
    return soft_dilate(soft_erode(x))


def soft_skel(x, iters: int):
    """Esqueleto blando: erosiones sucesivas, guardando lo que se pierde."""
    skel = F.relu(x - soft_open(x))
    for _ in range(iters):
        x = soft_erode(x)
        delta = F.relu(x - soft_open(x))
        skel = skel + F.relu(delta - skel * delta)
    return skel


def cldice_loss(prob, target, iters=10, smooth=1.0):
    """1 − clDice. ``prob`` y ``target`` en [0, 1], forma (B,1,H,W)."""
    sp, st = soft_skel(prob, iters), soft_skel(target, iters)
    tprec = ((sp * target).sum() + smooth) / (sp.sum() + smooth)  # esqueleto pred dentro del GT
    tsens = ((st * prob).sum() + smooth) / (st.sum() + smooth)    # esqueleto GT cubierto
    return 1.0 - 2.0 * tprec * tsens / (tprec + tsens)


# ── Topograph ────────────────────────────────────────────────────────
_TOPO = None


def topograph_loss(logits, target):
    """Término de Topograph para una salida binaria de un canal.

    Se importa al usarlo: el resto de pérdidas no necesita la
    extensión compilada. Ruta del repo en ``TOPOGRAPH_PATH``.
    """
    global _TOPO
    if _TOPO is None:
        sys.path.insert(0, os.environ.get("TOPOGRAPH_PATH", "/kaggle/working/Topograph"))
        from losses.topograph import TopographLoss
        _TOPO = TopographLoss(
            softmax=True, num_processes=1, include_background=False,
            use_c=True, eight_connectivity=True,  # 8-conex., como el post
        )
    x2 = torch.cat([torch.zeros_like(logits), logits], dim=1)  # [1-σ, σ]
    y2 = torch.cat([1 - target, target], dim=1)                # GT one-hot
    return _TOPO(x2, y2)


class Compuesta(nn.Module):
    """Suma ponderada de BCE, Dice, IoU y clDice, siempre en float32.

    En float16 (AMP) las sumas de clDice desbordan: un lote tiene
    ~2·10⁶ píxeles y el máximo de float16 es 65 504.
    """

    def __init__(self, peso_bce=0.0, peso_dice=0.0, peso_iou=0.0, peso_cldice=0.0,
                 peso_topo=0.0, iters=10, calentamiento=0):
        super().__init__()
        self.w = {"bce": peso_bce, "dice": peso_dice, "iou": peso_iou,
                  "cldice": peso_cldice, "topo": peso_topo}
        self.iters = iters
        self.calentamiento = calentamiento
        self.epoca = calentamiento + 1   # fuera de train: activa
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = smp.losses.DiceLoss(mode="binary", from_logits=True)
        self.iou = smp.losses.JaccardLoss(mode="binary", from_logits=True)

    def set_epoca(self, epoca: int) -> None:
        """La llama train.py al empezar cada época."""
        self.epoca = epoca

    def forward(self, logits, target):
        with torch.autocast(device_type="cuda", enabled=False):
            logits, target = logits.float(), target.float()
            terms = {
                "bce": lambda: self.bce(logits, target),
                "dice": lambda: self.dice(logits, target),
                "iou": lambda: self.iou(logits, target),
                "cldice": lambda: cldice_loss(torch.sigmoid(logits), target, self.iters),
                "topo": lambda: topograph_loss(logits, target),
            }
            w = dict(self.w)
            if self.epoca <= self.calentamiento:
                w["topo"] = 0.0                  # calentamiento
            return sum(v * terms[k]() for k, v in w.items() if v)


def _compuesta(**defaults):
    return lambda **args: Compuesta(**{**defaults, **args})


LOSSES = {
    "bce_dice": BCEDice,
    "bce_iou": _compuesta(peso_bce=0.5, peso_iou=0.5),
    "bce_dice_cldice": _compuesta(peso_bce=0.5, peso_dice=0.25, peso_cldice=0.25),
    "bce_cldice": _compuesta(peso_bce=0.5, peso_cldice=0.5),
    "bce_iou_cldice": _compuesta(peso_bce=0.5, peso_iou=0.25, peso_cldice=0.25),
    # Topograph: α y calentamiento fijados en configs/*_topograph.yaml
    "bce_dice_topo": _compuesta(peso_bce=0.5, peso_dice=0.5, peso_topo=0.04),
    "bce_iou_topo": _compuesta(peso_bce=0.5, peso_iou=0.5, peso_topo=0.04),
    "bce_dice_cldice_topo": _compuesta(peso_bce=0.5, peso_dice=0.25, peso_cldice=0.25,
                                       peso_topo=0.04),
    "bce_cldice_topo": _compuesta(peso_bce=0.5, peso_cldice=0.5, peso_topo=0.04),
    "bce_iou_cldice_topo": _compuesta(peso_bce=0.5, peso_iou=0.25, peso_cldice=0.25,
                                      peso_topo=0.04),
}


def build_loss(cfg: dict) -> nn.Module:
    e = cfg["entrenamiento"]
    if e["perdida"] not in LOSSES:
        raise KeyError(f"Pérdida desconocida: {e['perdida']}")
    return LOSSES[e["perdida"]](**e.get("perdida_args", {}))
