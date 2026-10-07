"""Pérdidas disponibles por nombre. Todas reciben logits."""

import segmentation_models_pytorch as smp
import torch.nn as nn


class BCEDice(nn.Module):
    def __init__(self, peso_bce=0.5, peso_dice=0.5):
        super().__init__()
        self.wb, self.wd = peso_bce, peso_dice
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = smp.losses.DiceLoss(mode="binary", from_logits=True)

    def forward(self, logits, target):
        return self.wb * self.bce(logits, target) + self.wd * self.dice(logits, target)


LOSSES = {"bce_dice": BCEDice}


def build_loss(cfg: dict) -> nn.Module:
    e = cfg["entrenamiento"]
    if e["perdida"] not in LOSSES:
        raise KeyError(f"Pérdida desconocida: {e['perdida']}")
    return LOSSES[e["perdida"]](**e.get("perdida_args", {}))
