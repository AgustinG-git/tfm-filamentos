"""Modelos disponibles por nombre."""

import segmentation_models_pytorch as smp

ARCHS = {"unet": smp.Unet}


def build_model(cfg: dict, pretrained: bool = True):
    m = cfg["modelo"]
    if m["arquitectura"] not in ARCHS:
        raise KeyError(f"Arquitectura desconocida: {m['arquitectura']}")
    return ARCHS[m["arquitectura"]](
        encoder_name=m["encoder"],
        encoder_weights=m["pesos"] if pretrained else None,
        in_channels=3,
        classes=1,
    )
