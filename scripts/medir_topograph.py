"""Mide el coste y la escala del término de Topograph con un modelo entrenado.

Con un modelo sin entrenar hay cientos de regiones críticas por imagen y la
medida no es representativa. Aquí se usa el checkpoint de una ejecución.

Uso (con el entorno activado):
    python scripts/medir_topograph.py $FILSEG_OUT/runs/<run_id> --batches 20

Guarda medicion_topograph.csv (una fila por batch) en la carpeta de la
ejecución y muestra la media de cada término y los segundos por batch.
"""

import argparse
import os
import time
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from filseg import contract
from filseg.config import load_config
from filseg.data import LABELS, FilamentDataset, load_cache
from filseg.gt import build_meta, load_coco
from filseg.losses import LOSSES, topograph_loss
from filseg.model import build_model
from filseg.split import load_split

REPO = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("run_dir", help="carpeta de la ejecución con checkpoint.pth")
    p.add_argument("--batches", type=int, default=20)
    a = p.parse_args()
    run_dir, data = Path(a.run_dir), Path(os.environ["FILSEG_DATA"])

    # Datos: los primeros pares de train, como en el entrenamiento
    cfg = load_config(REPO / "configs/base.yaml")
    bs = cfg["entrenamiento"]["batch"]
    ann = next((data / "train").glob("*.json"))
    split = load_split(REPO / contract.SPLIT_PATH, ann)
    coco = load_coco(ann)
    tr = split["subsets"]["train"][: a.batches * bs]
    cache = load_cache(coco, build_meta(coco), data / "train" / "train_images", tr)
    samples = LABELS[cfg["pre"]["etiquetas"]](tr, cache)
    loader = DataLoader(
        FilamentDataset([cache.images[s] for s, _ in samples], [m for _, m in samples]),
        batch_size=bs,
    )

    # Modelo entrenado
    model = build_model(cfg, pretrained=False).cuda()
    ckpt = run_dir / contract.RUN_FILES["checkpoint"]
    model.load_state_dict(torch.load(ckpt, map_location="cuda"))
    model.eval()
    base = LOSSES["bce_dice_f32"]()

    filas = []
    for x, y in loader:
        x, y = x.cuda(), y.cuda()
        with torch.no_grad():
            logits = model(x).float()
        logits.requires_grad_(True)
        t0 = time.time()
        t = topograph_loss(logits, y)          # forward + backward
        if t.requires_grad:
            t.backward()
        filas.append({
            "bce_dice": base(logits.detach(), y).item(),
            "topo": t.item(),
            "segundos": time.time() - t0,
        })

    df = pd.DataFrame(filas)
    df.to_csv(run_dir / "medicion_topograph.csv", index=False)
    print(df.describe().loc[["mean", "50%", "max"]].round(3))
    r = df["topo"].mean() / df["bce_dice"].mean()
    print(f"topo / bce_dice = {r:.1f}  ->  con α = {1 / r:.4f} pesan lo mismo")


if __name__ == "__main__":
    main()
