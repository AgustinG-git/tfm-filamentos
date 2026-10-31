"""Figura de ejemplo de MAGFiLO para el capítulo de contexto.

Tres paneles: (a) imagen H-alfa completa; (b) las anotaciones de cada
anotador, una por color; (c) detalle de un filamento con su máscara y su
espina. Usa una imagen de entrenamiento con tres anotadores.

Uso:
    python scripts/figura_magfilo.py            # elige la imagen sola
    python scripts/figura_magfilo.py --stem X   # una imagen concreta
Salida: memoria/figuras/magfilo_ejemplo.png
"""

import argparse
import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from filseg import contract
from filseg.gt import ann_to_mask, build_meta, load_coco
from filseg.split import load_split

REPO = Path(__file__).resolve().parents[1]
DATA = Path(os.environ["FILSEG_DATA"])
COLORES = ["#e41a1c", "#377eb8", "#4daf4a"]   # un color por anotador


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem")
    a = ap.parse_args()

    path = next((DATA / "train").glob("*.json"))
    coco = load_coco(path)
    train = set(load_split(REPO / contract.SPLIT_PATH, path)["subsets"]["train"])
    meta = build_meta(coco)
    meta = meta[meta.annotator_image.isin(train)]
    anns = {}
    for x in coco["annotations"]:
        anns.setdefault(str(x["image_id"]), []).append(x)
    meta = meta.assign(n=meta.annotator_image.map(lambda i: len(anns.get(i, []))))

    if a.stem:
        stem = a.stem
    else:  # 3 anotadores y más filamentos
        por = meta.groupby("stem").agg(k=("annotator", "nunique"), n=("n", "sum"))
        stem = por[por.k == 3].n.idxmax()
    filas = meta[meta.stem == stem].sort_values("annotator")
    r0 = filas.iloc[0]
    h, w = int(r0.height), int(r0.width)
    img = cv2.imread(str(DATA / "train" / "train_images" / r0.file_name), cv2.IMREAD_GRAYSCALE)

    fig, ax = plt.subplots(1, 3, figsize=(15, 5.3))
    ax[0].imshow(img, cmap="gray")
    ax[0].set_title("(a) Observación H-alfa")
    ax[1].imshow(img, cmap="gray")
    grande = None
    for k, (_, r) in enumerate(filas.iterrows()):
        for x in anns.get(r.annotator_image, []):
            m = ann_to_mask(x, h, w).astype(np.uint8)
            cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            for c in cs:
                ax[1].plot(c[:, 0, 0], c[:, 0, 1], color=COLORES[k], lw=0.8)
            if k == 0 and (grande is None or m.sum() > grande[1].sum()):
                grande = (x, m)
        ax[1].plot([], [], color=COLORES[k], label=f"Anotador {k + 1}")
    ax[1].legend(loc="lower right", fontsize=8)
    ax[1].set_title("(b) Anotaciones de los tres anotadores")

    x, m = grande  # filamento más grande del anotador 1
    ys, xs = np.nonzero(m)
    mg = 60
    y0, y1 = max(ys.min() - mg, 0), min(ys.max() + mg, h)
    x0, x1 = max(xs.min() - mg, 0), min(xs.max() + mg, w)
    ax[2].imshow(img[y0:y1, x0:x1], cmap="gray", extent=(x0, x1, y1, y0))
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for c in cs:
        ax[2].plot(c[:, 0, 0], c[:, 0, 1], color=COLORES[0], lw=1.2, label="Máscara")
    esp = np.asarray(x.get("spine") or [], float).reshape(-1, 2)
    if len(esp):
        ax[2].plot(esp[:, 0], esp[:, 1], color="yellow", lw=1.5, label="Espina")
    h_, l_ = ax[2].get_legend_handles_labels()
    ax[2].legend(dict(zip(l_, h_)).values(), dict(zip(l_, h_)).keys(),
                 loc="lower right", fontsize=8)
    ax[2].set_title("(c) Detalle: máscara y espina")
    for e in ax:
        e.set_xticks([]), e.set_yticks([])

    dest = REPO / "memoria" / "figuras" / "magfilo_ejemplo.png"
    fig.tight_layout()
    fig.savefig(dest, dpi=200, bbox_inches="tight")
    print(f"✔ {stem} -> {dest}")


if __name__ == "__main__":
    main()
