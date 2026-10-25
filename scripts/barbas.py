"""Barbas de los filamentos: definición, estadísticas y cobertura por ejecución.

Una barba es la parte del esqueleto de la máscara que se aleja de la espina
anotada (campo ``spine`` de MAGFiLO) más de lo que explica el grosor:

    barbas = esqueleto(máscara) − tubo(espina, r),   r = K_TUBO · w / 2

con ``w = área / longitud(espina)`` el grosor medio del filamento. Se
descartan trozos de menos de ``MIN_RAMA`` píxeles (ruido del esqueleto).
Todo se mide a 2048, por anotación de val_B.

Uso (con el entorno activado):
    python scripts/barbas.py gt     # estadísticas, techo a 512 y ejemplos
    python scripts/barbas.py runs   # cobertura de barbas y espina por ejecución

Salidas en $FILSEG_OUT/barbas/.
"""

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from skimage.morphology import skeletonize

from filseg import contract
from filseg.data import resize_mask
from filseg.gt import ann_to_mask, build_meta, load_coco
from filseg.official_metric import rles_to_layers
from filseg.split import load_split

REPO = Path(__file__).resolve().parents[1]
OUT, DATA = Path(os.environ["FILSEG_OUT"]), Path(os.environ["FILSEG_DATA"])
DEST = OUT / "barbas"
K_TUBO = 1.5    # radio del tubo: 1,5 × semigrosor
MIN_RAMA = 6    # px de esqueleto mínimos por barba
TOL = 3         # tolerancia (px a 2048) al medir
MARGEN = 20     # margen del recorte (px)


def datos():
    path = next((DATA / "train").glob("*.json"))
    coco = load_coco(path)
    B = set(load_split(REPO / contract.SPLIT_PATH, path)["subsets"]["val_B"])
    meta = build_meta(coco).set_index("annotator_image")
    return [a for a in coco["annotations"] if str(a["image_id"]) in B], meta


def partes(a, h, w):
    """Máscara, espina y barbas de una anotación, recortadas a su caja."""
    m = ann_to_mask(a, h, w).astype(bool)
    pts = np.asarray(a.get("spine") or [], float).reshape(-1, 2)
    if not m.any() or len(pts) < 2:
        return None
    ys, xs = np.nonzero(m)
    y0, y1 = max(ys.min() - MARGEN, 0), min(ys.max() + MARGEN + 1, h)
    x0, x1 = max(xs.min() - MARGEN, 0), min(xs.max() + MARGEN + 1, w)
    caja = (slice(y0, y1), slice(x0, x1))
    mc = m[caja]
    esp = np.zeros(mc.shape, np.uint8)
    cv2.polylines(esp, [np.round(pts - [x0, y0]).astype(np.int32)], False, 1, 1)
    largo = np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()
    grosor = mc.sum() / max(largo, 1.0)
    dist = cv2.distanceTransform((1 - esp).astype(np.uint8), cv2.DIST_L2, 5)
    sk = skeletonize(mc) & (dist > K_TUBO * grosor / 2)
    n, lab, st, _ = cv2.connectedComponentsWithStats(sk.astype(np.uint8), connectivity=8)
    keep = [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= MIN_RAMA]
    return {"caja": caja, "completa": m, "mascara": mc, "espina": esp.astype(bool),
            "barbas": np.isin(lab, keep), "lab": np.where(np.isin(lab, keep), lab, 0),
            "grosor": grosor}


def medir(p, pred):
    """Píxeles de barba/espina cubiertos por ``pred`` y barbas detectadas."""
    pr = cv2.dilate(pred[p["caja"]].astype(np.uint8), np.ones((2 * TOL + 1,) * 2, np.uint8)) > 0
    ids = [i for i in np.unique(p["lab"]) if i]
    det = sum((pr[p["lab"] == i]).mean() >= 0.5 for i in ids)
    return {
        "barbas_px": int(p["barbas"].sum()), "barbas_cub": int((p["barbas"] & pr).sum()),
        "espina_px": int(p["espina"].sum()), "espina_cub": int((p["espina"] & pr).sum()),
        "n_barbas": len(ids), "barbas_det": int(det),
    }


def resumen(df):
    s = df.sum(numeric_only=True)
    return pd.Series({
        "cob_barbas": s.barbas_cub / max(s.barbas_px, 1),
        "cob_espina": s.espina_cub / max(s.espina_px, 1),
        "barbas_det": s.barbas_det / max(s.n_barbas, 1),
    })


def modo_gt(anns, meta):
    import matplotlib.pyplot as plt
    filas, ejemplos = [], []
    for a in anns:
        r = meta.loc[str(a["image_id"])]
        h, w = int(r.height), int(r.width)
        p = partes(a, h, w)
        if p is None:
            filas.append({"sin_espina": 1})
            continue
        small = resize_mask(p["completa"])                    # lo que ve el modelo
        up = cv2.resize(small.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR) >= 0.5
        filas.append({**medir(p, up), "grosor": p["grosor"], "sin_espina": 0})
        if p["barbas"].any():
            ejemplos.append((p["barbas"].sum(), a["id"], p))
    df = pd.DataFrame(filas).fillna(0)
    con = df[df.sin_espina == 0]
    print(f"anotaciones val_B: {len(df)} | sin espina o vacías: {int(df.sin_espina.sum())}")
    print(f"con al menos una barba: {(con.n_barbas > 0).mean():.1%} | barbas por filamento: {con.n_barbas.mean():.2f}")
    print(f"esqueleto de barbas / espina: {con.barbas_px.sum() / con.espina_px.sum():.2f} | grosor mediano: {con.grosor.median():.1f} px")
    print("techo a 512 (GT reducido y vuelto a 2048):")
    print(resumen(con).round(3).to_string())

    DEST.mkdir(parents=True, exist_ok=True)
    ejemplos.sort(key=lambda e: -e[0])
    rng = np.random.default_rng(0)
    elegidos = ejemplos[:4] + [ejemplos[i] for i in rng.choice(len(ejemplos), 4, replace=False)]
    for k, (_, aid, p) in enumerate(elegidos):
        rgb = np.dstack([p["mascara"] * 0.35] * 3)
        rgb[p["espina"]] = (0.2, 0.5, 1.0)                     # espina: azul
        rgb[cv2.dilate(p["barbas"].astype(np.uint8), np.ones((3, 3))) > 0] = (1.0, 0.2, 0.2)
        plt.imsave(DEST / f"ejemplo_{k:02d}_ann{aid}.png", rgb)
    print(f"ejemplos: {DEST}/ejemplo_*.png (gris máscara, azul espina, rojo barbas)")


def modo_runs(anns, meta):
    cfg = {d.name: json.loads((d / "run_config.json").read_text())
           for d in (OUT / "runs").iterdir()
           if (d / "run_config.json").exists() and "-humo" not in d.name}
    nombre = lambda r: r.split("_", 1)[1].rsplit("_", 1)[0]
    runs = {r: (nombre(r), "sin post") for r, c in cfg.items() if "fuente" not in c}
    runs |= {r: (nombre(c["fuente"]), "B2") for r, c in cfg.items()
             if c.get("config_post", {}).get("nombre") == "b2-optuna"}

    gt = []
    for a in anns:
        r = meta.loc[str(a["image_id"])]
        h, w = int(r.height), int(r.width)
        p = partes(a, h, w)
        if p is not None:
            p.pop("completa")
            gt.append((str(a["image_id"]).split("-", 1)[1], h, w, p))

    filas = []
    for rid, (ent, post) in sorted(runs.items()):
        preds = pd.read_csv(OUT / "runs" / rid / contract.RUN_FILES["val_predictions"])
        por_stem = preds.groupby(preds.filament_id.str.rsplit("_", n=1).str[0]).segmentation_rle.apply(list)
        cache, medidas = {}, []
        for stem, h, w, p in gt:
            if stem not in cache:
                rles = por_stem.get(stem, [])
                cache[stem] = rles_to_layers(rles, h, w).max(0) > 0 if rles else np.zeros((h, w), bool)
            medidas.append(medir(p, cache[stem]))
        filas.append({"entrenamiento": ent, "post": post, "run_id": rid,
                      **resumen(pd.DataFrame(medidas)).to_dict()})
        print(f"✔ {ent} ({post})", flush=True)

    DEST.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(filas).sort_values(["post", "entrenamiento"])
    df.to_csv(DEST / "barbas_runs.csv", index=False)
    pd.set_option("display.width", 200)
    print(df.drop(columns="run_id").round(3).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("modo", choices=["gt", "runs"])
    modo = ap.parse_args().modo
    {"gt": modo_gt, "runs": modo_runs}[modo](*datos())
