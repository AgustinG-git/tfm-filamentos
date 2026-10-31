"""Barbas de los filamentos: definición, estadísticas y cobertura por ejecución.

Una barba es la parte del esqueleto de la máscara que se aleja de la espina
anotada (campo ``spine`` de MAGFiLO) más de lo que explica el grosor:

    barbas = esqueleto(máscara) − tubo(espina, r),   r = K_TUBO · w / 2

con ``w = área / longitud(espina)`` el grosor medio del filamento. Se
descartan trozos de menos de ``MIN_RAMA`` píxeles (ruido del esqueleto).
Todo se mide a 2048, por anotación de val_B.

Medidas (sumando píxeles de todas las anotaciones):
- cobertura: fracción del esqueleto de barbas real que cubre la predicción;
- precisión: fracción de las ramas predichas junto al filamento (esqueleto
  de la predicción en la zona de barbas) que caen sobre una barba real. Lo
  predicho en esa zona fuera de las barbas cuenta en contra;
- F1 de las dos, y ramas inventadas por filamento.
La zona de barbas es el entorno del filamento (``K_ZONA`` grosores) sin el
tubo de la espina ni los demás filamentos de la misma anotación.

Uso (con el entorno activado):
    python scripts/barbas.py gt     # estadísticas, techo a 512 y ejemplos
    python scripts/barbas.py runs   # cobertura de barbas y espina por ejecución
    python scripts/barbas.py runs --filtro "hp-.*-t00_"   # solo algunas

Salidas en $FILSEG_OUT/barbas/.
"""

import argparse
import json
import re
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
K_ZONA = 2.0    # alcance de la zona de barbas, en grosores
TOL = 3         # tolerancia (px a 2048) al medir


def datos():
    path = next((DATA / "train").glob("*.json"))
    coco = load_coco(path)
    B = set(load_split(REPO / contract.SPLIT_PATH, path)["subsets"]["val_B"])
    meta = build_meta(coco).set_index("annotator_image")
    return [a for a in coco["annotations"] if str(a["image_id"]) in B], meta


def disco(r):
    return np.ones((2 * r + 1,) * 2, np.uint8)


def ramas(sk):
    """Componentes del esqueleto con al menos MIN_RAMA píxeles."""
    n, lab, st, _ = cv2.connectedComponentsWithStats(sk.astype(np.uint8), connectivity=8)
    keep = [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= MIN_RAMA]
    return np.where(np.isin(lab, keep), lab, 0)


def mascaras_por_imagen(anns, meta):
    """Máscaras completas de cada anotación, agrupadas por par anotador-imagen."""
    out = {}
    for a in anns:
        r = meta.loc[str(a["image_id"])]
        out.setdefault(str(a["image_id"]), []).append(
            (a, ann_to_mask(a, int(r.height), int(r.width)).astype(bool)))
    return out


def partes(a, m, otras):
    """Espina, barbas y zona de barbas de una anotación, en su caja.

    ``m``: su máscara completa; ``otras``: unión de los demás filamentos
    de la misma anotación (se excluyen de la zona).
    """
    pts = np.asarray(a.get("spine") or [], float).reshape(-1, 2)
    if not m.any() or len(pts) < 2:
        return None
    h, w = m.shape
    largo = np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()
    grosor = m.sum() / max(largo, 1.0)
    margen = int(np.ceil(K_ZONA * grosor)) + 2 * TOL + 2
    ys, xs = np.nonzero(m)
    y0, y1 = max(ys.min() - margen, 0), min(ys.max() + margen + 1, h)
    x0, x1 = max(xs.min() - margen, 0), min(xs.max() + margen + 1, w)
    caja = (slice(y0, y1), slice(x0, x1))
    mc = m[caja]
    esp = np.zeros(mc.shape, np.uint8)
    cv2.polylines(esp, [np.round(pts - [x0, y0]).astype(np.int32)], False, 1, 1)
    dist = cv2.distanceTransform((1 - esp).astype(np.uint8), cv2.DIST_L2, 5)
    fuera_tubo = dist > K_TUBO * grosor / 2
    lab = ramas(skeletonize(mc) & fuera_tubo)
    zona = (
        (cv2.dilate(mc.astype(np.uint8), disco(int(np.ceil(K_ZONA * grosor)))) > 0)
        & fuera_tubo
        & ~(cv2.dilate(otras[caja].astype(np.uint8), disco(TOL)) > 0)
    )
    return {"caja": caja, "mascara": mc, "espina": esp.astype(bool), "barbas": lab > 0,
            "lab": lab, "zona": zona, "grosor": grosor}


def medir(p, pred):
    """Cobertura, precisión y ramas inventadas de ``pred`` en una anotación."""
    pc = pred[p["caja"]]
    pr = cv2.dilate(pc.astype(np.uint8), disco(TOL)) > 0
    ids = [i for i in np.unique(p["lab"]) if i]
    det = sum(pr[p["lab"] == i].mean() >= 0.5 for i in ids)
    cerca = cv2.dilate(p["barbas"].astype(np.uint8), disco(TOL)) > 0
    lab_p = ramas(skeletonize(pc) & p["zona"])          # ramas predichas
    ids_p = [i for i in np.unique(lab_p) if i]
    inventadas = sum(not cerca[lab_p == i].any() for i in ids_p)
    return {
        "barbas_px": int(p["barbas"].sum()), "barbas_cub": int((p["barbas"] & pr).sum()),
        "espina_px": int(p["espina"].sum()), "espina_cub": int((p["espina"] & pr).sum()),
        "n_barbas": len(ids), "barbas_det": int(det),
        "pred_px": int((lab_p > 0).sum()), "pred_ok": int(((lab_p > 0) & cerca).sum()),
        "inventadas": int(inventadas),
    }


def resumen(df):
    s = df.sum(numeric_only=True)
    cob = s.barbas_cub / max(s.barbas_px, 1)
    prec = s.pred_ok / max(s.pred_px, 1)
    return pd.Series({
        "cob_barbas": cob,
        "prec_barbas": prec,
        "f1_barbas": 2 * cob * prec / max(cob + prec, 1e-12),
        "barbas_det": s.barbas_det / max(s.n_barbas, 1),
        "inventadas_fil": s.inventadas / max(len(df), 1),
        "cob_espina": s.espina_cub / max(s.espina_px, 1),
    })


def recorrer(anns, meta):
    """(stem, h, w, partes) de cada anotación de val_B con espina."""
    for img, lista in mascaras_por_imagen(anns, meta).items():
        for k, (a, m) in enumerate(lista):
            otras = np.zeros_like(m)
            for j, (_, mj) in enumerate(lista):
                if j != k:
                    otras |= mj
            p = partes(a, m, otras)
            yield img.split("-", 1)[1], m, a, p


def modo_gt(anns, meta):
    import matplotlib.pyplot as plt
    filas, ejemplos = [], []
    for _, m, a, p in recorrer(anns, meta):
        if p is None:
            filas.append({"sin_espina": 1})
            continue
        h, w = m.shape
        small = resize_mask(m)                                # lo que ve el modelo
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


def modo_runs(anns, meta, filtro=None):
    cfg = {d.name: json.loads((d / "run_config.json").read_text())
           for d in (OUT / "runs").iterdir()
           if (d / "run_config.json").exists() and "-humo" not in d.name}
    if filtro:  # por run_id o por su fuente
        cfg = {r: c for r, c in cfg.items()
               if re.search(filtro, r) or re.search(filtro, c.get("fuente", ""))}
    nombre = lambda r: r.split("_", 1)[1].rsplit("_", 1)[0]
    runs = {r: (nombre(r), "sin post") for r, c in cfg.items() if "fuente" not in c}
    runs |= {r: (nombre(c["fuente"]), "B2") for r, c in cfg.items()
             if c.get("config_post", {}).get("nombre") == "b2-optuna"}

    gt = [(stem, *m.shape, p) for stem, m, _, p in recorrer(anns, meta) if p is not None]

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
    ap.add_argument("--filtro", help="regex sobre run_id (modo runs)")
    a = ap.parse_args()
    if a.modo == "gt":
        modo_gt(*datos())
    else:
        modo_runs(*datos(), filtro=a.filtro)
