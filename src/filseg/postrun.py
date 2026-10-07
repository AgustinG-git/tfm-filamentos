"""Nuevo post-proceso sobre una ejecución ya entrenada (sin reentrenar).

Parte de los logits guardados de una ejecución (la "fuente"):

1. Busca en val_A el mejor umbral y área mínima de la rejilla de la
   config (``post.ajuste``), a 2048 y con las reglas oficiales.
2. Aplica los parámetros elegidos a toda la validación y evalúa con la
   métrica oficial en val_A y val_B.
3. Guarda una ejecución nueva con las salidas del contrato, su
   submission y su fila del registro.

Uso en Kaggle::

    from filseg.postrun import run_post, find_run
    src = find_run("20261007-1639_base_df8c991")
    run_post("configs/b1_umbral.yaml", source=src, data_dir=DATA, repo_dir="tfm-filamentos")
"""

from __future__ import annotations

import itertools
import os
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.special import expit
from tqdm.auto import tqdm

from . import contract, postproc
from .config import load_config
from .contract import FULL_SIZE, RUN_FILES
from .gt import ann_to_mask, build_gt_df, load_coco
from .metrics import counts_from_labels, evaluate, plot_official, pq_from_counts
from .runs import (
    append_registry, assert_valid_run, git_commit, labels_to_rles,
    make_run_config, make_run_id, predictions_frame, read_run, write_run,
)
from .split import load_split


def sigmoid(x: np.ndarray) -> np.ndarray:
    return expit(x.astype(np.float32))


def find_run(run_id: str, root: str | Path = "/kaggle/input") -> Path:
    """Carpeta de una ejecución añadida como input en Kaggle."""
    for d, dirs, _ in os.walk(root, followlinks=True):
        if run_id in dirs:
            return Path(d) / run_id
    raise FileNotFoundError(f"No encuentro {run_id} en {root}: ¿lo has añadido con Add Input?")


def gt_idx_full(coco: dict, pairs) -> dict:
    """Filamentos GT a 2048 como índices planos, por par."""
    keep = set(pairs)
    out = defaultdict(list)
    h, w = FULL_SIZE
    for a in coco["annotations"]:
        ai = str(a["image_id"])
        if ai in keep:
            idx = np.flatnonzero(ann_to_mask(a, h, w))
            if idx.size:
                out[ai].append(idx)
    return {ai: out[ai] for ai in pairs}


def tune(probs_full: dict, pairs, gt_idx: dict, umbrales, areas_512) -> pd.DataFrame:
    """PQ en ``pairs`` para cada combinación (umbral, área a 512).

    ``probs_full``: stem -> probabilidad ya reescalada a 2048.
    Usa ``postproc.umbral_cc``, el mismo código que la predicción final.
    """
    scale = FULL_SIZE[0] * FULL_SIZE[1] / contract.TRAIN_SIZE**2
    rows = []
    for u, a in tqdm(list(itertools.product(umbrales, areas_512)), desc="Rejilla"):
        labels = {s: postproc.umbral_cc(p, umbral=u, area_min=a * scale) for s, p in probs_full.items()}
        counts = [counts_from_labels(gt_idx[ai], *labels[ai.split("-", 1)[1]]) for ai in pairs]
        rows.append({
            "umbral": u, "area_min_512": a, "pq": pq_from_counts(counts),
            **{k: sum(c[k] for c in counts) for k in ("tp", "fp", "fn")},
        })
    return pd.DataFrame(rows).sort_values("pq", ascending=False, kind="stable").reset_index(drop=True)


def run_post(
    config: str | Path,
    *,
    source: str | Path,
    data_dir: str | Path,
    repo_dir: str | Path,
    out_dir: str | Path = "/kaggle/working",
    nota: str = "",
) -> dict:
    repo, data, out_dir, source = Path(repo_dir), Path(data_dir), Path(out_dir), Path(source)
    cfg_path = Path(config) if Path(config).is_absolute() else repo / config
    cfg = load_config(cfg_path)
    ajuste = cfg["post"].get("ajuste")
    if not ajuste or cfg["post"]["metodo"] != "umbral_cc":
        raise KeyError("La config necesita post.metodo = umbral_cc y post.ajuste")

    src = read_run(source)
    stems, logits = src.logits()
    src_cfg = src.config["config"]
    print(f"▶ fuente: {src.run_id}")

    ann = next((data / "train").glob("*.json"))
    split_path = repo / contract.SPLIT_PATH
    split = load_split(split_path, ann)
    coco = load_coco(ann)
    A, B = split["subsets"]["val_A"], split["subsets"]["val_B"]

    # ── Ajuste en val_A ──────────────────────────────────────────────
    stems_A = sorted({ai.split("-", 1)[1] for ai in A})
    pos = {s: i for i, s in enumerate(stems)}
    h, w = FULL_SIZE
    probs_A = {
        s: cv2.resize(sigmoid(logits[pos[s]]), (w, h), interpolation=cv2.INTER_LINEAR)
        for s in tqdm(stems_A, desc="Reescalar val_A")
    }
    grid = tune(probs_A, A, gt_idx_full(coco, A), ajuste["umbrales"], ajuste["areas_min_512"])
    best = grid.iloc[0]
    del probs_A
    post = {
        "metodo": "umbral_cc",
        "umbral": float(best["umbral"]),
        "area_min_512": int(best["area_min_512"]),
        "conectividad": cfg["post"].get("conectividad", 8),
    }
    print(f"Elegido en val_A: umbral {post['umbral']} | área {post['area_min_512']} | PQ {best['pq']:.4f}")

    # ── Predicciones y evaluación oficial ────────────────────────────
    preds = {
        s: labels_to_rles(*postproc.apply(sigmoid(l), post))
        for s, l in tqdm(zip(stems, logits), total=len(stems), desc="Val 2048")
    }
    pred_df = predictions_frame(preds)
    res_A = evaluate(build_gt_df(coco, A), pred_df)
    res_B = evaluate(build_gt_df(coco, B), pred_df)
    if not np.isclose(res_A.metrics["pq"], best["pq"], atol=1e-6):
        raise AssertionError(f"PQ de la rejilla {best['pq']} ≠ oficial {res_A.metrics['pq']}")

    commit = git_commit(repo)
    run_id = make_run_id(cfg["nombre"], commit)
    run_dir = out_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    grid.to_csv(run_dir / "rejilla_val_A.csv", index=False)
    plot_official(res_B.overlap_df, run_dir / "figuras", tag="val_B")
    shutil.copy(source / RUN_FILES["checkpoint"], run_dir / RUN_FILES["checkpoint"])

    run_config = make_run_config(
        run_id=run_id, config={**src_cfg, "post": post}, split_path=split_path,
        git_commit=commit, selection=src.config["selection"], postproc=post,
        metrics_val_A=res_A.metrics, metrics_val_B=res_B.metrics,
        fuente=src.run_id, config_post=cfg, nota=nota,
    )
    write_run(
        run_dir, run_config=run_config, val_predictions=pred_df,
        history=src.history, val_logits=(stems, logits),
    )

    # ── Submission con el modelo de la fuente ────────────────────────
    import torch                                   # pesados: solo aquí
    from .data import load_test_images
    from .model import build_model
    from .train import predict

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(src_cfg, pretrained=False).to(device)
    model.load_state_dict(torch.load(run_dir / RUN_FILES["checkpoint"], map_location=device))
    test_imgs, test_hw = load_test_images(data / "test" / "test_images")
    assert set(test_hw.values()) == {FULL_SIZE}, "Test no es 2048×2048"
    test_stems = sorted(test_imgs)
    test_logits = predict(model, [test_imgs[s] for s in test_stems], src_cfg, device)
    sub = predictions_frame({
        s: labels_to_rles(*postproc.apply(sigmoid(l), post))
        for s, l in tqdm(zip(test_stems, test_logits), total=len(test_stems), desc="Test")
    })
    sub.to_csv(run_dir / "submission.csv", index=False)
    sub.to_csv(out_dir / "submission.csv", index=False)

    # ── Validación y registro ────────────────────────────────────────
    assert_valid_run(run_dir, split, split_path)
    mA, mB = res_A.metrics, res_B.metrics
    registry = out_dir / "registro.csv"
    append_registry(registry if registry.exists() else repo / "registro.csv", registry, {
        "run_id": run_id, "fecha": run_config["created_at"][:10],
        "config": cfg_path.name, "commit": commit[:7],
        "perdida": src_cfg["entrenamiento"]["perdida"],
        "etiquetas": src_cfg["pre"]["etiquetas"], "aumentos": src_cfg["pre"]["aumentos"],
        "post": f"umbral_cc u={post['umbral']} a={post['area_min_512']}",
        "epoca": src.config["selection"]["best_epoch"],
        "pq_A": round(mA["pq"], 4), "pq_B": round(mB["pq"], 4),
        "sq_B": round(mB["sq"], 4), "rq_B": round(mB["rq"], 4),
        "tp_B": mB["tp"], "fp_B": mB["fp"], "fn_B": mB["fn"],
        "nota": nota or f"post sobre {src.run_id}",
    })
    print(f"✔ {run_id} | PQ A {mA['pq']:.4f} | PQ B {mB['pq']:.4f}")
    return {"run_id": run_id, "run_dir": run_dir, "grid": grid, "val_A": mA, "val_B": mB}
