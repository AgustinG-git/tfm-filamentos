"""Nuevo post-proceso sobre una ejecución ya entrenada (sin reentrenar).

Parte de los logits guardados de una ejecución (la "fuente"):

1. Busca en val_A el mejor umbral y área mínima (``post.ajuste``), a
   2048 y con las reglas oficiales. Dos métodos:
   - ``rejilla``: listas de valores, se prueban todas las combinaciones;
   - ``optuna``: rangos continuos, búsqueda bayesiana (TPE) con un
     número fijo de intentos y semilla fija.
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
from .compare import append_conteos, medidas, run_conteos
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


def score(probs_full: dict, pairs, gt_idx: dict, umbral: float, area_512: float) -> dict:
    """PQ en ``pairs`` con un umbral y un área (a 512) concretos.

    Usa ``postproc.umbral_cc``, el mismo código que la predicción final.
    """
    scale = FULL_SIZE[0] * FULL_SIZE[1] / contract.TRAIN_SIZE**2
    labels = {s: postproc.umbral_cc(p, umbral=umbral, area_min=area_512 * scale) for s, p in probs_full.items()}
    counts = [counts_from_labels(gt_idx[ai], *labels[ai.split("-", 1)[1]]) for ai in pairs]
    return {
        "umbral": umbral, "area_min_512": area_512, "pq": pq_from_counts(counts),
        **{k: sum(c[k] for c in counts) for k in ("tp", "fp", "fn")},
    }


def _sorted(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows).sort_values("pq", ascending=False, kind="stable").reset_index(drop=True)


def tune(probs_full: dict, pairs, gt_idx: dict, umbrales, areas_512) -> pd.DataFrame:
    """Rejilla: todas las combinaciones (umbral, área a 512)."""
    combos = list(itertools.product(umbrales, areas_512))
    return _sorted([score(probs_full, pairs, gt_idx, u, a) for u, a in tqdm(combos, desc="Rejilla")])


def tune_optuna(probs_full: dict, pairs, gt_idx: dict, ajuste: dict, seed: int = 42) -> pd.DataFrame:
    """Búsqueda bayesiana (TPE) en rangos continuos.

    ``ajuste``: ``umbral: [min, max]``, ``area_min_512: [min, max]``
    (si falta, área fija 0) e ``intentos``.
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    u_lo, u_hi = ajuste["umbral"]
    a_lo, a_hi = ajuste.get("area_min_512", [0, 0])
    rows = []

    def objective(trial):
        u = trial.suggest_float("umbral", u_lo, u_hi)
        a = trial.suggest_float("area_min_512", a_lo, a_hi) if a_hi > a_lo else float(a_lo)
        rows.append(score(probs_full, pairs, gt_idx, u, a))
        return rows[-1]["pq"]

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=ajuste["intentos"], show_progress_bar=True)
    return _sorted(rows)


def score_pers(probs512: dict, probs_full: dict, pairs, gt_idx: dict,
               umbral: float, tau: float, delta: float, area=None) -> dict:
    """PQ en ``pairs`` con el post-proceso por persistencia (E2).
    Con ``area`` (a 512), combinado con el filtro de área."""
    labels = {s: postproc.persistencia_full(probs512[s], pf, umbral, tau, delta, area)
              for s, pf in probs_full.items()}
    counts = [counts_from_labels(gt_idx[ai], *labels[ai.split("-", 1)[1]]) for ai in pairs]
    return {
        "umbral": umbral, "tau": tau, "delta": delta, "area_min_512": area,
        "pq": pq_from_counts(counts),
        **{k: sum(c[k] for c in counts) for k in ("tp", "fp", "fn")},
    }


def tune_optuna_pers(probs512: dict, probs_full: dict, pairs, gt_idx: dict,
                     ajuste: dict, seed: int = 42) -> pd.DataFrame:
    """TPE sobre (umbral, tau, delta). Rangos en ``ajuste``; delta
    opcional (si falta, sin fusión)."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    u_lo, u_hi = ajuste["umbral"]
    t_lo, t_hi = ajuste["tau"]
    d_lo, d_hi = ajuste.get("delta", [0, 0])
    area_rango = ajuste.get("area_min_512")      # None: E2 puro
    rows = []

    def objective(trial):
        u = trial.suggest_float("umbral", u_lo, u_hi)
        t = trial.suggest_float("tau", t_lo, t_hi)
        d = trial.suggest_float("delta", d_lo, min(d_hi, u - 0.01)) if d_hi > d_lo else float(d_lo)
        a = trial.suggest_float("area_min_512", *area_rango) if area_rango else None
        rows.append(score_pers(probs512, probs_full, pairs, gt_idx, u, t, d, a))
        return rows[-1]["pq"]

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=ajuste["intentos"], show_progress_bar=True)
    return _sorted(rows)


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
    post_metodo = cfg["post"]["metodo"]
    if not ajuste or post_metodo not in ("umbral_cc", "persistencia"):
        raise KeyError("La config necesita post.metodo (umbral_cc | persistencia) y post.ajuste")
    if post_metodo == "persistencia" and ajuste.get("metodo") != "optuna":
        raise KeyError("persistencia solo se ajusta con optuna")
    metodo = ajuste.get("metodo", "rejilla")
    if metodo not in ("rejilla", "optuna"):
        raise KeyError(f"post.ajuste.metodo '{metodo}' no existe: rejilla | optuna")

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
    gt_A = gt_idx_full(coco, A)
    if post_metodo == "persistencia":
        probs512_A = {s: sigmoid(logits[pos[s]]) for s in stems_A}
        grid = tune_optuna_pers(probs512_A, probs_A, A, gt_A, ajuste, seed=cfg.get("semilla", 42))
        del probs512_A
    elif metodo == "optuna":
        grid = tune_optuna(probs_A, A, gt_A, ajuste, seed=cfg.get("semilla", 42))
    else:
        grid = tune(probs_A, A, gt_A, ajuste["umbrales"], ajuste["areas_min_512"])
    best = grid.iloc[0]
    del probs_A
    if post_metodo == "persistencia":
        post = {
            "metodo": "persistencia",
            "umbral": float(best["umbral"]),
            "tau": float(best["tau"]),
            "delta": float(best["delta"]),
        }
        if ajuste.get("area_min_512"):
            post["area_min_512"] = float(best["area_min_512"])
        print(f"Elegido en val_A (persistencia): umbral {post['umbral']:.3f} | "
              f"tau {post['tau']:.3f} | delta {post['delta']:.3f} | "
              f"área {post.get('area_min_512', '-')} | PQ {best['pq']:.4f}")
    else:
        post = {
            "metodo": "umbral_cc",
            "umbral": float(best["umbral"]),          # exacto: el redondeo
            "area_min_512": float(best["area_min_512"]),  # solo al mostrarlo
            "conectividad": cfg["post"].get("conectividad", 8),
        }
        print(f"Elegido en val_A ({metodo}): umbral {post['umbral']:.3f} | "
              f"área {post['area_min_512']:.1f} | PQ {best['pq']:.4f}")

    # ── Predicciones y evaluación oficial ────────────────────────────
    preds = {
        s: labels_to_rles(*postproc.apply(sigmoid(l), post))
        for s, l in tqdm(zip(stems, logits), total=len(stems), desc="Val 2048")
    }
    pred_df = predictions_frame(preds)
    res_A = evaluate(build_gt_df(coco, A), pred_df)
    res_B = evaluate(build_gt_df(coco, B), pred_df)
    if not np.isclose(res_A.metrics["pq"], best["pq"], atol=1e-6):
        raise AssertionError(f"PQ de la búsqueda {best['pq']} ≠ oficial {res_A.metrics['pq']}")

    commit = git_commit(repo)
    run_id = make_run_id(cfg["nombre"], commit)
    run_dir = out_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    grid.to_csv(run_dir / "busqueda_val_A.csv", index=False)
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
    conteos = run_conteos(run_id, {"val_A": res_A, "val_B": res_B})
    conteos.to_csv(run_dir / "conteos.csv", index=False)
    dst = out_dir / "conteos.csv"
    append_conteos(dst if dst.exists() else repo / "conteos.csv", dst, conteos)
    mA, mB = res_A.metrics, res_B.metrics
    registry = out_dir / "registro.csv"
    append_registry(registry if registry.exists() else repo / "registro.csv", registry, {
        "run_id": run_id, "fecha": run_config["created_at"][:10],
        "config": cfg_path.name, "commit": commit[:7],
        "perdida": src_cfg["entrenamiento"]["perdida"],
        "etiquetas": src_cfg["pre"]["etiquetas"], "aumentos": src_cfg["pre"]["aumentos"],
        "post": (f"persistencia u={post['umbral']:.3f} tau={post['tau']:.3f} d={post['delta']:.3f}"
                 + (f" a={post['area_min_512']:.1f}" if "area_min_512" in post else "")
                 if post["metodo"] == "persistencia"
                 else f"umbral_cc u={post['umbral']:.3f} a={post['area_min_512']:.1f}"),
        "epoca": src.config["selection"]["best_epoch"],
        "pq_A": round(mA["pq"], 4), "pq_B": round(mB["pq"], 4),
        "sq_B": round(mB["sq"], 4), "rq_B": round(mB["rq"], 4),
        "tp_B": mB["tp"], "fp_B": mB["fp"], "fn_B": mB["fn"],
        **medidas(conteos[conteos["subset"] == "val_B"]),
        "nota": nota or f"post sobre {src.run_id}",
    })
    print(f"✔ {run_id} | PQ A {mA['pq']:.4f} | PQ B {mB['pq']:.4f}")
    return {"run_id": run_id, "run_dir": run_dir, "grid": grid, "val_A": mA, "val_B": mB}
