"""Una ejecución completa según el contrato v1.

1. Entrena con la config.
2. Elige la época en val_A (PQ a 512, umbral 0,5, área 0).
3. Predice val, post-procesa con la config y evalúa con la métrica
   oficial en val_A y val_B.
4. Guarda las salidas del contrato, los gráficos de la rúbrica, la
   submission y una fila nueva del registro.

Uso en Kaggle::

    from filseg.train import run
    run("configs/base.yaml", data_dir=DATA, repo_dir="tfm-filamentos")
"""

from __future__ import annotations

import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import expit
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from . import contract, postproc
from .config import load_config
from .data import LABELS, FilamentDataset, load_cache, load_test_images
from .gt import build_gt_df, build_meta, load_coco
from .losses import LOSSES, build_loss
from .metrics import counts_from_labels, evaluate, plot_official, pq_from_counts
from .model import ARCHS, build_model
from .runs import (
    append_registry, assert_valid_run, git_commit, labels_to_rles,
    make_run_config, make_run_id, predictions_frame, write_run,
)
from .split import load_split


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _worker_init(_):
    s = torch.initial_seed() % 2**32  # semilla por worker
    np.random.seed(s)
    random.seed(s)


def check_pieces(cfg: dict) -> None:
    """Falla al arrancar si alguna pieza no existe."""
    checks = [
        (cfg["pre"]["etiquetas"], LABELS, "etiquetas"),
        (cfg["modelo"]["arquitectura"], ARCHS, "arquitectura"),
        (cfg["entrenamiento"]["perdida"], LOSSES, "pérdida"),
        (cfg["post"]["metodo"], postproc.POSTPROC, "post-proceso"),
    ]
    for name, registry, kind in checks:
        if name not in registry:
            raise KeyError(f"{kind} '{name}' no existe. Disponibles: {list(registry)}")
    if cfg["pre"]["aumentos"] != "ninguno":
        raise KeyError("Aumentos aún no implementados: usa 'ninguno'")


def sigmoid(x: np.ndarray) -> np.ndarray:
    return expit(x.astype(np.float32))  # sin avisos de overflow


@torch.no_grad()
def predict(model, images: list, cfg: dict, device: str) -> np.ndarray:
    """Logits (N, 512, 512) float32."""
    model.eval()
    loader = DataLoader(  # sin workers: datos ya en memoria
        FilamentDataset(images), batch_size=cfg["entrenamiento"]["batch"],
        shuffle=False, num_workers=0,
    )
    amp = cfg["entrenamiento"]["amp"] and device == "cuda"
    out = []
    for x in loader:
        with torch.autocast(device_type="cuda", enabled=amp):
            out.append(model(x.to(device)).float()[:, 0].cpu().numpy())
    return np.concatenate(out)


def selection_pq(logits, stems, pairs, cache) -> float:
    """PQ a 512 en los pares dados, con el post-proceso del contrato."""
    sel = contract.SELECTION
    labels = {
        s: postproc.umbral_cc(sigmoid(l), umbral=sel["thr"], area_min=sel["min_area"])
        for s, l in zip(stems, logits)
    }
    rows = [counts_from_labels(cache.gt_idx[ai], *labels[ai.split("-", 1)[1]]) for ai in pairs]
    return pq_from_counts(rows)


def run(
    config: str | Path,
    *,
    data_dir: str | Path,
    repo_dir: str | Path,
    out_dir: str | Path = "/kaggle/working",
    nota: str = "",
) -> dict:
    repo, data, out_dir = Path(repo_dir), Path(data_dir), Path(out_dir)
    cfg_path = Path(config) if Path(config).is_absolute() else repo / config
    cfg = load_config(cfg_path)
    check_pieces(cfg)
    humo = bool(cfg.get("humo", False))
    ent = cfg["entrenamiento"]
    epochs = 2 if humo else ent["epocas"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_everything(cfg["semilla"])

    commit = git_commit(repo)
    run_id = make_run_id(cfg["nombre"] + ("-humo" if humo else ""), commit)
    run_dir = out_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"▶ {run_id} | {device}")

    # ── Datos ────────────────────────────────────────────────────────
    ann = next((data / "train").glob("*.json"))
    split_path = repo / contract.SPLIT_PATH
    split = load_split(split_path, ann)
    coco = load_coco(ann)
    meta = build_meta(coco)
    tr, A, B = (split["subsets"][k] for k in contract.SUBSETS)
    if humo:
        tr, A, B = tr[:50], A[:20], B[:20]
    cache = load_cache(coco, meta, data / "train" / "train_images", tr + A + B)

    samples = LABELS[cfg["pre"]["etiquetas"]](tr, cache)
    train_loader = DataLoader(
        FilamentDataset([cache.images[s] for s, _ in samples], [m for _, m in samples]),
        batch_size=ent["batch"], shuffle=True, drop_last=True,
        num_workers=ent["workers"], pin_memory=True, worker_init_fn=_worker_init,
        generator=torch.Generator().manual_seed(cfg["semilla"]),
        persistent_workers=ent["workers"] > 0,  # no recrea procesos
    )
    stems_A = sorted({ai.split("-", 1)[1] for ai in A})
    imgs_A = [cache.images[s] for s in stems_A]

    # ── Entrenamiento con selección en val_A ─────────────────────────
    model = build_model(cfg).to(device)
    criterion = build_loss(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=ent["lr"], weight_decay=ent["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    amp = ent["amp"] and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    ckpt = run_dir / contract.RUN_FILES["checkpoint"]

    history, best_pq, best_epoch = [], -1.0, 0
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        total = 0.0
        for x, y in tqdm(train_loader, desc=f"Época {epoch}", leave=False):
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=amp):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += loss.item() * len(x)
        lr = opt.param_groups[0]["lr"]
        sched.step()

        pq_A = selection_pq(predict(model, imgs_A, cfg, device), stems_A, A, cache)
        history.append({
            "epoch": epoch, "train_loss": total / len(train_loader.dataset),
            "pq_512_A": pq_A, "lr": lr, "segundos": round(time.time() - t0),
        })
        flag = ""
        if pq_A > best_pq:
            best_pq, best_epoch = pq_A, epoch
            torch.save(model.state_dict(), ckpt)
            flag = " ★"
        print(f"[{epoch:02d}/{epochs}] loss {history[-1]['train_loss']:.4f} | PQ@512 A {pq_A:.4f}{flag}")

    # ── Predicción y evaluación oficial ──────────────────────────────
    model.load_state_dict(torch.load(ckpt, map_location=device))
    stems_val = sorted({ai.split("-", 1)[1] for ai in A + B})
    logits = predict(model, [cache.images[s] for s in stems_val], cfg, device)
    preds = {
        s: labels_to_rles(*postproc.apply(sigmoid(l), cfg["post"]))
        for s, l in tqdm(zip(stems_val, logits), total=len(stems_val), desc="Val 2048")
    }
    pred_df = predictions_frame(preds)
    res_A = evaluate(build_gt_df(coco, A), pred_df)
    res_B = evaluate(build_gt_df(coco, B), pred_df)
    plot_official(res_B.overlap_df, run_dir / "figuras", tag="val_B")

    # ── Salidas del contrato ─────────────────────────────────────────
    run_config = make_run_config(
        run_id=run_id, config=cfg, split_path=split_path, git_commit=commit,
        selection={**contract.SELECTION, "best_epoch": best_epoch, "pq": best_pq},
        postproc=cfg["post"], metrics_val_A=res_A.metrics,
        metrics_val_B=res_B.metrics, nota=nota,
    )
    write_run(
        run_dir, run_config=run_config, val_predictions=pred_df,
        history=pd.DataFrame(history), val_logits=(stems_val, logits),
    )

    # ── Submission ───────────────────────────────────────────────────
    if not humo:
        test_imgs, test_hw = load_test_images(data / "test" / "test_images")
        assert set(test_hw.values()) == {contract.FULL_SIZE}, "Test no es 2048×2048"
        test_stems = sorted(test_imgs)
        test_logits = predict(model, [test_imgs[s] for s in test_stems], cfg, device)
        sub = predictions_frame({
            s: labels_to_rles(*postproc.apply(sigmoid(l), cfg["post"]))
            for s, l in tqdm(zip(test_stems, test_logits), total=len(test_stems), desc="Test")
        })
        sub.to_csv(run_dir / "submission.csv", index=False)
        sub.to_csv(out_dir / "submission.csv", index=False)

    # ── Validación y registro ────────────────────────────────────────
    mA, mB = res_A.metrics, res_B.metrics
    if not humo:
        assert_valid_run(run_dir, split, split_path)
        append_registry(repo / "registro.csv", out_dir / "registro.csv", {
            "run_id": run_id, "fecha": run_config["created_at"][:10],
            "config": cfg_path.name, "commit": commit[:7],
            "perdida": ent["perdida"], "etiquetas": cfg["pre"]["etiquetas"],
            "aumentos": cfg["pre"]["aumentos"],
            "post": f"{cfg['post']['metodo']} u={cfg['post'].get('umbral', 0.5)} "
                    f"a={cfg['post'].get('area_min_512', 0)}",
            "epoca": best_epoch, "pq_A": round(mA["pq"], 4), "pq_B": round(mB["pq"], 4),
            "sq_B": round(mB["sq"], 4), "rq_B": round(mB["rq"], 4),
            "tp_B": mB["tp"], "fp_B": mB["fp"], "fn_B": mB["fn"], "nota": nota,
        })

    print(f"✔ {run_id} | época {best_epoch} | PQ A {mA['pq']:.4f} | PQ B {mB['pq']:.4f}")
    return {"run_id": run_id, "run_dir": run_dir, "val_A": mA, "val_B": mB}
