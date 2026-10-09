"""Comparar ejecuciones: conteos por anotación y bootstrap pareado.

Cada ejecución guarda sus **conteos**: para cada par anotador-imagen de
val_A y val_B, sus TP, FP, FN y suma de IoU (reglas oficiales), más las
relaciones del gráfico oficial (IoU > 0). Se acumulan en ``conteos.csv``,
igual que ``registro.csv``, así que comparar no necesita las carpetas
de las ejecuciones: basta con ese fichero.

Bootstrap pareado por días: se sortean días de val_B con reemplazo y,
sobre el mismo sorteo, se calcula el PQ de cada modelo y su diferencia
con la referencia. Se sortean días (no imágenes) porque las imágenes de
un mismo día son casi iguales.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .contract import IOU_THR

COUNT_COLS = ["n_gt", "n_pred", "tp", "fp", "fn", "sum_iou", "gt_1n", "pred_n1", "gt_1_0", "pred_0_1"]
CONTEOS_COLUMNS = ["run_id", "subset", "annotator_image", "date", *COUNT_COLS]


def pair_counts(overlap_df: pd.DataFrame) -> pd.DataFrame:
    """Conteos de cada par anotador-imagen a partir de la tabla de solapes.

    - ``tp``, ``fp``, ``fn``, ``sum_iou``: reglas oficiales (IoU > 0,5).
    - ``gt_1n``: filamentos reales que tocan ≥ 2 predicciones (fragmentados).
    - ``pred_n1``: predicciones que tocan ≥ 2 filamentos reales (fusiones).
    - ``gt_1_0`` / ``pred_0_1``: sin contacto alguno (perdidos / inventados).
    Las cuatro últimas, con IoU > 0, como el gráfico oficial.
    """
    rows = []
    for r in overlap_df.itertuples(index=False):
        iou = np.asarray(r.iou_matrix, dtype=float).reshape(r.n_gt, r.n_pred)
        hit, touch = iou > IOU_THR, iou > 0
        rows.append({
            "annotator_image": r.annotator_image,
            "date": r.annotator_image.split("-", 1)[1][:8],
            "n_gt": int(r.n_gt),
            "n_pred": int(r.n_pred),
            "tp": int(hit.sum()),
            "fp": int((hit.sum(0) == 0).sum()),
            "fn": int((hit.sum(1) == 0).sum()),
            "sum_iou": float(iou[hit].sum()),
            "gt_1n": int((touch.sum(1) >= 2).sum()),
            "pred_n1": int((touch.sum(0) >= 2).sum()),
            "gt_1_0": int((touch.sum(1) == 0).sum()),
            "pred_0_1": int((touch.sum(0) == 0).sum()),
        })
    return pd.DataFrame(rows)


def run_conteos(run_id: str, results: dict) -> pd.DataFrame:
    """``results``: subconjunto -> ``EvalResult`` (de ``metrics.evaluate``)."""
    parts = [pair_counts(res.overlap_df).assign(run_id=run_id, subset=s) for s, res in results.items()]
    return pd.concat(parts, ignore_index=True)[CONTEOS_COLUMNS]


def append_conteos(src: str | Path, dst: str | Path, new: pd.DataFrame) -> pd.DataFrame:
    """Lee los conteos del repo, añade los nuevos y escribe en ``dst``."""
    src = Path(src)
    old = pd.read_csv(src) if src.exists() else pd.DataFrame(columns=CONTEOS_COLUMNS)
    old = old[~old["run_id"].isin(new["run_id"].unique())]   # sin duplicados
    out = pd.concat([old, new], ignore_index=True)[CONTEOS_COLUMNS]
    out.to_csv(dst, index=False)
    return out


def _pq(c: np.ndarray) -> np.ndarray:
    """PQ global de conteos sumados; columnas tp, fp, fn, sum_iou."""
    den = c[..., 0] + 0.5 * (c[..., 1] + c[..., 2])
    return np.where(den > 0, c[..., 3] / np.maximum(den, 1e-12), 0.0)


def bootstrap(
    conteos: pd.DataFrame,
    run_ids: list[str],
    ref: str,
    subset: str = "val_B",
    n: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    """PQ con intervalo al 95 % y diferencia con ``ref``, por días.

    Devuelve una fila por ejecución: PQ observado, su intervalo, la
    diferencia con la referencia, su intervalo y la fracción de sorteos
    en que la ejecución supera a la referencia.
    """
    data = conteos[(conteos["subset"] == subset) & conteos["run_id"].isin(run_ids)]
    missing = set(run_ids) - set(data["run_id"])
    if missing:
        raise KeyError(f"Sin conteos para: {sorted(missing)}")
    days = sorted(data["date"].unique())
    cols = ["tp", "fp", "fn", "sum_iou"]
    # (ejecución, día, conteo): suma por día de cada ejecución
    by_day = np.stack([
        data[data["run_id"] == rid].groupby("date")[cols].sum().reindex(days, fill_value=0).to_numpy(float)
        for rid in run_ids
    ])
    pairs = data.groupby("run_id")["annotator_image"].nunique()
    if pairs.nunique() != 1:
        raise ValueError(f"Las ejecuciones no cubren los mismos pares: {pairs.to_dict()}")

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(days), size=(n, len(days)))      # mismo sorteo para todas
    boot = _pq(by_day[:, idx, :].sum(axis=2))                    # (ejecución, n)
    point = _pq(by_day.sum(axis=1))
    r = run_ids.index(ref)
    diff = boot - boot[r]

    out = pd.DataFrame({
        "run_id": run_ids,
        "pq": point,
        "pq_lo": np.percentile(boot, 2.5, axis=1),
        "pq_hi": np.percentile(boot, 97.5, axis=1),
        "dif": point - point[r],
        "dif_lo": np.percentile(diff, 2.5, axis=1),
        "dif_hi": np.percentile(diff, 97.5, axis=1),
        "p_mejor": (diff > 0).mean(axis=1),
    })
    out["significativa"] = (out["dif_lo"] > 0) | (out["dif_hi"] < 0)
    out.loc[r, ["dif_lo", "dif_hi", "p_mejor", "significativa"]] = [0.0, 0.0, np.nan, False]
    return out.round(4)


def asignaciones(conteos: pd.DataFrame, run_ids: list[str], subset: str = "val_B") -> pd.DataFrame:
    """Totales de relaciones del gráfico oficial por ejecución (lo que pidió Manuel)."""
    return (
        conteos[(conteos["subset"] == subset) & conteos["run_id"].isin(run_ids)]
        .groupby("run_id")[COUNT_COLS].sum()
        .reindex(run_ids)
        .assign(
            pct_gt_fragmentados=lambda d: (100 * d["gt_1n"] / d["n_gt"]).round(1),
            pct_pred_fusiones=lambda d: (100 * d["pred_n1"] / d["n_pred"]).round(1),
        )
        .reset_index()
    )


def conteos_desde_input(
    data_dir: str | Path,
    repo_dir: str | Path,
    root: str | Path = "/kaggle/input",
    out_dir: str | Path = "/kaggle/working",
) -> pd.DataFrame:
    """Calcula los conteos de las ejecuciones ya hechas que haya en ``root``.

    Para ejecuciones anteriores a que existiera ``conteos.csv``: busca
    carpetas con ``run_config.json``, evalúa sus predicciones guardadas
    en val_A y val_B y añade sus conteos (sin reentrenar).
    """
    from . import contract
    from .gt import build_gt_df, load_coco
    from .metrics import evaluate
    from .runs import read_run
    from .split import load_split

    data, repo, out_dir = Path(data_dir), Path(repo_dir), Path(out_dir)
    ann = next((data / "train").glob("*.json"))
    split = load_split(repo / contract.SPLIT_PATH, ann)
    coco = load_coco(ann)
    gt = {s: build_gt_df(coco, split["subsets"][s]) for s in ("val_A", "val_B")}

    dst = out_dir / "conteos.csv"
    acc = None
    for cfg in sorted(Path(root).rglob(contract.RUN_FILES["config"])):
        run = read_run(cfg.parent)
        res = {s: evaluate(g, run.val_predictions) for s, g in gt.items()}
        acc = append_conteos(dst if dst.exists() else repo / "conteos.csv", dst, run_conteos(run.run_id, res))
        print(f"✔ conteos de {run.run_id} | PQ B {res['val_B'].metrics['pq']:.4f}")
    if acc is None:
        raise FileNotFoundError(f"No hay ejecuciones en {root}")
    return acc
