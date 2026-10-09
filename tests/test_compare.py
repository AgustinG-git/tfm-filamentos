import numpy as np
import pandas as pd
import pytest

from filseg.compare import CONTEOS_COLUMNS, append_conteos, asignaciones, bootstrap, pair_counts, run_conteos
from filseg.metrics import evaluate

from conftest import frame, rect


def fake_conteos(run_id, tp, fp, fn, iou=0.7, days=40, seed=0):
    """Conteos sintéticos: mismos pares para todas las ejecuciones."""
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(days):
        for who in ("ana", "luis"):
            t = rng.poisson(tp)
            rows.append({
                "run_id": run_id, "subset": "val_B",
                "annotator_image": f"{who}-2022{d:04d}000000Bh", "date": f"2022{d:04d}",
                "n_gt": t + fn, "n_pred": t + fp, "tp": t, "fp": rng.poisson(fp),
                "fn": rng.poisson(fn), "sum_iou": t * iou,
                "gt_1n": 1, "pred_n1": 0, "gt_1_0": 0, "pred_0_1": 0,
            })
    return pd.DataFrame(rows)[CONTEOS_COLUMNS]


def test_conteos_coinciden_con_el_evaluador():
    gt = frame([("ana-img1_1", rect(0, 10, 0, 10)), ("ana-img1_2", rect(0, 10, 20, 30))])
    pred = frame([("img1_1", rect(0, 6, 0, 10)), ("img1_2", rect(7, 10, 0, 10)),
                  ("img1_3", rect(100, 110, 100, 110))])
    res = evaluate(gt, pred)
    c = pair_counts(res.overlap_df).iloc[0]
    m = res.metrics
    assert (c.tp, c.fp, c.fn) == (m["tp"], m["fp"], m["fn"])
    assert c.sum_iou == pytest.approx(m["sum_iou"])
    assert (c.gt_1n, c.pred_n1, c.gt_1_0, c.pred_0_1) == (1, 0, 1, 1)   # partido, perdido, inventado


def test_bootstrap_misma_ejecucion_no_es_significativa():
    a = fake_conteos("a", 6, 4, 4)
    b = a.assign(run_id="b")
    out = bootstrap(pd.concat([a, b]), ["a", "b"], ref="a")
    row = out.set_index("run_id").loc["b"]
    assert row.dif == 0 and not row.significativa


def test_bootstrap_detecta_una_mejora_clara_y_es_reproducible():
    a = fake_conteos("a", 6, 6, 4, seed=1)
    b = a.assign(run_id="b", fp=a["fp"] // 3)            # muchos menos FP
    data = pd.concat([a, b])
    out = bootstrap(data, ["a", "b"], ref="a")
    row = out.set_index("run_id").loc["b"]
    assert row.dif > 0 and row.dif_lo > 0 and row.significativa and row.p_mejor > 0.95
    assert out.equals(bootstrap(data, ["a", "b"], ref="a"))   # misma semilla


def test_bootstrap_exige_los_mismos_pares():
    a = fake_conteos("a", 6, 4, 4)
    b = a.assign(run_id="b").iloc[:-2]
    with pytest.raises(ValueError):
        bootstrap(pd.concat([a, b]), ["a", "b"], ref="a")


def test_append_conteos_sin_duplicados(tmp_path):
    a = fake_conteos("a", 6, 4, 4, days=3)
    append_conteos(tmp_path / "no_existe.csv", tmp_path / "c.csv", a)
    out = append_conteos(tmp_path / "c.csv", tmp_path / "c.csv", a)   # repetir no duplica
    assert len(out) == len(a)
    tabla = asignaciones(out, ["a"])
    assert tabla.loc[0, "gt_1n"] == len(a)
    assert run_conteos  # importable
