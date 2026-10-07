import numpy as np
import pytest
import yaml

from filseg import postproc
from filseg.config import load_config
from filseg.metrics import counts_from_labels, evaluate

from conftest import frame


def test_herencia_de_config(tmp_path):
    (tmp_path / "base.yaml").write_text(yaml.safe_dump(
        {"nombre": "base", "entrenamiento": {"perdida": "bce_dice", "lr": 1e-3}}
    ))
    (tmp_path / "exp.yaml").write_text(yaml.safe_dump(
        {"hereda": "base.yaml", "nombre": "exp", "entrenamiento": {"perdida": "cldice"}}
    ))
    cfg = load_config(tmp_path / "exp.yaml")
    assert cfg["nombre"] == "exp"
    assert cfg["entrenamiento"] == {"perdida": "cldice", "lr": 1e-3}  # lr heredado


def test_config_base_del_repo():
    from pathlib import Path
    cfg = load_config(Path(__file__).parents[1] / "configs" / "humo.yaml")
    assert cfg["humo"] is True
    assert cfg["entrenamiento"]["perdida"] == "bce_dice"
    assert cfg["post"] == {"metodo": "umbral_cc", "umbral": 0.5, "area_min_512": 0, "conectividad": 8}


def test_umbral_cc():
    prob = np.zeros((64, 64), np.float32)
    prob[0:10, 0:10] = 0.9      # 100 px
    prob[20:22, 20:22] = 0.9    # 4 px
    prob[40:50, 40:50] = 0.5    # justo en el umbral: fuera
    labels, n = postproc.umbral_cc(prob)
    assert n == 2
    labels, n = postproc.umbral_cc(prob, area_min=5)
    assert n == 1 and (labels == 1).sum() == 100


def test_apply_reescala_a_2048():
    prob = np.zeros((512, 512), np.float32)
    prob[100:110, 100:110] = 0.9
    post = {"metodo": "umbral_cc", "umbral": 0.5, "area_min_512": 0}
    labels, n = postproc.apply(prob, post)
    assert labels.shape == (2048, 2048) and n == 1
    area = (labels == 1).sum()
    assert 1300 < area < 1800            # ~100 px × 16
    post["area_min_512"] = 101           # 101×16 > área -> se elimina
    assert postproc.apply(prob, post)[1] == 0


def test_recuento_512_igual_que_el_oficial():
    """El recuento rápido da lo mismo que la métrica oficial."""
    rng = np.random.default_rng(1)
    gt_masks, labels = [], np.zeros((512, 512), np.int32)
    for j in range(6):
        r, c = rng.integers(0, 450, 2)
        g = np.zeros((512, 512), bool)
        g[r:r + 20, c:c + 30] = True
        gt_masks.append(g)
        dr, dc = rng.integers(-12, 12, 2)
        labels[max(r + dr, 0):r + dr + 20, max(c + dc, 0):c + dc + 30] = j + 1
    labels, n = postproc.umbral_cc((labels > 0).astype(np.float32))  # sin solapes
    fast = counts_from_labels([np.flatnonzero(g) for g in gt_masks], labels, n)

    def big(m):  # mismo dibujo en un lienzo 2048
        out = np.zeros((2048, 2048), np.uint8)
        out[:512, :512] = m
        return out

    gt = frame([(f"ana-img1_{i + 1}", big(g)) for i, g in enumerate(gt_masks)])
    pred = frame([(f"img1_{j}", big(labels == j)) for j in range(1, n + 1)])
    m = evaluate(gt, pred).metrics
    assert (fast["tp"], fast["fp"], fast["fn"]) == (m["tp"], m["fp"], m["fn"])
    assert fast["sum_iou"] == pytest.approx(m["sum_iou"])


def test_rejilla_elige_umbral_y_area():
    """La rejilla descarta el ruido con el umbral y el área."""
    from filseg.postrun import tune

    prob = np.zeros((2048, 2048), np.float32)
    prob[100:140, 100:180] = 0.9      # filamento real
    prob[500:540, 500:580] = 0.55     # mancha dudosa
    prob[900:902, 900:902] = 0.9      # punto de 4 px
    gt = np.zeros((2048, 2048), bool)
    gt[100:140, 100:180] = True
    grid = tune({"img1": prob}, ["ana-img1"], {"ana-img1": [np.flatnonzero(gt)]},
                umbrales=[0.5, 0.6], areas_512=[0, 1])
    best = grid.iloc[0]
    assert (best.umbral, best.area_min_512) == (0.6, 1)   # 1 px a 512 = 16 a 2048
    assert best.pq == pytest.approx(1.0)
    worst = grid.iloc[-1]
    assert (worst.umbral, worst.area_min_512, worst.fp) == (0.5, 0, 2)
