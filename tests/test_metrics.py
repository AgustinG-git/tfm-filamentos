"""Los ejemplos de README_autoevaluacion como casos ejecutables."""

import numpy as np
import pytest

from filseg.contract import ContractError
from filseg.metrics import evaluate, filter_gt, find_overlaps
from filseg.official_metric import get_pq_score

from conftest import frame, rect


def counts(res):
    m = res.metrics
    return m["tp"], m["fp"], m["fn"]


def test_prediccion_perfecta(gt_100):
    pred = frame([("img1_1", rect(0, 10, 0, 10))])
    res = evaluate(gt_100, pred)
    assert counts(res) == (1, 0, 0)
    assert res.metrics["pq"] == pytest.approx(1.0)


def test_iou_exactamente_05_no_es_tp(gt_100):
    pred = frame([("img1_1", rect(0, 5, 0, 10))])   # 50 px dentro
    res = evaluate(gt_100, pred)
    assert counts(res) == (0, 1, 1)
    assert res.metrics["pq"] == 0.0


def test_fragmento_sobrante_casi_anula_el_acierto(gt_100):
    pred = frame([
        ("img1_1", rect(0, 6, 0, 10)),    # 60 px -> IoU 0,6
        ("img1_2", rect(7, 10, 0, 10)),   # 30 px -> IoU 0,3
    ])
    res = evaluate(gt_100, pred)
    assert counts(res) == (1, 1, 0)
    assert res.metrics["pq"] == pytest.approx(0.6 / 1.5)


def test_partir_por_la_mitad_da_1fn_2fp(gt_100):
    pred = frame([
        ("img1_1", rect(0, 9, 0, 5)),     # 45 px
        ("img1_2", rect(0, 9, 5, 10)),    # 45 px
    ])
    res = evaluate(gt_100, pred)
    assert counts(res) == (0, 2, 1)


def test_fusion_de_dos_iguales_da_2fn_1fp():
    gt = frame([
        ("ana-img1_1", rect(0, 10, 0, 10)),
        ("ana-img1_2", rect(0, 10, 20, 30)),
    ])
    fused = rect(0, 10, 0, 10) | rect(0, 10, 20, 30) | rect(0, 1, 10, 20)  # 210 px
    res = evaluate(gt, frame([("img1_1", fused)]))
    assert counts(res) == (0, 1, 2)


def test_cada_anotador_cuenta_por_separado():
    gt = frame([
        ("ana-img1_1", rect(0, 10, 0, 10)),
        ("luis-img1_1", rect(0, 10, 0, 10)),
    ])
    pred = frame([
        ("img1_1", rect(0, 10, 0, 10)),
        ("img1_2", rect(100, 110, 100, 110)),   # inventado
    ])
    res = evaluate(gt, pred)
    assert counts(res) == (2, 2, 0)              # el FP cuenta 2 veces
    assert res.metrics["n_annotator_images"] == 2


def test_imagen_sin_gt_no_se_evalua(gt_100):
    pred = frame([
        ("img1_1", rect(0, 10, 0, 10)),
        ("img2_1", rect(0, 10, 0, 10)),          # nadie anotó img2
    ])
    res = evaluate(gt_100, pred)
    assert counts(res) == (1, 0, 0)
    assert res.metrics["n_pred_ignored"] == 1


def test_duplicados_se_rechazan_y_explican_el_agujero():
    gt = frame([
        ("ana-img1_1", rect(0, 10, 0, 10)),
        ("ana-img1_2", rect(50, 60, 50, 60)),    # se pierde
    ])
    dup = frame([("img1_1", rect(0, 10, 0, 10)), ("img1_2", rect(0, 10, 0, 10))])
    assert find_overlaps(dup) == ["img1"]
    with pytest.raises(ContractError):
        evaluate(gt, dup)
    honest = evaluate(gt, frame([("img1_1", rect(0, 10, 0, 10))]))
    inflated = evaluate(gt, dup, check_overlaps=False)
    assert honest.metrics["pq"] == pytest.approx(1 / 1.5)
    assert inflated.metrics["pq"] == pytest.approx(2 / 2.5)


def test_componentes_coinciden_con_el_oficial():
    rng = np.random.default_rng(0)
    gt_items, pred_items = [], []
    for k in range(3):
        for j in range(4):
            r, c = rng.integers(0, 1900, 2)
            gt_items.append((f"ana-img{k}_{j + 1}", rect(r, r + 30, c, c + 40)))
            dr, dc = rng.integers(-15, 15, 2)
            pred_items.append((f"img{k}_{j + 1}", rect(r + dr, r + dr + 30, c + dc, c + dc + 40)))
    res = evaluate(frame(gt_items), frame(pred_items), check_overlaps=False)
    m = res.metrics
    assert m["pq"] == pytest.approx(get_pq_score(res.overlap_df), abs=1e-12)
    assert m["pq"] == pytest.approx(m["sq"] * m["rq"])


@pytest.mark.parametrize("bad_id", ["img_1_1", "img1", "img1_a"])
def test_ids_de_prediccion_invalidos(gt_100, bad_id):
    with pytest.raises(ContractError):
        evaluate(gt_100, frame([(bad_id, rect(0, 10, 0, 10))]))


def test_ids_de_gt_invalidos():
    gt = frame([("img1_1", rect(0, 10, 0, 10))])   # sin anotador
    with pytest.raises(ContractError):
        evaluate(gt, frame([("img1_1", rect(0, 10, 0, 10))]))


def test_filter_gt():
    gt = frame([
        ("ana-img1_1", rect(0, 10, 0, 10)),
        ("luis-img1_1", rect(0, 10, 0, 10)),
    ])
    assert filter_gt(gt, ["luis-img1"])["filament_id"].tolist() == ["luis-img1_1"]
