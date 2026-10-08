import numpy as np
import pandas as pd
import pytest

from filseg.contract import RUN_FILES, ContractError
from filseg.gt import build_meta
from filseg.runs import (
    assert_valid_run,
    labels_to_rles,
    make_run_config,
    predictions_frame,
    read_run,
    validate_run,
    write_run,
)
from filseg.split import make_split, save_split, stems_of

from test_gt_split import coco_example


@pytest.fixture
def split_and_path(tmp_path):
    split = make_split(build_meta(coco_example()))
    return split, save_split(split, tmp_path / "split_v1.json")


def fake_run(tmp_path, split, split_path, commit="abc1234def", extra_stem=None, overlap=False):
    stems = sorted(stems_of(split, "val_A", "val_B"))
    labels = np.zeros((2048, 2048), np.int32)
    labels[0:10, 0:10] = 1
    labels[20:30, 20:30] = 2
    preds = {stems[0]: labels_to_rles(labels, 2)}
    if overlap:
        preds[stems[0]] = preds[stems[0]] + preds[stems[0]][:1]
    if extra_stem:
        preds[extra_stem] = labels_to_rles(labels, 1)
    cfg = make_run_config(
        run_id="20261007-1200_base_abc1234",
        config={"nombre": "base"},
        split_path=split_path,
        git_commit=commit,
        selection={"subset": "val_A", "best_epoch": 27, "pq": 0.4},
        postproc={"thr": 0.65, "min_area": 25},
        metrics_val_A={"pq": 0.4},
    )
    run_dir = tmp_path / "runs" / cfg["run_id"]
    write_run(
        run_dir,
        run_config=cfg,
        val_predictions=predictions_frame(preds),
        history=pd.DataFrame({"epoch": [1], "val_pq_A": [0.4]}),
        val_logits=(stems, np.zeros((len(stems), 512, 512))),
    )
    (run_dir / RUN_FILES["checkpoint"]).write_bytes(b"")
    return run_dir


def test_ejecucion_valida(tmp_path, split_and_path):
    split, path = split_and_path
    run_dir = fake_run(tmp_path, split, path)
    assert validate_run(run_dir, split, path) == []
    run = read_run(run_dir)
    assert run.val_predictions["filament_id"].str.endswith(("_1", "_2")).all()
    stems, logits = run.logits()
    assert logits.dtype == np.float16 and len(stems) == logits.shape[0]


def test_falta_un_fichero(tmp_path, split_and_path):
    split, path = split_and_path
    run_dir = fake_run(tmp_path, split, path)
    (run_dir / RUN_FILES["history"]).unlink()
    assert validate_run(run_dir, split, path) == [f"falta {RUN_FILES['history']}"]


def test_problemas_detectados(tmp_path, split_and_path):
    split, path = split_and_path
    train_stem = sorted(stems_of(split, "train"))[0]
    run_dir = fake_run(tmp_path, split, path, commit="unknown", extra_stem=train_stem, overlap=True)
    problems = " | ".join(validate_run(run_dir, split, path))
    assert "git_commit" in problems
    assert "fuera de val" in problems
    assert "solapadas" in problems
    with pytest.raises(ContractError):
        assert_valid_run(run_dir, split, path)


def test_labels_to_rles_igual_que_codificar_una_a_una():
    """La versión rápida da exactamente los mismos RLE."""
    from pycocotools import mask as mask_utils
    from filseg.postproc import umbral_cc

    rng = np.random.default_rng(3)
    prob = (rng.random((300, 200)) > 0.7).astype(np.float32)   # muchas manchas
    prob[10:60, 20:25] = 1.0                                    # y una tira
    labels, n = umbral_cc(prob)
    assert n > 100
    fast = labels_to_rles(labels, n)
    slow = [
        mask_utils.encode(np.asfortranarray((labels == i).astype(np.uint8)))["counts"].decode()
        for i in range(1, n + 1)
    ]
    assert fast == slow
    empty = np.zeros((5, 4), np.int32)
    empty[0, 0] = 1
    assert labels_to_rles(empty, 2)[1] == mask_utils.encode(np.zeros((5, 4), np.uint8, order="F"))["counts"].decode()
