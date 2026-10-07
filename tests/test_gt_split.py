import json

import pandas as pd
import pytest
from sklearn.model_selection import GroupShuffleSplit

from filseg.contract import ContractError
from filseg.gt import build_gt_df, build_meta
from filseg.split import check_split, load_split, make_split, save_split, stems_of


def coco_example(n_days=40, bad_id=None, size=2048):
    images, anns = [], []
    for d in range(n_days):
        stem = f"2022{(d // 28) + 1:02d}{(d % 28) + 1:02d}000000Bh"
        for who in ("ana", "luis"):
            iid = f"{who}-{stem}"
            images.append({"id": iid, "file_name": f"{stem}.jpeg", "height": size, "width": size})
            anns.append({"image_id": iid, "segmentation": [[0, 0, 20, 0, 20, 10, 0, 10]]})
    if bad_id is not None:
        images[0]["id"] = bad_id
        anns[0]["image_id"] = bad_id
    return {"images": images, "annotations": anns}


# ── GT ───────────────────────────────────────────────────────────────
def test_meta_y_gt_en_formato_oficial():
    coco = coco_example(3)
    meta = build_meta(coco)
    assert len(meta) == 6
    assert meta["n_filaments"].eq(1).all()
    gt = build_gt_df(coco, annotator_images=["ana-20220101000000Bh"])
    assert gt["filament_id"].tolist() == ["ana-20220101000000Bh_1"]
    assert gt["area"].iloc[0] > 0


@pytest.mark.parametrize("bad_id", [
    "20220101000000Bh",            # sin anotador
    "ana-20229999000000Bh",        # no coincide con el fichero
    "ana_x-20220101000000Bh",      # '_' rompe el parseo oficial
])
def test_ids_incompatibles_con_el_oficial(bad_id):
    with pytest.raises(ContractError):
        build_meta(coco_example(3, bad_id=bad_id))


def test_tamano_distinto_de_2048():
    with pytest.raises(ContractError):
        build_meta(coco_example(3, size=1024))


# ── Split ────────────────────────────────────────────────────────────
def test_split_reproduce_la_baseline_original():
    meta = build_meta(coco_example())
    split = make_split(meta)
    # lógica original del notebook de baseline
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    _, va = next(gss.split(meta, groups=meta["date"]))
    val = meta.iloc[va].reset_index(drop=True)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    a, _ = next(gss2.split(val, groups=val["date"]))
    assert set(split["subsets"]["val_A"]) == set(val.iloc[a]["annotator_image"])
    assert set(split["subsets"]["val_A"]) | set(split["subsets"]["val_B"]) == set(val["annotator_image"])


def test_split_independiente_del_orden_de_filas():
    meta = build_meta(coco_example())
    shuffled = meta.sample(frac=1, random_state=1).reset_index(drop=True)
    s1, s2 = make_split(meta), make_split(shuffled)
    assert all(set(s1["subsets"][k]) == set(s2["subsets"][k]) for k in s1["subsets"])


def test_split_sin_fugas_y_anotadores_juntos():
    split = make_split(build_meta(coco_example()))
    a, b = stems_of(split, "val_A"), stems_of(split, "val_B")
    assert not a & b
    # los dos anotadores de una imagen caen en el mismo subconjunto
    for name, ids in split["subsets"].items():
        stems = pd.Series([i.split("-", 1)[1] for i in ids])
        assert stems.value_counts().eq(2).all(), name


def test_split_detecta_fugas():
    split = make_split(build_meta(coco_example()))
    split["subsets"]["val_B"].append(split["subsets"]["train"][0])
    with pytest.raises(ContractError):
        check_split(split)


def test_guardar_cargar_y_no_sobrescribir(tmp_path):
    ann_path = tmp_path / "train.json"
    ann_path.write_text(json.dumps(coco_example()))
    meta = build_meta(json.loads(ann_path.read_text()))
    split = make_split(meta, annotations_path=ann_path)
    path = save_split(split, tmp_path / "split_v1.json")
    assert load_split(path, ann_path)["subsets"] == split["subsets"]
    with pytest.raises(ContractError):
        save_split(split, path)
    ann_path.write_text(json.dumps(coco_example(41)))   # otro JSON
    with pytest.raises(ContractError):
        load_split(path, ann_path)
