"""Datos a 512: imágenes, etiquetas de entrenamiento y GT para selección.

Preproceso mínimo (imprescindible, no es un experimento):
- reducir a 512 con INTER_AREA (a 2048 no cabe en GPU);
- gris -> 3 canales y normalización ImageNet (encoder preentrenado).
"""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from .contract import TRAIN_SIZE
from .gt import ann_to_mask

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def resize_img(img, size=TRAIN_SIZE):
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def resize_mask(mask, size=TRAIN_SIZE):
    small = cv2.resize(mask.astype(np.float32), (size, size), interpolation=cv2.INTER_AREA)
    return small >= 0.5


@dataclass
class Cache:
    """Todo a 512, indexado por stem o por par anotador-imagen."""
    images: dict = field(default_factory=dict)    # stem -> uint8
    masks: dict = field(default_factory=dict)     # par -> uint8 (unión)
    gt_idx: dict = field(default_factory=dict)    # par -> [índices]


def load_cache(coco, meta, img_dir, pairs, workers=8) -> Cache:
    """Carga imágenes y máscaras de los pares pedidos."""
    img_dir = Path(img_dir)
    anns = defaultdict(list)
    for a in coco["annotations"]:
        anns[str(a["image_id"])].append(a)
    rows = meta.set_index("annotator_image").loc[list(pairs)]

    def load_pair(item):
        ai, r = item
        h, w = int(r.height), int(r.width)
        full = [ann_to_mask(a, h, w) for a in anns[ai]]
        union = np.maximum.reduce(full) if full else np.zeros((h, w), np.uint8)
        idx = [np.flatnonzero(m) for m in (resize_mask(f) for f in full) if m.any()]
        return ai, resize_mask(union).astype(np.uint8), idx

    def load_img(stem_file):
        stem, fname = stem_file
        img = cv2.imread(str(img_dir / fname), cv2.IMREAD_GRAYSCALE)
        return stem, resize_img(img)

    cache = Cache()
    stems = rows[["stem", "file_name"]].drop_duplicates("stem").itertuples(index=False)
    with ThreadPoolExecutor(workers) as ex:
        for stem, img in tqdm(ex.map(load_img, stems), desc="Imágenes"):
            cache.images[stem] = img
        for ai, mask, idx in tqdm(ex.map(load_pair, rows.iterrows()), total=len(rows), desc="Máscaras"):
            cache.masks[ai] = mask
            cache.gt_idx[ai] = idx
    return cache


def load_test_images(test_dir, workers=8) -> tuple[dict, dict]:
    """Imágenes de test a 512 y su tamaño original."""
    paths = sorted(p for p in Path(test_dir).iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})

    def load(p):
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        return p.stem, img.shape, resize_img(img)

    with ThreadPoolExecutor(workers) as ex:
        out = list(tqdm(ex.map(load, paths), total=len(paths), desc="Test"))
    return {stem: img for stem, _, img in out}, {stem: hw for stem, hw, _ in out}


# ── Etiquetas de entrenamiento ───────────────────────────────────────
def por_anotacion(pairs, cache):
    """Cada anotación es un ejemplo: (stem, máscara de ese anotador)."""
    return [(ai.split("-", 1)[1], cache.masks[ai]) for ai in pairs]


LABELS = {"por_anotacion": por_anotacion}


# ── Dataset ──────────────────────────────────────────────────────────
def to_tensor(img: np.ndarray) -> torch.Tensor:
    x = img.astype(np.float32)[None] / 255.0                # (1,H,W)
    x = (np.repeat(x, 3, 0) - MEAN[:, None, None]) / STD[:, None, None]
    return torch.from_numpy(x)


class FilamentDataset(Dataset):
    """Sin aumentos: la baseline no los usa."""

    def __init__(self, images: list, masks: list | None = None):
        self.images, self.masks = images, masks

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        x = to_tensor(self.images[i])
        if self.masks is None:
            return x
        return x, torch.from_numpy(self.masks[i].astype(np.float32))[None]
