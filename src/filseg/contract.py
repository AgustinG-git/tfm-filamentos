"""Contrato v1: decisiones congeladas que comparten todas las ejecuciones.

Cambiar cualquier valor de este fichero invalida las comparaciones con las
ejecuciones anteriores. Si hace falta cambiar algo, se crea un contrato v2
(y un split v2 si afecta al split), nunca se edita v1. Ver CONTRACT.md.
"""

CONTRACT_VERSION = "v1"


class ContractError(ValueError):
    """Algo no cumple el contrato (ids, ficheros, split, solapes...)."""


# ── Split ────────────────────────────────────────────────────────────
SPLIT_VERSION = "v1"
SPLIT_PATH = "splits/split_v1.json"
SPLIT_SEED = 42
VAL_SIZE = 0.2    # fracción de días para validación
HALF_SIZE = 0.5   # fracción de días de val para B
SUBSETS = ("train", "val_A", "val_B")
TUNE_SUBSET = "val_A"     # selección y ajuste
REPORT_SUBSET = "val_B"   # cifra que se reporta

# ── Geometría ────────────────────────────────────────────────────────
FULL_SIZE = (2048, 2048)  # (alto, ancho) oficial
TRAIN_SIZE = 512

# ── Métrica ──────────────────────────────────────────────────────────
IOU_THR = 0.5             # TP si IoU > 0,5 (estricto)

# Selección del checkpoint durante el entrenamiento. Fija para todas
# las versiones: post-proceso estricto, sin heurísticas ajustadas.
SELECTION = {
    "subset": TUNE_SUBSET,
    "metric": "pq_global_512",
    "thr": 0.5,
    "min_area": 0,
}

# ── Salidas de una ejecución ─────────────────────────────────────────
RUN_FILES = {
    "config": "run_config.json",
    "val_predictions": "val_predictions.csv",
    "val_logits": "val_logits.npz",
    "history": "history.csv",
    "checkpoint": "checkpoint.pth",
}
OPTIONAL_RUN_FILES = {"submission": "submission.csv"}

REQUIRED_CONFIG_KEYS = (
    "run_id",
    "contract_version",
    "split_version",
    "split_sha256",
    "git_commit",
    "created_at",
    "config",          # la configuración YAML resuelta
    "selection",       # época elegida y su PQ en val_A
    "postproc",        # parámetros finales del post-proceso
    "metrics_val_A",   # PQ oficial en val_A según el pipeline
)

PRED_COLUMNS = ("filament_id", "segmentation_rle")
LOGITS_DTYPE = "float16"
