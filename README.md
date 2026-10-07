# tfm-filamentos

Segmentación de instancias de filamentos solares en imágenes H-alfa
(MAGFiLO, *Filament Segmentation Challenge 2026*). Código del TFM del
Máster en Lógica, Computación e Inteligencia Artificial (Universidad de
Sevilla).

Cada versión del modelo es un **pipeline completo** (preproceso,
entrenamiento, post-proceso) definido por una configuración. Todas las
versiones se evalúan con el **mismo evaluador** y bajo el mismo
**contrato** (ver [CONTRACT.md](CONTRACT.md)).

## Estado

| Paso | Estado |
|---|---|
| 1. Contrato v1: split, GT oficial, métrica oficial, salidas | ✅ Hecho |
| 2. Arquitectura mínima: datos, modelo, BCE+Dice, entrenamiento, post-proceso | Pendiente |
| 3. Validación sin reentrenar: checkpoint viejo por el pipeline nuevo → 0,3819 | Pendiente |
| 4. Baseline sólida con el contrato (B0, B1, B2) | Pendiente |
| 5. Modo humo de todas las pérdidas | Pendiente |
| 6. Experimentos: pérdidas → etiquetas → hiperparámetros → post-proceso topológico | Pendiente |

## Estructura

```
tfm-filamentos/
├── CONTRACT.md              # reglas comunes a todas las ejecuciones
├── configs/base.yaml        # la baseline; los experimentos heredan de aquí
├── splits/split_v1.json     # split congelado (se genera una vez)
├── scripts/make_split.py
├── src/filseg/
│   ├── contract.py          # constantes del contrato v1
│   ├── official_metric.py   # copia literal del notebook oficial
│   ├── metrics.py           # evaluador: comprobaciones + PQ/SQ/RQ + gráficos
│   ├── gt.py                # GT oficial desde el JSON COCO
│   ├── split.py             # crear / cargar / verificar el split
│   └── runs.py              # salidas de una ejecución y su validación
├── tests/                   # los ejemplos del README de autoevaluación, ejecutables
├── notebooks/
└── memoria/
```

## Uso

### Instalación

```bash
pip install -e ".[dev]"
pytest
```

### En Kaggle

```python
!git clone https://github.com/<usuario>/tfm-filamentos.git
!pip install -q -e tfm-filamentos
import os, subprocess
os.environ["FILSEG_COMMIT"] = subprocess.check_output(
    ["git", "-C", "tfm-filamentos", "rev-parse", "HEAD"], text=True
).strip()
```

Para fijar una versión concreta del código, clona y haz `git checkout <commit>`.

### Generar el split (una sola vez)

```python
from pathlib import Path
DATA = Path("/kaggle/input/competitions/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026")
ANN = next((DATA / "train").glob("*.json"))
!python tfm-filamentos/scripts/make_split.py --annotations "{ANN}" --out split_v1.json
```

Descarga `split_v1.json`, cópialo a `splits/` y haz commit.

**Comprobación:** el split debe reproducir la baseline original. `val_A`
y `val_B` juntos deben sumar **234 pares anotador-imagen de 132 días**, y
`train`, **920 pares de 524 días**.

### Evaluar predicciones

```python
from filseg.gt import load_coco, build_gt_df
from filseg.split import load_split
from filseg.metrics import evaluate, plot_official

split = load_split("splits/split_v1.json", ANN)
gt_B = build_gt_df(load_coco(ANN), split["subsets"]["val_B"])
res = evaluate(gt_B, pred_df)        # pred_df: filament_id, segmentation_rle
print(res.metrics)                   # pq, sq, rq, tp, fp, fn, ...
plot_official(res.overlap_df, "figuras", tag="val_B")
```

## Rúbrica de la competición

El leaderboard (PQ) es solo una parte:

- **Cuantitativo (70 %):** PQ, distribución de Dice, distribución de IoU y
  distribución de relaciones 1:n / n:1.
- **Cualitativo (30 %):** descripción del pipeline, morfología de las
  segmentaciones y calidad del código (modularidad y documentación).

La evaluación depende de que el código fuente sea público.
