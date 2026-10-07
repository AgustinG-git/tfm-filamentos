# Contrato v1

Reglas que cumplen **todas** las ejecuciones del TFM. Gracias a ellas,
cualquier diferencia entre dos resultados se debe a la configuración que
cambia, no al protocolo.

Los valores están en `src/filseg/contract.py`. **No se editan:** si algo
tiene que cambiar, se crea un contrato v2 y se vuelven a ejecutar las
versiones que se quieran comparar.

---

## 1. Unidad de evaluación y GT

- La unidad es el **par anotador-imagen**, igual que en la competición.
  En el JSON, cada entrada de `images` es un par, con
  `id = <anotador>-<imagen>`.
- El GT de evaluación son **las anotaciones originales, una por una**,
  en formato oficial (`filament_id = <anotador>-<imagen>_<n>`), generadas
  por `filseg.gt.build_gt_df`.
- Este GT **no cambia** con la estrategia de etiquetas de entrenamiento.
  Entrenar con "media" o "mayoría" cambia lo que aprende el modelo, no
  el examen.
- `build_meta` rechaza ids que el evaluador oficial parsearía mal (sin
  `-`, con `_`, o que no coinciden con el fichero) e imágenes que no
  midan 2048 × 2048.

## 2. Split

- Fichero único: `splits/split_v1.json`. Se genera **una vez** con
  `scripts/make_split.py`, se versiona en git y nadie lo recalcula.
- Tres subconjuntos, agrupados por **fecha** (`stem[:8]`): `train`,
  `val_A` y `val_B`. Ninguna imagen ni ningún día aparece en dos.
- Reproduce el split de la baseline original:
  `GroupShuffleSplit(0.2, seed 42)` para train/val y
  `GroupShuffleSplit(0.5, seed 42)` para A/B.
- El split guarda el SHA-256 del JSON de anotaciones; `load_split`
  rechaza un JSON distinto.

## 3. Selección y ajuste: en A. Reporte: en B

| Decisión | Dónde | Cómo |
|---|---|---|
| Época del checkpoint | `val_A` | PQ global a 512 con umbral 0,5 y área mínima 0 (post-proceso estricto, igual para todas las versiones) |
| Parámetros del post-proceso | `val_A` | Lo que defina la configuración (p. ej. grid de umbral y área) |
| Cifra que se reporta | `val_B` | Evaluador oficial a resolución original |
| Leaderboard de Kaggle | test | Solo confirmación puntual |

- Ninguna decisión usa `val_B`. Explorar visualmente casos, galerías o
  diagramas de persistencia también se hace **solo en A**.
- La submission usa el checkpoint elegido en A. No hay reentreno con
  todo el train.

## 4. Métrica

- **Definición de referencia:** `filseg/official_metric.py`, copia
  literal del notebook oficial de autoevaluación. No se modifica.
- **PQ global:** TP si IoU > 0,5 (estricto), sin clases, agregado sobre
  todos los pares anotador-imagen.
- **Rúbrica cuantitativa:** además del PQ, se generan siempre los tres
  gráficos oficiales (distribución de IoU, de Dice y relaciones 1:n /
  n:1), con las funciones oficiales y sus definiciones (IoU > 0).
- **Comprobaciones previas** (`filseg.metrics.evaluate`):
  - ids de GT y de predicción con el formato que espera el oficial;
  - **predicciones sin solapes entre sí**: el oficial contaría cada copia
    como un TP adicional;
  - el PQ propio coincide con `get_pq_score` (si no, error).
- Métricas adicionales (SQ, RQ, recuentos, definiciones propias de 1:n y
  n:1, bootstrap) son complementos y nunca sustituyen a las oficiales.

## 5. Salidas de una ejecución

Carpeta `runs/<run_id>/`, con `run_id = <AAAAMMDD-HHMM>_<config>_<commit7>`:

| Fichero | Contenido |
|---|---|
| `run_config.json` | Configuración resuelta, commit, versión y SHA del split, época elegida, post-proceso final y PQ oficial en `val_A` |
| `val_predictions.csv` | Instancias de **todas** las imágenes de val (A y B), una fila por instancia: `filament_id = <imagen>_<n>`, RLE oficial a 2048 |
| `val_logits.npz` | `stems` y `logits` (float16, N × 512 × 512) de todas las imágenes de val |
| `history.csv` | Métricas por época |
| `checkpoint.pth` | Pesos del checkpoint elegido |
| `submission.csv` | Opcional: predicciones de test |

`filseg.runs.validate_run` comprueba todo lo anterior. Una ejecución que
no lo supere no entra en ninguna comparación.

- El evaluador recalcula el PQ de `val_A` desde `val_predictions.csv` y
  debe coincidir con `metrics_val_A` de `run_config.json`.
- Las predicciones son **por imagen**; el evaluador las compara con cada
  anotación de esa imagen.
- Una ejecución sin commit conocido no es trazable y no es válida. En
  Kaggle, sin carpeta `.git`, se pasa con la variable `FILSEG_COMMIT`.

## 6. Configuración

- Cada etapa (pre, entrenamiento, post) se elige por nombre en un YAML de
  `configs/`. Los experimentos heredan de `base.yaml` y cambian solo lo
  que estudian.
- Lo que fija este contrato no aparece en las configuraciones.
- Una combinación inválida (p. ej. etiquetas blandas con una pérdida que
  exige etiquetas binarias) falla al arrancar, no a mitad de
  entrenamiento.
