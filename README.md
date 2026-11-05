# tfm-filamentos

Código del Trabajo de Fin de Máster **«Métodos topológicos en segmentación
de imágenes»** (Máster Universitario en Lógica, Computación e Inteligencia
Artificial, Universidad de Sevilla, 2026).

- **Autor:** Agustín Galiana Carballido
- **Tutor:** Manuel Soriano Trigueros
- **Memoria:** [`memoria/TFM_Agustin_Galiana.pdf`](memoria/TFM_Agustin_Galiana.pdf)
- **Versión depositada:** etiqueta [`v1.0-deposito`](../../tree/v1.0-deposito)

## De qué trata

El trabajo estudia qué aporta la topología computacional a la segmentación
de estructuras finas, aplicada a los filamentos solares de las imágenes
H-alfa de GONG (conjunto de datos MAGFiLO, *Solar Filament Segmentation
Challenge 2026* en Kaggle).

La topología se emplea de dos formas:

1. **En el entrenamiento**, con pérdidas topológicas: clDice y
   [Topograph](https://github.com/AlexanderHBerger/Topograph) (Lux et al.,
   ICLR 2025).
2. **En el post-proceso**, decidiendo qué componentes conservar por su
   persistencia en el diagrama $H_0$ en lugar de por su tamaño (E2 y B2+P).

Resultados principales (val B, detalle en el capítulo 6 y el anexo B de la
memoria):

- Sin post-proceso, Topograph da las predicciones más fiables: todas sus
  variantes tienen más precisión y entre un 38 y un 63 % menos de
  predicciones sin filamento que las pérdidas sin él.
- Con el filtro de área (B2), el PQ de las distintas pérdidas apenas se
  diferencia: Topograph y el filtro corrigen el mismo tipo de error.
- El post-proceso por persistencia conserva los filamentos y reduce la
  fragmentación. Combinado con el área (B2+P) en modelos entrenados con
  Topograph, da menos errores de estructura y más precisión que B2.

## Estructura

```
tfm-filamentos/
├── CONTRACT.md              # reglas comunes a todas las ejecuciones
├── configs/                 # un YAML por experimento (heredan de base.yaml)
├── splits/split_v1.json     # partición congelada train / val_A / val_B
├── src/filseg/              # biblioteca
│   ├── contract.py          # decisiones congeladas del protocolo
│   ├── split.py             # partición agrupada por días
│   ├── gt.py                # referencia oficial desde el JSON COCO
│   ├── data.py              # carga de imágenes y etiquetas
│   ├── model.py             # U-Net con codificador ResNet-34
│   ├── losses.py            # BCE, Dice, IoU, clDice y Topograph
│   ├── train.py             # una ejecución completa
│   ├── postproc.py          # B1, B2, E2 y B2+P
│   ├── postrun.py           # ajuste de un post-proceso sin reentrenar
│   ├── official_metric.py   # copia literal de la métrica oficial
│   ├── metrics.py           # evaluador (PQ, SQ, RQ) y comprobaciones
│   ├── compare.py           # conteos, relaciones 1:n / n:1, bootstrap
│   └── runs.py              # identificadores, salidas y registro
├── scripts/                 # experimentos (ver abajo)
├── tests/                   # pruebas automáticas (pytest)
├── registro.csv             # una fila por ejecución, con PQ en val_A y val_B
├── conteos.csv              # conteos por par anotador-imagen
├── barbas_runs.csv          # medidas de barbas por ejecución y post-proceso
└── memoria/                 # fuentes LaTeX y PDF de la memoria
```

## Instalación

```bash
pip install -e ".[train,dev]"
pytest
```

Topograph se usa a través de su implementación oficial, con la extensión
C++ compilada. La ruta al repositorio clonado se indica en
`TOPOGRAPH_PATH`.

Variables de entorno:

| Variable | Contenido |
|---|---|
| `FILSEG_DATA` | carpeta `MAGFiLO_1.0_Kaggle_2026` de la competición |
| `FILSEG_OUT` | carpeta de salidas (`runs/`, `logs/`, registro y conteos) |
| `TOPOGRAPH_PATH` | repositorio oficial de Topograph |

## Reproducir los experimentos

Cada ejecución crea `runs/<run_id>` en `FILSEG_OUT`, con la configuración,
la época elegida, las predicciones y logits de validación y el envío para
Kaggle, y añade una fila a `registro.csv` y sus conteos a `conteos.csv`.
El `run_id` combina fecha, configuración y commit del código.

| Paso | Comando | Memoria |
|---|---|---|
| Partición (una sola vez) | `python scripts/make_split.py --annotations <JSON de train>` | 5.1 |
| Entrenar una pérdida | `python scripts/entrenar.py configs/loss_topograph.yaml` | 6.2 |
| Varias en cola | `bash scripts/cola.sh "nota" configs/a.yaml configs/b.yaml` | 6.2 |
| Post-proceso B1 y B2 | `python scripts/postproceso.py --filtro "_base_"` | 6.1 |
| Post-proceso E2 | `python scripts/e2.py` | 6.3 |
| Post-proceso B2+P | `python scripts/e2.py --config configs/b2p_area_persistencia.yaml` | 6.3 |
| Ajuste de hiperparámetros | `python scripts/ajuste_hp.py --horas 9` | 6.4.2 |
| Barbas | `python scripts/barbas.py gt` y `python scripts/barbas.py runs` | 6.4.1 |
| Bootstrap y tipos de error | `python scripts/resultados.py conteos.csv` | 6.2, B.3 |
| Figuras del capítulo 6 | `python scripts/figuras_cap6.py` | 6.1, 6.2 |

`entrenar.py` acepta `--humo` para una prueba rápida antes de una
ejecución larga.

El entrenamiento en GPU no es determinista: repetir una ejecución
reproduce sus cifras con una diferencia del orden de 0,01–0,02 en PQ.

### Configuraciones

- `base.yaml`: baseline, BCE + Dice, semilla 42 (`base_s43.yaml`, semilla 43).
- `loss_*.yaml`: las diez pérdidas comparadas (IoU, clDice, sus
  combinaciones y cada una con Topograph; `_s43` para la segunda semilla).
- `b1_*.yaml`, `b2_*.yaml`: post-proceso por umbral y por umbral y área.
- `e2_persistencia.yaml`, `b2p_area_persistencia.yaml`: post-procesos por
  persistencia.
- `humo.yaml`: prueba rápida.

## Compilar la memoria

```bash
cd memoria
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Necesita `sello.pdf` en la misma carpeta (incluido).

## Licencia

MIT. Ver [LICENSE](LICENSE).
