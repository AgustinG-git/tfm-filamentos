"""Lanza una ejecución del pipeline fuera de Kaggle (servidor del grupo).

Uso (con el entorno activado, ver README, «En el servidor»):
    python scripts/entrenar.py configs/base.yaml --nota "servidor"
    python scripts/entrenar.py configs/base.yaml --humo

Rutas por variables de entorno:
    FILSEG_DATA  carpeta MAGFiLO_1.0_Kaggle_2026
    FILSEG_OUT   salidas: runs/, logs/, registro.csv, conteos.csv
"""

import argparse
import os
from pathlib import Path

from filseg.train import run

REPO = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("config", help="ruta a la config, relativa al repo")
    p.add_argument("--nota", default="", help="va a la columna nota del registro")
    p.add_argument("--humo", action="store_true", help="prueba rápida")
    a = p.parse_args()
    run(a.config, data_dir=os.environ["FILSEG_DATA"], repo_dir=REPO,
        out_dir=os.environ["FILSEG_OUT"], nota=a.nota, humo=a.humo or None)


if __name__ == "__main__":
    main()
