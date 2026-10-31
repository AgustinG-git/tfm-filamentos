"""Bootstrap y asignaciones del capítulo 6 a partir de conteos.csv.

Escribe memoria/tablas/bootstrap.tex (lo incluye capitulo6.tex) y
muestra las asignaciones (1:n, n:1, sin contacto) de cada ejecución.

Uso:
    python scripts/resultados.py [ruta/conteos.csv]
"""

import sys
from pathlib import Path

import pandas as pd

from filseg.compare import asignaciones, bootstrap

REPO = Path(__file__).resolve().parents[1]

# nombre: (sin post, B2 Optuna)
RUNS = {
    "Baseline (s42)": ("20261025-1053_base_cc3f8d7", "20261025-1151_b2-optuna_4768e3e"),
    "Baseline (s43)": ("20261025-1108_base-s43_cc3f8d7", "20261025-1207_b2-optuna_4768e3e"),
    "IoU": ("20261023-1126_iou_2a5a7f4", "20261023-1530_b2-optuna_11668be"),
    "clDice": ("20261023-1142_cldice_beeb24f", "20261023-1535_b2-optuna_11668be"),
    "IoU+clDice": ("20261023-1201_iou-cldice_11668be", "20261023-1540_b2-optuna_11668be"),
    "Topograph (s42)": ("20261023-1240_topograph_11668be", "20261023-1550_b2-optuna_11668be"),
    "Topograph (s43)": ("20261025-1123_topograph-s43_4768e3e", "20261025-1212_b2-optuna_4768e3e"),
    "IoU+Topo": ("20261023-1305_iou-topograph_11668be", "20261023-1555_b2-optuna_11668be"),
    "clDice+Topo": ("20261023-1330_cldice-topograph_11668be", "20261023-1559_b2-optuna_11668be"),
    "IoU+clDice+Topo": ("20261023-1358_iou-cldice-topograph_11668be", "20261023-1604_b2-optuna_11668be"),
}
REF = "Baseline (s42)"


def f(x):
    return f"{x:+.3f}".replace(".", "{,}")


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "conteos.csv"
    conteos = pd.read_csv(path)
    nombres = list(RUNS)
    filas = {}
    for k, post in enumerate(["sin", "B2"]):
        ids = [RUNS[n][k] for n in nombres]
        b = bootstrap(conteos, ids, ref=RUNS[REF][k]).set_index("run_id").loc[ids]
        filas[post] = b
        print(f"\n== {post} ==")
        print(b.assign(nombre=nombres).set_index("nombre").to_string())
        print(asignaciones(conteos, ids).assign(nombre=nombres).set_index("nombre")
              .drop(columns="run_id").to_string())

    lineas = []
    for i, n in enumerate(nombres):
        celdas = []
        for post in ["sin", "B2"]:
            r = filas[post].iloc[i]
            if n == REF:
                celdas += ["---", "---"]
            else:
                ic = f"[{f(r.dif_lo)},\\,{f(r.dif_hi)}]"
                celdas += [f(r.dif) + ("$^*$" if r.significativa else ""), ic]
        lineas.append(f"{n} & " + " & ".join(celdas) + r" \\")
    dest = REPO / "memoria" / "tablas"
    dest.mkdir(exist_ok=True)
    (dest / "bootstrap.tex").write_text("\n".join(lineas) + "\n", encoding="utf-8")
    print(f"\n✔ {dest / 'bootstrap.tex'}")


if __name__ == "__main__":
    main()
