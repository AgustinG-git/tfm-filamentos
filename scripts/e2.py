"""Post-proceso por persistencia (E2) sobre ejecuciones ya entrenadas.

Aplica configs/e2_persistencia.yaml (sin reentrenar) a los
entrenamientos que casen con --filtro, en el orden de PRIORIDAD.
Se salta lo ya hecho; si una falla, sigue con las demás.

Uso (con el entorno activado):
    python scripts/e2.py --humo                 # 1 ejecución, 3 intentos
    nohup python scripts/e2.py > "$FILSEG_OUT/logs/e2.log" 2>&1 &
"""

import argparse
import json
import os
import re
import tempfile
import traceback
from pathlib import Path

import yaml

from filseg.postrun import run_post

REPO = Path(__file__).resolve().parents[1]
OUT, DATA = Path(os.environ["FILSEG_OUT"]), Path(os.environ["FILSEG_DATA"])
CFG = REPO / "configs" / "e2_persistencia.yaml"
# referencias y mejores del ajuste primero; después la familia
PRIORIDAD = [
    "_hp-base-t00_", "_hp-topograph-t00_", "_hp-cldice-t00_",
    "_hp-base-t02_", "_hp-topograph-t02_", "_hp-cldice-t11_",
    "_iou_", "_iou-cldice_", "_iou-topograph_", "_cldice-topograph_", "_iou-cldice-topograph_",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--humo", action="store_true", help="1 ejecución y 3 intentos")
    a = ap.parse_args()

    runs = sorted(d for d in (OUT / "runs").iterdir() if (d / "run_config.json").exists())
    cfgs = {d: json.loads((d / "run_config.json").read_text()) for d in runs}
    hechos = {c["fuente"] for c in cfgs.values()
              if c.get("config_post", {}).get("nombre") == "e2-persistencia"}
    fuentes = []
    for patron in PRIORIDAD:
        fuentes += [d for d in runs if patron in d.name and "fuente" not in cfgs[d]
                    and d.name not in hechos and d not in fuentes]
    cfg = CFG
    if a.humo:
        fuentes = fuentes[:1]
        c = yaml.safe_load(CFG.read_text())
        c["post"]["ajuste"]["intentos"] = 3
        c["nombre"] = "e2-humo"
        c["hereda"] = str(CFG.parent / c["hereda"])
        tmp = Path(tempfile.mkdtemp()) / "e2_humo.yaml"
        tmp.write_text(yaml.safe_dump(c))
        cfg = tmp
    print(f"{len(fuentes)} ejecuciones: {[d.name for d in fuentes]}", flush=True)

    for d in fuentes:
        print(f"▶ {d.name}", flush=True)
        try:
            run_post(cfg, source=d, data_dir=DATA, repo_dir=REPO, out_dir=OUT,
                     nota="E2 humo" if a.humo else "E2 persistencia")
            print(f"✔ {d.name}", flush=True)
        except Exception:
            traceback.print_exc()
            print(f"✘ {d.name}", flush=True)


if __name__ == "__main__":
    main()
