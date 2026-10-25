"""Pasa el post-proceso a todas las ejecuciones entrenadas de FILSEG_OUT.

Sin reentrenar (filseg.postrun.run_post). Qué se aplica a cada una:
- a todas:              B2 Optuna (umbral y área mínima ajustados en val_A);
- a las baselines       además B1 umbral, B2 umbral+área y B1 Optuna,
  (BCE+Dice, semilla    para reproducir la exploración del post-proceso.
  42)

Se salta lo ya hecho (misma fuente y misma config), así que se puede
relanzar sin duplicar filas. Si una falla, sigue con las demás.

Uso (con el entorno activado):
    python scripts/postproceso.py
"""

import json
import os
import traceback
from pathlib import Path

from filseg.config import load_config
from filseg.postrun import run_post

REPO = Path(__file__).resolve().parents[1]
OUT, DATA = Path(os.environ["FILSEG_OUT"]), Path(os.environ["FILSEG_DATA"])
SIEMPRE = ["configs/b2_optuna.yaml"]
BASELINES = ("bce_dice_f32", "bce_dice")
BASELINE = ["configs/b1_umbral.yaml", "configs/b2_umbral_area.yaml", "configs/b1_optuna.yaml"]


def leer(run_dir: Path) -> dict:
    return json.loads((run_dir / "run_config.json").read_text())


def main():
    runs = sorted(d for d in (OUT / "runs").iterdir() if (d / "run_config.json").exists())
    configs = [leer(d) for d in runs]
    hechos = {(c["fuente"], c["config_post"]["nombre"]) for c in configs if "fuente" in c}
    entrenados = [
        (d, c) for d, c in zip(runs, configs)
        if "fuente" not in c and "-humo" not in d.name   # solo entrenamientos
    ]
    print(f"{len(entrenados)} ejecuciones entrenadas en {OUT / 'runs'}")

    for d, c in entrenados:
        es_base = (c["config"]["entrenamiento"]["perdida"] in BASELINES
                   and c["config"].get("semilla") == 42)
        for cfg in SIEMPRE + (BASELINE if es_base else []):
            nombre = load_config(REPO / cfg)["nombre"]
            if (c["run_id"], nombre) in hechos:
                print(f"= {c['run_id']} + {nombre}: ya hecho")
                continue
            print(f"▶ {c['run_id']} + {nombre}", flush=True)
            try:
                run_post(cfg, source=d, data_dir=DATA, repo_dir=REPO,
                         out_dir=OUT, nota=c.get("nota", ""))
                print(f"✔ {c['run_id']} + {nombre}", flush=True)
            except Exception:
                traceback.print_exc()
                print(f"✘ {c['run_id']} + {nombre}", flush=True)


if __name__ == "__main__":
    main()
