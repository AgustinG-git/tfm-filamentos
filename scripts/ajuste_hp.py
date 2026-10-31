"""Ajuste de hiperparámetros del entrenamiento con Optuna (noche).

Tres estudios, uno detrás de otro, con el mismo tiempo cada uno:
baseline (BCE+Dice), Topograph (BCE+Dice+Topo) y clDice (BCE+Dice+clDice).

- Objetivo: PQ oficial en val_A (contrato v1: nada se decide en B).
- Cada intento es una ejecución normal: fila en registro.csv con val_B.
- Poda (MedianPruner) con el PQ@512 de val_A de cada época.
- El primer intento de cada estudio es la config actual (referencia).
- Al terminar, B2 Optuna sobre el mejor intento de cada estudio.
- Reanudable: el estado está en $FILSEG_OUT/ajuste_hp.db.

Uso (con el entorno activado):
    nohup python scripts/ajuste_hp.py --horas 9 \
        > "$FILSEG_OUT/logs/ajuste_hp.log" 2>&1 &
    python scripts/ajuste_hp.py --resumen   # ver cómo va
    python scripts/ajuste_hp.py --humo      # prueba antes de la noche
"""

import argparse
import gc
import os
from pathlib import Path

import optuna
import torch
import yaml

from filseg.postrun import run_post
from filseg.train import run

REPO = Path(__file__).resolve().parents[1]
OUT, DATA = Path(os.environ["FILSEG_OUT"]), Path(os.environ["FILSEG_DATA"])
DB = f"sqlite:///{OUT / 'ajuste_hp.db'}"
TMP = REPO / "configs" / "ajuste"   # configs generadas (no se versionan)


def comunes(t):
    return {
        "lr": t.suggest_float("lr", 1e-4, 1e-3, log=True),
        "weight_decay": t.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
    }


def esp_base(t):
    d = t.suggest_float("peso_dice", 0.25, 0.75)
    return {"peso_bce": 1 - d, "peso_dice": d}


def esp_topo(t):
    return {"peso_bce": 0.5, "peso_dice": 0.5, "calentamiento": 5,
            "peso_topo": t.suggest_float("peso_topo", 0.01, 0.16, log=True)}


def esp_cldice(t):
    c = t.suggest_float("peso_cldice", 0.10, 0.40)
    return {"peso_bce": 0.5, "peso_dice": 0.5 - c, "peso_cldice": c, "iters": 10}


# nombre: (config de partida, espacio, intento de referencia)
ESTUDIOS = {
    "base": ("base.yaml", esp_base,
             {"lr": 3e-4, "weight_decay": 1e-4, "peso_dice": 0.5}),
    "topograph": ("loss_topograph.yaml", esp_topo,
                  {"lr": 3e-4, "weight_decay": 1e-4, "peso_topo": 0.04}),
    "cldice": ("loss_cldice.yaml", esp_cldice,
               {"lr": 3e-4, "weight_decay": 1e-4, "peso_cldice": 0.25}),
}


def objetivo(nombre, humo=False):
    padre, espacio, _ = ESTUDIOS[nombre]

    def f(trial):
        ent = comunes(trial)
        ent["perdida_args"] = espacio(trial)
        cfg = {"hereda": f"../{padre}",
               "nombre": f"hp-{nombre}-t{trial.number:02d}",
               "entrenamiento": ent}
        TMP.mkdir(exist_ok=True)
        path = TMP / f"hp_{nombre}_t{trial.number:02d}.yaml"
        path.write_text(yaml.safe_dump(cfg, allow_unicode=True))

        def poda(epoca, pq):
            trial.report(pq, epoca)
            if trial.should_prune():
                raise optuna.TrialPruned(f"época {epoca}, PQ@512 A {pq:.4f}")

        try:
            res = run(path, data_dir=DATA, repo_dir=REPO, out_dir=OUT,
                      nota=f"ajuste hp {nombre} t{trial.number}", al_epoca=poda,
                      humo=humo or None)
        finally:
            gc.collect()
            torch.cuda.empty_cache()   # libera GPU entre intentos
        trial.set_user_attr("run_id", res["run_id"])
        trial.set_user_attr("pq_B", res["val_B"]["pq"])  # solo informativo
        return res["val_A"]["pq"]

    return f


def estudio(nombre, humo=False):
    return optuna.create_study(
        study_name=f"hp-{nombre}", storage=None if humo else DB, direction="maximize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=4),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=10),
    )


def resumen():
    for nombre in ESTUDIOS:
        try:
            s = optuna.load_study(study_name=f"hp-{nombre}", storage=DB)
        except KeyError:
            continue
        df = s.trials_dataframe()
        print(f"\n== {nombre} ==")
        print(df.drop(columns=["datetime_start", "datetime_complete"],
                      errors="ignore").to_string(index=False))


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--horas", type=float, default=9.0, help="total, se reparte")
    p.add_argument("--solo", nargs="*", default=list(ESTUDIOS))
    p.add_argument("--resumen", action="store_true")
    p.add_argument("--humo", action="store_true", help="2 intentos mini por estudio")
    a = p.parse_args()
    if a.resumen:
        return resumen()

    segundos = a.horas * 3600 / len(a.solo)
    for nombre in a.solo:
        s = estudio(nombre, a.humo)
        if not s.trials:  # referencia = config actual
            s.enqueue_trial(ESTUDIOS[nombre][2])
        s.optimize(objetivo(nombre, a.humo), timeout=None if a.humo else segundos,
                   n_trials=2 if a.humo else None,
                   catch=(RuntimeError,))  # p. ej. OOM: sigue
        if a.humo:
            print(f"✔ humo {nombre}: {[t.state.name for t in s.trials]}")
    if a.humo:
        return

    for nombre in a.solo:  # B2 sobre el mejor de cada estudio
        best = estudio(nombre).best_trial
        print(f"▶ {nombre}: mejor t{best.number} PQ A {best.value:.4f}")
        run_post("configs/b2_optuna.yaml", source=OUT / "runs" / best.user_attrs["run_id"],
                 data_dir=DATA, repo_dir=REPO, out_dir=OUT,
                 nota=f"B2 sobre mejor hp {nombre}")
    resumen()


if __name__ == "__main__":
    main()
