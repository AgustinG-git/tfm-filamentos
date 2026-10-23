#!/bin/bash
# Ejecuta varias configs una detrás de otra (una sola GPU, sin solapes).
# Antes espera a que termine cualquier entrenamiento en marcha.
#
# Uso, con el entorno activado:
#   nohup bash scripts/cola.sh "servidor" configs/a.yaml configs/b.yaml \
#       > "$FILSEG_OUT/logs/cola.log" 2>&1 &
#
# Log de cada ejecución: $FILSEG_OUT/logs/<fecha>_<config>.log
set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"
NOTA="$1"; shift
mkdir -p "$FILSEG_OUT/logs"

while pgrep -f "entrenar.py" > /dev/null; do sleep 60; done

for cfg in "$@"; do
  log="$FILSEG_OUT/logs/$(date +%Y%m%d-%H%M)_$(basename "$cfg" .yaml).log"
  echo "$(date '+%F %T') ▶ $cfg  ->  $log"
  if python "$REPO/scripts/entrenar.py" "$cfg" --nota "$NOTA" > "$log" 2>&1; then
    echo "$(date '+%F %T') ✔ $cfg"
  else
    echo "$(date '+%F %T') ✘ $cfg (ver log)"
  fi
done
echo "$(date '+%F %T') cola terminada"
