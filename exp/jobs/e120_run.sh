#!/bin/bash
#SBATCH -J oeix-exp120
#SBATCH -A p2026_0089_neu
#SBATCH -p cpu
#SBATCH -c 48
#SBATCH --mem=48G
#SBATCH -t 02:00:00
#SBATCH -o /home/qi_zim_neu/olmoearth_inferenceX/slurm/%x-%j.out
# exp120 against olmoearth-inferencex 1.20.3 from PyPI: a clean install (checked), arm A on exp116's 356 unit maps,
# arm B on the Dynamic World tile, then the summary with P1 to P3.
set -u
export PYTHONUNBUFFERED=1
export UV_CACHE_DIR=/scratch/qi_zim_neu/olmoearth_inferenceX/uv-cache
export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt
C=/scratch/qi_zim_neu/olmoearth_inferenceX/checks/exp120
cd "$C" || exit 1
export TMPDIR=$C/tmp; mkdir -p "$TMPDIR" out
U=$HOME/.local/bin/uv
rm -rf venv && $U venv -q --python 3.12 venv
rc=1
for i in $(seq 1 10); do
  if $U pip install -q --python venv/bin/python --refresh-package olmoearth-inferencex \
       "olmoearth-inferencex[geo]==1.20.3"; then rc=0; break; fi
  echo "install try $i failed (index lag?), retry in 60 s"; sleep 60
done
echo "install exit $rc"; [ "$rc" = 0 ] || exit 1
PY=$C/venv/bin/python
$PY -I -c "import importlib.metadata as m, oe_inferencex as o; print('version', o.__version__, m.version('olmoearth-inferencex')); import sys; sys.exit(o.__version__ != '1.20.3')" || { echo "not 1.20.3"; exit 1; }
echo "== arm A"; date
$PY -I exp/exp120_valid_profiles.py units --units /home/qi_zim_neu/olmoearth_inferenceX/exp/out/exp79_units \
    --exp116 exp116_ab.json --out out/exp120_units.json --workers 48
echo "arm A exit $?"; date
echo "== arm B"
$PY -I exp/exp120_valid_profiles.py tile --out out/exp120_tile.json --work "$TMPDIR/tile" --workers 17
echo "arm B exit $?"; date
echo "== summary"
$PY -I exp/exp120_valid_profiles.py summary --units-out out/exp120_units.json --tile-out out/exp120_tile.json \
    --out out/exp120_summary.json
echo "summary exit $?"
sha256sum out/*.json exp/exp120_valid_profiles.py exp116_ab.json
