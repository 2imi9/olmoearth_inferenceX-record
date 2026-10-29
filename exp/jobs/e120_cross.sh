#!/bin/bash
#SBATCH -J oeix-exp120-cross
#SBATCH -A p2026_0089_neu
#SBATCH -p cpu
#SBATCH -c 48
#SBATCH --mem=48G
#SBATCH -t 02:00:00
#SBATCH -o /home/qi_zim_neu/olmoearth_inferenceX/slurm/%x-%j.out
# exp120's second route (written from the plan alone) with the same 1.20.3 venv, then the cell-by-cell comparison.
set -u
export PYTHONUNBUFFERED=1
C=/scratch/qi_zim_neu/olmoearth_inferenceX/checks/exp120
cd "$C" || exit 1
export TMPDIR=$C/tmp2; mkdir -p "$TMPDIR" out
PY=$C/venv/bin/python
$PY -I -c "import oe_inferencex as o; print('version', o.__version__)"
ls -la out/
echo "== second route smoke"
$PY -I exp/exp120_crosscheck.py smoke; echo "smoke exit $?"
echo "== second route arm A"; date
$PY -I exp/exp120_crosscheck.py units --units /home/qi_zim_neu/olmoearth_inferenceX/exp/out/exp79_units \
    --exp116 exp116_ab.json --out out/exp120_crosscheck_units.json --workers 48
echo "arm A exit $?"; date
echo "== second route arm B"
$PY -I exp/exp120_crosscheck.py tile --out out/exp120_crosscheck_tile.json --work "$TMPDIR/tile" --workers 17
echo "arm B exit $?"; date
echo "== compare"
$PY -I exp/exp120_compare.py --units out/exp120_units.json --tile out/exp120_tile.json \
    --second-units out/exp120_crosscheck_units.json --second-tile out/exp120_crosscheck_tile.json \
    --out out/exp120_compare.json
echo "compare exit $?"
sha256sum out/*.json exp/*.py
