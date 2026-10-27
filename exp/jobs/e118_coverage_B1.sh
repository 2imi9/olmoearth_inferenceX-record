#!/bin/bash
#SBATCH -J oeix-exp118-B1
#SBATCH -A p2026_0089_neu
#SBATCH -p cpu
#SBATCH -c 64
#SBATCH --mem=32G
#SBATCH -t 03:00:00
#SBATCH -o /home/qi_zim_neu/olmoearth_inferenceX/slurm/%x-%j.out
SHA=78769cbe0063545f3370e58c935e637172c08585
export UV_CACHE_DIR=/scratch/qi_zim_neu/olmoearth_inferenceX/uv-cache
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
D=/scratch/qi_zim_neu/olmoearth_inferenceX/exp118/$SLURM_JOB_ID
mkdir -p "$D" && cd "$D" || exit 1
git clone -q --branch cluster-check https://github.com/2imi9/olmoearth_inferenceX-record.git rec || exit 1
test "$(cd rec && git rev-parse HEAD)" = "$SHA" || { echo "sha mismatch"; exit 1; }
~/.local/bin/uv venv -q --python 3.12 venv || exit 1
~/.local/bin/uv pip install -q --python venv/bin/python "olmoearth-inferencex==1.18.1" || exit 1
PY="$D/venv/bin/python -I"
t0=$SECONDS
$PY rec/exp/exp118_grade_coverage.py rec B1 5000 64 > exp118_coverage_B1.json 2> exp118_coverage_B1.err
echo "B1 exit $? in $((SECONDS - t0)) s"; tail -3 exp118_coverage_B1.err
sha256sum exp118_coverage_B1.json
$PY -c "import json; o = json.load(open('exp118_coverage_B1.json')); print('quantities', len(o['quantities']), 'errors', o['errors'], 'P2', o['P2_every_valid_interval_holds'], 'P3', o['P3_normal_below_bar_somewhere'], 'least valid', o['valid_coverage_least'], 'least normal', o['normal_coverage_least'])"
echo "B1 DONE"
