#!/bin/bash
#SBATCH -J oeix-exp118-A
#SBATCH -A p2026_0089_neu
#SBATCH -p cpu
#SBATCH -c 1
#SBATCH --mem=4G
#SBATCH -t 00:30:00
#SBATCH -o /home/qi_zim_neu/olmoearth_inferenceX/slurm/%x-%j.out
SHA=a0a77f41054aa4f75ff2371a6ecfd08dabc40bfc
export UV_CACHE_DIR=/scratch/qi_zim_neu/olmoearth_inferenceX/uv-cache
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
D=/scratch/qi_zim_neu/olmoearth_inferenceX/exp118/$SLURM_JOB_ID
mkdir -p "$D" && cd "$D" || exit 1
git clone -q --branch cluster-check2 https://github.com/2imi9/olmoearth_inferenceX-record.git rec || exit 1
test "$(cd rec && git rev-parse HEAD)" = "$SHA" || { echo "sha mismatch"; exit 1; }
~/.local/bin/uv venv -q --python 3.12 venv || exit 1
~/.local/bin/uv pip install -q --python venv/bin/python numpy || exit 1
t0=$SECONDS
venv/bin/python -I rec/exp/exp118_crosscheck/second_route.py rec > exp118_second_route.json 2> second_route.err
echo "exit $? in $((SECONDS - t0)) s"; tail -25 second_route.err
sha256sum exp118_second_route.json
echo A DONE
