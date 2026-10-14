#!/bin/bash
#SBATCH -A p2026_0089_neu
#SBATCH -p cpu
#SBATCH -c 32
#SBATCH --mem=180G
#SBATCH -t 04:00:00
#SBATCH -J e78export
#SBATCH -o /home/qi_zim_neu/olmoearth_inferenceX/slurm/e78-%j.out
set -u
cd /home/qi_zim_neu/olmoearth_inferenceX
git fetch -q origin main; git reset -q --hard origin/main
export PYTHONUNBUFFERED=1
export HF_HOME=/scratch/qi_zim_neu/olmoearth_inferenceX/hf/paper_embeddings
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=32
source .venv/bin/activate
python -c "import torch; print('torch', torch.__version__, 'threads', torch.get_num_threads())"
echo "== smoke =="
python exp/exp78_error_rate_estimation.py --smoke || exit 1
echo "== export (the only stage that needs the embeddings) =="
python exp/exp78_error_rate_estimation.py --stage export
echo "== sizes =="
du -sh exp/out/exp78_units; ls -la exp/out/exp78_units | head -30
