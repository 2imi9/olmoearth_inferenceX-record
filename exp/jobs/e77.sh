#!/bin/bash
#SBATCH -A p2026_0089_neu
#SBATCH -p cpu
#SBATCH -c 32
#SBATCH --mem=180G
#SBATCH -t 05:00:00
#SBATCH -J e77scene
#SBATCH -o /home/qi_zim_neu/olmoearth_inferenceX/slurm/e77-%j.out
set -u
cd /home/qi_zim_neu/olmoearth_inferenceX
git fetch -q origin main; git reset -q --hard origin/main
export PYTHONUNBUFFERED=1
export HF_HOME=/scratch/qi_zim_neu/olmoearth_inferenceX/hf/paper_embeddings
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=32
source .venv/bin/activate
python -c "import torch; print(\"torch\", torch.__version__, \"threads\", torch.get_num_threads())"
echo "== smoke =="
python exp/exp77_scene_contamination.py --smoke || exit 1
echo "== run =="
python exp/exp77_scene_contamination.py
