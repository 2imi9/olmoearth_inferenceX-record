#!/bin/bash
#SBATCH -A p2026_0089_neu
#SBATCH -p b200-batch,rtx-batch
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=250G
#SBATCH -t 12:00:00
#SBATCH -J e74enc
#SBATCH -o /home/qi_zim_neu/olmoearth_inferenceX/slurm/e74-%j.out
set -u
cd /home/qi_zim_neu/olmoearth_inferenceX
export HF_HOME=/scratch/qi_zim_neu/olmoearth_inferenceX/hf/paper_embeddings
export HF_HUB_OFFLINE=1
source .venv/bin/activate
python exp/exp74_suite_encoders.py
