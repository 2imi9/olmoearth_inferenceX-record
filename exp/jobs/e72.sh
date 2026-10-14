#!/bin/bash
#SBATCH -A p2026_0089_neu
#SBATCH -p rtx-batch
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH -J e72xcheck
#SBATCH -o /home/qi_zim_neu/olmoearth_inferenceX/slurm/e72-%j.out
set -u
cd /home/qi_zim_neu/olmoearth_inferenceX
export HF_HOME=/scratch/qi_zim_neu/olmoearth_inferenceX/hf/paper_embeddings
export HF_HUB_OFFLINE=1
source .venv/bin/activate
python exp/exp72_agent_crosscheck.py
