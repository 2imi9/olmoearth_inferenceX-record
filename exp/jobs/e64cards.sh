#!/bin/bash
#SBATCH -A p2026_0089_neu
#SBATCH -p cpu
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 01:00:00
#SBATCH -J e64cards
#SBATCH -o /home/qi_zim_neu/olmoearth_inferenceX/slurm/e64cards-%j.out
set -u
cd /home/qi_zim_neu/olmoearth_inferenceX
source .venv/bin/activate
python exp/exp64_agent_benchmark.py --stage cards --sources geoid,sen1
ls /scratch/qi_zim_neu/olmoearth_inferenceX/exp64_cards/v1 | wc -l
