#!/bin/bash
#SBATCH -A p2026_0089_neu
#SBATCH -p cpu
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 00:40:00
#SBATCH -J e77demo
#SBATCH -o /home/qi_zim_neu/olmoearth_inferenceX/slurm/e77demo-%j.out
set -u
cd /home/qi_zim_neu/olmoearth_inferenceX
git fetch -q origin main; git reset -q --hard origin/main
export PYTHONUNBUFFERED=1
source .venv/bin/activate
for H in /scratch/qi_zim_neu/olmoearth_inferenceX/hf /scratch/qi_zim_neu/olmoearth_inferenceX/hf/paper_embeddings; do
  if [ -f "$H/dw_test/dw_test.zip" ]; then export HF_HOME=$H; fi
done
echo "HF_HOME=${HF_HOME:-unset}"; ls -la "${HF_HOME:-/nonexistent}/dw_test/" 2>&1 | head -5
python scripts/make_demo_sample.py
