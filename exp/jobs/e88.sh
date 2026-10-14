#!/bin/bash
#SBATCH -A p2026_0089_neu
#SBATCH -p cpu
#SBATCH -c 32
#SBATCH --mem=180G
#SBATCH -t 04:00:00
#SBATCH -J e88modality
#SBATCH -o /home/qi_zim_neu/olmoearth_inferenceX/slurm/e88-%j.out
# exp88 (docs/plan/missing_modality.md), two modes:
#   sbatch exp/jobs/e88.sh align   identifiers, shapes and split sizes only; allowed before the plan is frozen
#   sbatch exp/jobs/e88.sh run     the full run; the script refuses it until the plan page says frozen
# CPU only, as e78.sh: exp78's export fitted the same PASTIS S1+S2 probe on this partition in about 80 s per task.
# Both smokes run first; --smoke-torch runs the alignment check and the full run end to end on a synthetic Hub
# cache read offline, so a loader or environment fault stops the job before any real file is opened.
set -u
MODE=${1:-align}
cd /home/qi_zim_neu/olmoearth_inferenceX
git fetch -q origin main; git reset -q --hard origin/main
export PYTHONUNBUFFERED=1
export HF_HOME=/scratch/qi_zim_neu/olmoearth_inferenceX/hf/paper_embeddings
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=32
source .venv/bin/activate
python -c "import torch; print('torch', torch.__version__, 'threads', torch.get_num_threads())"
git log -1 --format='commit %H %s'
echo "== smoke =="
python exp/exp88_missing_modality.py --smoke || exit 1
python exp/exp88_missing_modality.py --smoke-torch || exit 1
case "$MODE" in
  align)
    echo "== alignment check (no probe, prediction, error or margin) =="
    python exp/exp88_missing_modality.py --check-alignment ;;
  run)
    echo "== full run =="
    python exp/exp88_missing_modality.py
    echo "== sizes =="
    ls -la exp/out/exp88_summary.json exp/out/exp88_units.npz ;;
  *)
    echo "unknown mode '$MODE': align or run"; exit 2 ;;
esac
