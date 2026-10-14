#!/bin/bash
#SBATCH -A p2026_0089_neu
#SBATCH -p cpu
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 03:00:00
#SBATCH -J pullvllm
#SBATCH -o /home/qi_zim_neu/olmoearth_inferenceX/slurm/pullvllm-%j.out
set -u
B=/scratch/qi_zim_neu/olmoearth_inferenceX
export APPTAINER_CACHEDIR=$B/apptainer_cache APPTAINER_TMPDIR=$B/apptainer_tmp
mkdir -p $APPTAINER_CACHEDIR $APPTAINER_TMPDIR
apptainer --version
SECONDS=0
apptainer pull --force $B/vllm-openai-nightly.sif docker://vllm/vllm-openai:nightly 2>&1 | grep -vE "^\s*$|Copying blob|Copying config|Writing manifest" | tail -15
echo "pull took $((SECONDS/60)) min"; ls -la $B/vllm-openai-nightly.sif
echo "=== what is inside: vllm version, nvcc, python ==="
apptainer exec $B/vllm-openai-nightly.sif bash -c 'python3 -c "import vllm; print(\"vllm\", vllm.__version__)"; nvcc --version | tail -1; python3 --version; vllm --help 2>/dev/null | head -2'
