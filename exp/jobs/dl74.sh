#!/bin/bash
#SBATCH -A p2026_0089_neu
#SBATCH -p cpu
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH -t 06:00:00
#SBATCH -J dl74
#SBATCH -o /home/qi_zim_neu/olmoearth_inferenceX/slurm/dl74-%j.out
export HF_HOME=/scratch/qi_zim_neu/olmoearth_inferenceX/hf/paper_embeddings
unset HF_HUB_OFFLINE
df -h /scratch/qi_zim_neu | tail -1
/home/qi_zim_neu/olmoearth_inferenceX/.venv/bin/python - <<'PY'
import time
from huggingface_hub import snapshot_download
ENC = ["anysat","clay_large","panopticon","olmoearth_tiny","olmoearth_nano","olmoearth_large","galileo_tiny","galileo_nano","galileo_base","croma_base","croma_large","terramind_base","terramind_large","satlas_base","copernicusfm"]
t=time.time()
p=snapshot_download("allenai/olmoearth-paper-embeddings", repo_type="dataset", cache_dir="/scratch/qi_zim_neu/olmoearth_inferenceX/hf/paper_embeddings",
                    allow_patterns=[f"{e}/*" for e in ENC] + ["eval_settings/*"], max_workers=8)
print("done", p, f"{(time.time()-t)/60:.0f} min")
PY
du -sh /scratch/qi_zim_neu/olmoearth_inferenceX/hf/paper_embeddings
