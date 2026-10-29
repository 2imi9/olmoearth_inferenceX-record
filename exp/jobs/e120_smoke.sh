#!/bin/bash
#SBATCH -J oeix-exp120-smoke
#SBATCH -A p2026_0089_neu
#SBATCH -p cpu
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH -t 00:30:00
#SBATCH -o /home/qi_zim_neu/olmoearth_inferenceX/slurm/%x-%j.out
# exp120: the script's smoke mode, then a timing of arm A on three real unit files at 10 draws per cell.
set -u
export PYTHONUNBUFFERED=1
C=/scratch/qi_zim_neu/olmoearth_inferenceX/checks/exp120
cd "$C" || exit 1
export TMPDIR=$C/tmp; mkdir -p "$TMPDIR"
PY=/scratch/qi_zim_neu/olmoearth_inferenceX/checks/fix-matrix/src/.venv/bin/python
$PY -I exp/exp120_valid_profiles.py smoke
echo "smoke exit $?"
$PY -I - <<'EOF'
import sys, time, json
sys.path.insert(0, "exp")
import exp120_valid_profiles as m
m.D = 10
U = "/home/qi_zim_neu/olmoearth_inferenceX/exp/out/exp79_units"
for n in ("anysat/breizhcrops.npz", "olmoearth_base/sen1floods11.npz", "croma_base/m_eurosat.npz"):
    t0 = time.time()
    try:
        r = m._unit_job((f"{U}/{n}", n))
    except Exception as exc:
        print(n, "ERROR", type(exc).__name__, exc); continue
    print(n, "units", r["n_units"], "e", round(r["error_rate"], 4), "valid cells", len(r["valid"]),
          "profile cells", len(r["profiles"]), "seconds at 10 draws", round(time.time() - t0, 1),
          "refusals", sorted({x for c in r["valid"] + r["profiles"] for x in c["refusals"]})[:3])
EOF
echo "timing exit $?"
