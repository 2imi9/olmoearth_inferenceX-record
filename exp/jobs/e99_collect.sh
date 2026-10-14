#!/bin/bash
#SBATCH -A p2026_0089_neu
#SBATCH -p cpu
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 03:00:00
#SBATCH -J e99collect
#SBATCH -o /home/qi_zim_neu/olmoearth_inferenceX/slurm/%x-%j.out
# exp99 stage 4 (exp/jobs/E99_README.md): copy what exp99 reads off the purged scratch (30 days) to the home directory,
# as a compact rslearn dataset that `oe-inferencex from-olmoearth` reads, exactly as e98_collect.sh does for exp98:
#   /home/qi_zim_neu/olmoearth_inferenceX/data/exp99/awf2017_run/      (data/ is gitignored; home has 7-day snapshots)
#     root_<i>/config.json, root_<i>/windows/<group>/<name>/{metadata.json, layers/output/..., layers/sentinel2_scl*/...}
#     records/ (versions, inventories, checks, the configs; not the request geometry), MANIFEST.sha256
# and a summary to exp/out/exp99/e99_collect.json. Only windows with a completed output are copied; items.json (the
# scenes' footprints) stays on scratch.
#   ssh aicr "E99_SHA=$SHA sbatch --parsable --dependency=afterok:$PRED_JOB" < exp/jobs/e99_collect.sh
# E99_RECOLLECT=1 replaces an earlier collection. E99_PILOT=1 collects the pilot to data/exp99/pilot/awf2017_run/.
# Home quota: 100 GiB. Expected: about 10 to 12 GB of probabilities and SCL for about 265 to 300 windows; refused
# before copying if the home directory's use plus this would exceed 95 GiB (exp98's collection is already there).
set -euo pipefail
ulimit -n 65536
REPO=/home/qi_zim_neu/olmoearth_inferenceX
SCRATCH=/scratch/qi_zim_neu/olmoearth_inferenceX
CODE=$SCRATCH/e99_code
if [ ! -d "$CODE/.git" ]; then
  git clone -q "$(git -C "$REPO" remote get-url origin)" "$CODE"
fi
cd "$CODE" || exit 1
git fetch -q origin main
if [ -n "${E99_SHA:-}" ]; then
  git cat-file -e "$E99_SHA^{commit}" 2>/dev/null || git fetch -q origin "$E99_SHA" \
    || { echo "E99_SHA=$E99_SHA is not a commit on origin"; exit 2; }
  git checkout -q --force --detach "$E99_SHA"
  [ "$(git rev-parse HEAD)" = "$E99_SHA" ] || { echo "checkout is $(git rev-parse HEAD), not E99_SHA=$E99_SHA (give the full sha)"; exit 2; }
else
  git checkout -q --force --detach origin/main
fi
DEPLOY=$SCRATCH/deploy
export E99_PILOT=${E99_PILOT:-0}
case "$E99_PILOT" in 0|1) ;; *) echo "E99_PILOT must be 0 or 1, got '$E99_PILOT'"; exit 2;; esac
if [ "$E99_PILOT" = 1 ]; then TAG=awf2017_pilot SFX=_e99_pilot DEST=$REPO/data/exp99/pilot/awf2017_run OUT=$REPO/exp/out/exp99/pilot
else TAG=awf2017 SFX=_e99 DEST=$REPO/data/exp99/awf2017_run OUT=$REPO/exp/out/exp99; fi
PY=$DEPLOY/venv/bin/python
export E99_RUN=$DEPLOY/${TAG}_run E99_SCL=$DEPLOY/${TAG}_scl E99_DEPLOY=$DEPLOY E99_DEST=$DEST E99_OUT=$OUT
export E99_CFG=$DEPLOY/${TAG}_config E99_INV=$DEPLOY/prepare_inventory$SFX.json E99_SFX=$SFX
export E99_HOME_LIMIT=$((95 * 2**30))
export PYTHONUNBUFFERED=1

echo "== $(date -Is) job ${SLURM_JOB_ID:-none} on $(hostname): exp99 collect, commit $(git rev-parse HEAD), pilot $E99_PILOT =="
"$PY" - "$DEPLOY/predict_check$SFX.json" "$E99_INV" <<'EOF'
import json, os, sys
c, inv = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
pilot = os.environ["E99_PILOT"] == "1"
assert inv.get("experiment") == c.get("experiment") == "exp99", "not exp99's records"
assert inv.get("pilot", False) == pilot and c.get("pilot", False) == pilot, \
    f"inventory pilot {inv.get('pilot', False)}, predict check pilot {c.get('pilot', False)}, this job {pilot}: no mixing"
assert c["ok"], "predict check not ok"
assert inv["complete"] or inv.get("allow_incomplete"), "prepare found the run incomplete"
EOF
if [ -e "$DEST" ]; then
  if [ "${E99_RECOLLECT:-0}" = 1 ]; then rm -rf "$DEST"; else echo "$DEST exists; E99_RECOLLECT=1 replaces it"; exit 2; fi
fi
rm -rf "$DEST.partial"
mkdir -p "$DEST.partial/records" "$OUT"
E99_HOME_USED=$( (du -sb "$HOME" 2>/dev/null || true) | cut -f1 )
[ -n "$E99_HOME_USED" ] || { echo "could not measure $HOME"; exit 1; }
export E99_HOME_USED

"$PY" - <<'EOF'
import glob, hashlib, json, os, shutil
run, scl, deploy = os.environ["E99_RUN"], os.environ["E99_SCL"], os.environ["E99_DEPLOY"]
dest = os.environ["E99_DEST"] + ".partial"
inv = json.load(open(os.environ["E99_INV"]))
sfx = os.environ["E99_SFX"]
plan, configs = [], {}
summary = {"job": os.environ.get("SLURM_JOB_ID"), "experiment": "exp99", "pilot": bool(inv.get("pilot")), "roots": [],
           "area_complete": inv["complete"], "allow_incomplete": bool(inv.get("allow_incomplete"))}
for r in inv["roots"]:
    root, side = os.path.join(run, r["root"]), os.path.join(scl, r["root"])
    cfg = json.load(open(os.path.join(root, "config.json")))
    cfg["layers"]["sentinel2_scl"] = json.load(open(os.path.join(side, "config.json")))["layers"]["sentinel2_scl"]
    configs[os.path.join(r["root_id"], "config.json")] = cfg
    n = 0
    for w in sorted(glob.glob(os.path.join(root, "windows", "*", "*"))):
        if not os.path.exists(os.path.join(w, "layers", "output", "completed")):
            continue
        n += 1
        rel = os.path.join(r["root_id"], os.path.relpath(w, root))
        plan.append((os.path.join(w, "metadata.json"), os.path.join(rel, "metadata.json")))
        trees = [(os.path.join(w, "layers", "output"), os.path.join(rel, "layers", "output"))]
        sw = os.path.join(side, os.path.relpath(w, root))
        for d in sorted(glob.glob(os.path.join(sw, "layers", "sentinel2_scl*"))):
            trees.append((d, os.path.join(rel, "layers", os.path.basename(d))))
        for src_dir, rel_dir in trees:
            for dp, _, fs in os.walk(src_dir):
                for f in fs:
                    s = os.path.join(dp, f)
                    plan.append((s, os.path.join(rel_dir, os.path.relpath(s, src_dir))))
    summary["roots"].append({"root_id": r["root_id"], "windows_copied": n})
need = sum(os.path.getsize(s) for s, _ in plan)
used, limit = int(os.environ["E99_HOME_USED"]), int(os.environ["E99_HOME_LIMIT"])
print(f"{len(plan)} files, {need / 2**30:.2f} GiB to copy; home holds {used / 2**30:.1f} GiB; limit {limit / 2**30:.0f} GiB")
if used + need > limit:
    raise SystemExit(f"refused: {used / 2**30:.1f} + {need / 2**30:.2f} GiB would pass {limit / 2**30:.0f} GiB of the "
                     "100 GiB home quota; free space first")
lines = []
def put(src, rel):
    dst = os.path.join(dest, rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    h = hashlib.sha256()
    with open(src, "rb") as fi, open(dst, "wb") as fo:
        for chunk in iter(lambda: fi.read(1 << 22), b""):
            h.update(chunk)
            fo.write(chunk)
    shutil.copystat(src, dst)
    lines.append(f"{h.hexdigest()}  {rel}")
for rel, cfg in configs.items():
    tmp = os.path.join(dest, rel)
    os.makedirs(os.path.dirname(tmp), exist_ok=True)
    with open(tmp, "w") as f:
        json.dump(cfg, f, indent=1)
    lines.append(f"{hashlib.sha256(open(tmp, 'rb').read()).hexdigest()}  {rel}")
for src, rel in plan:
    put(src, rel)
# records: the request geometry holds positions and stays on scratch; its sha256 is in the config's .sha256 file
for f in ("versions.json", f"scl_check{sfx}.json", f"predict_check{sfx}.json",
          os.path.basename(os.environ["E99_CFG"]) + ".sha256", "freeze.txt", "pip_check.txt", "overrides.txt",
          "stage_spelling.txt"):
    if os.path.exists(os.path.join(deploy, f)):
        put(os.path.join(deploy, f), os.path.join("records", f))
put(os.path.join(os.environ["E99_OUT"], "e99_prepare_inventory.json"), os.path.join("records", "prepare_inventory.json"))
for f in ("model.yaml", "dataset.json", "olmoearth_run.yaml"):
    put(os.path.join(os.environ["E99_CFG"], f), os.path.join("records", os.path.basename(os.environ["E99_CFG"]), f))
with open(os.path.join(dest, "MANIFEST.sha256"), "w") as f:
    f.write("\n".join(sorted(lines, key=lambda l: l[66:])) + "\n")
summary.update(files=len(lines), bytes=sum(os.path.getsize(os.path.join(dest, l[66:])) for l in lines),
               manifest_sha256=hashlib.sha256(open(os.path.join(dest, "MANIFEST.sha256"), "rb").read()).hexdigest())
json.dump(summary, open(os.path.join(os.environ["E99_OUT"], "e99_collect.json"), "w"), indent=1)
print(json.dumps(summary))
EOF

echo "== $(date -Is) the copy, verified against its manifest =="
( cd "$DEST.partial" && sha256sum -c --quiet MANIFEST.sha256 ) || { echo "manifest check failed; $DEST.partial left for inspection"; exit 1; }
mv "$DEST.partial" "$DEST"
"$PY" - <<'EOF'
import json, os
p = os.path.join(os.environ["E99_OUT"], "e99_collect.json")
s = json.load(open(p))
s["dest"] = os.environ["E99_DEST"]
s["manifest_verified"] = True
json.dump(s, open(p, "w"), indent=1)
EOF
echo "== $(date -Is) done: $DEST ($(du -sh "$DEST" | cut -f1)), $OUT/e99_collect.json; next e99_read.sh =="
