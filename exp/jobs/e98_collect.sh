#!/bin/bash
#SBATCH -A p2026_0089_neu
#SBATCH -p cpu
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 03:00:00
#SBATCH -J e98collect
#SBATCH -o /home/qi_zim_neu/olmoearth_inferenceX/slurm/%x-%j.out
# exp98 stage 4 of 4 (exp/jobs/E98_README.md): copy what exp98 reads off the purged scratch (30 days) to the home
# directory, as a compact rslearn dataset that `oe-inferencex from-olmoearth` reads:
#   /home/qi_zim_neu/olmoearth_inferenceX/data/exp98/awf_run/        (data/ is gitignored; home has 7-day snapshots)
#     root_<i>/config.json                   the run's config.json plus the sentinel2_scl layer
#     root_<i>/windows/<group>/<name>/metadata.json
#     root_<i>/windows/<group>/<name>/layers/output/p0_..._p9/geotiff.tif + completed     the 10 softmax bands
#     root_<i>/windows/<group>/<name>/layers/sentinel2_scl[.N]/SCL/geotiff.tif + completed   SCL per item group
# root_<i> is the i-th dataset root of the run, as numbered in prepare's inventory (whose scratch copy holds the real
# directory name, which the partitioner may have named after a grid corner). items.json (WGS84 scene footprints) is
# not copied: the from-olmoearth reader does not read it, and it stays on scratch.
#     records/                               versions, inventories, checks, the e98 configs (not the geometry)
#     MANIFEST.sha256                        sha256 of every file above, verified after the copy
# and a summary to exp/out/exp98/e98_collect.json. Only windows with a completed output are copied.
#   ssh aicr "sbatch --parsable --dependency=afterok:$PRED_JOB" < exp/jobs/e98_collect.sh
# No other job that resets the home checkout (e98.sh, the optional post stage, any eNN.sh) may run beside this one:
# chain them on it.
# E98_KEEP_IMAGERY=1 also copies the 12-band sentinel2 layers (about 40 to 65 GB, see e98_prepare.sh); exp98 does not
# read them, and the quota check below will usually refuse it. E98_RECOLLECT=1 replaces an earlier collection.
# E98_PILOT=1 collects the pilot run (awf_pilot_run, awf_pilot_scl, *_pilot.json) to data/exp98/pilot/awf_run/ and its
# summary to exp/out/exp98/pilot/; it refuses a full run's inventory, and without it a pilot's.
#
# Home quota: 100 GiB. Expected: about 9 GB of probabilities (10 float32 bands of 1024 x 1024 per window, LZW) and
# well under 1 GB of SCL, for ~224 windows. The job refuses before copying if the home directory's use plus this
# would exceed 95 GiB.
set -euo pipefail
# The soft limit on open files is 1,024 on the cpu nodes (hard 131,072); olmoearth_run's worker pools exhaust it
# (job 1243542 failed in 26 s on "Too many open files" while starting its dataset-build pool).
ulimit -n 65536
REPO=/home/qi_zim_neu/olmoearth_inferenceX
cd "$REPO" || exit 1
git fetch -q origin main
if [ -n "${E98_SHA:-}" ]; then
  # A pinned stage checks out its own commit, whatever main has become since the chain was submitted (the chain spans
  # many hours and main moves). The checkout is left detached; a later `git reset --hard origin/main` still works on it.
  git cat-file -e "$E98_SHA^{commit}" 2>/dev/null || git fetch -q origin "$E98_SHA" \
    || { echo "E98_SHA=$E98_SHA is not a commit on origin"; exit 2; }
  git checkout -q --force --detach "$E98_SHA"
  [ "$(git rev-parse HEAD)" = "$E98_SHA" ] || { echo "checkout is $(git rev-parse HEAD), not E98_SHA=$E98_SHA (give the full sha)"; exit 2; }
else
  git reset -q --hard origin/main
fi
SCRATCH=/scratch/qi_zim_neu/olmoearth_inferenceX
DEPLOY=$SCRATCH/deploy
export E98_PILOT=${E98_PILOT:-0}
case "$E98_PILOT" in 0|1) ;; *) echo "E98_PILOT must be 0 or 1, got '$E98_PILOT'"; exit 2;; esac
if [ "$E98_PILOT" = 1 ]; then TAG=awf_pilot SFX=_pilot DEST=$REPO/data/exp98/pilot/awf_run OUT=$REPO/exp/out/exp98/pilot
else TAG=awf SFX= DEST=$REPO/data/exp98/awf_run OUT=$REPO/exp/out/exp98; fi
PY=$DEPLOY/venv/bin/python
export E98_RUN=$DEPLOY/${TAG}_run E98_SCL=$DEPLOY/${TAG}_scl E98_DEPLOY=$DEPLOY E98_DEST=$DEST E98_OUT=$OUT
export E98_CFG=$DEPLOY/${TAG}_config E98_INV=$DEPLOY/prepare_inventory$SFX.json E98_SFX=$SFX
export E98_KEEP_IMAGERY=${E98_KEEP_IMAGERY:-0} E98_HOME_LIMIT=$((95 * 2**30))
export PYTHONUNBUFFERED=1

echo "== $(date -Is) job ${SLURM_JOB_ID:-none} on $(hostname): exp98 collect, commit $(git rev-parse HEAD), pilot $E98_PILOT =="
"$PY" - "$DEPLOY/predict_check$SFX.json" "$E98_INV" <<'EOF'
import json, os, sys
c, inv = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
pilot = os.environ["E98_PILOT"] == "1"
assert inv.get("pilot", False) == pilot and c.get("pilot", False) == pilot, \
    f"inventory pilot {inv.get('pilot', False)}, predict check pilot {c.get('pilot', False)}, this job {pilot}: no mixing"
assert c["ok"], "predict check not ok"
assert "area_complete" in c and "complete" in inv, "records predate the completeness check: rerun prepare and predict"
assert inv["complete"] or inv.get("allow_incomplete"), "prepare found the area incomplete"
EOF
if [ -e "$DEST" ]; then
  if [ "${E98_RECOLLECT:-0}" = 1 ]; then rm -rf "$DEST"; else echo "$DEST exists; E98_RECOLLECT=1 replaces it"; exit 2; fi
fi
rm -rf "$DEST.partial"
mkdir -p "$DEST.partial/records" "$OUT"
# du exits non-zero on an unreadable file; the sum it prints is still what counts against the quota
E98_HOME_USED=$( (du -sb "$HOME" 2>/dev/null || true) | cut -f1 )
[ -n "$E98_HOME_USED" ] || { echo "could not measure $HOME"; exit 1; }
export E98_HOME_USED

"$PY" - <<'EOF'
import glob, hashlib, json, os, re, shutil
run, scl, deploy = os.environ["E98_RUN"], os.environ["E98_SCL"], os.environ["E98_DEPLOY"]
dest = os.environ["E98_DEST"] + ".partial"
keep_img = os.environ["E98_KEEP_IMAGERY"] == "1"
inv = json.load(open(os.environ["E98_INV"]))
sfx = os.environ["E98_SFX"]

plan = []                                   # (source, path relative to dest)
configs = {}
summary = {"job": os.environ.get("SLURM_JOB_ID"), "pilot": bool(inv.get("pilot")), "roots": [], "keep_imagery": keep_img,
           "area_complete": inv["complete"], "allow_incomplete": bool(inv.get("allow_incomplete"))}
for r in inv["roots"]:
    root, side = os.path.join(run, r["root"]), os.path.join(scl, r["root"])
    cfg = json.load(open(os.path.join(root, "config.json")))
    cfg["layers"]["sentinel2_scl"] = json.load(open(os.path.join(side, "config.json")))["layers"]["sentinel2_scl"]
    configs[os.path.join(r["root_id"], "config.json")] = cfg
    n = 0
    grid = {}                               # CRS -> [resolutions, pixel bbox] of the copied windows, for the reader

    for w in sorted(glob.glob(os.path.join(root, "windows", "*", "*"))):
        if not os.path.exists(os.path.join(w, "layers", "output", "completed")):
            continue
        n += 1
        rel = os.path.join(r["root_id"], os.path.relpath(w, root))
        plan.append((os.path.join(w, "metadata.json"), os.path.join(rel, "metadata.json")))
        meta = json.load(open(os.path.join(w, "metadata.json")))
        pr, b = meta["projection"], meta["bounds"]
        g = grid.setdefault(str(pr["crs"]), [set(), list(b)])
        g[0].add((pr["x_resolution"], pr["y_resolution"]))
        g[1] = [min(g[1][0], b[0]), min(g[1][1], b[1]), max(g[1][2], b[2]), max(g[1][3], b[3])]
        trees = [(os.path.join(w, "layers", "output"), os.path.join(rel, "layers", "output"))]
        sw = os.path.join(side, os.path.relpath(w, root))
        for d in sorted(glob.glob(os.path.join(sw, "layers", "sentinel2_scl*"))):
            trees.append((d, os.path.join(rel, "layers", os.path.basename(d))))
        if keep_img:
            for d in sorted(glob.glob(os.path.join(w, "layers", "sentinel2*"))):
                if re.fullmatch(r"sentinel2(\.\d+)?", os.path.basename(d)):
                    trees.append((d, os.path.join(rel, "layers", os.path.basename(d))))
        for src_dir, rel_dir in trees:
            for dp, _, fs in os.walk(src_dir):
                for f in fs:
                    s = os.path.join(dp, f)
                    plan.append((s, os.path.join(rel_dir, os.path.relpath(s, src_dir))))
    # The from-olmoearth reader (oe_inferencex/olmoearth.py at 58d0a34, from-olmoearth branch) holds one CRS's grid in
    # memory and refuses one above MAX_GRID_BYTES = 4 GiB (:55, _check_size :256-274) at 4 x 10 + 4 bytes per pixel for
    # the scores (:371) and 8 + 1 + 4 for the conditions (:779). The same sum here says whether it will read this root.
    rg = []
    for crs, (res, (x0, y0, x1, y1)) in sorted(grid.items()):
        H, W = abs(y1 - y0), abs(x1 - x0)
        rg.append({"crs": crs, "resolutions": len(res), "px": [H, W], "scores_gib": round(44 * H * W / 2**30, 2),
                   "conditions_gib": round(13 * H * W / 2**30, 2),
                   "reader_fits": len(res) == 1 and 44 * H * W <= 4 * 2**30})
    summary["roots"].append({"root_id": r["root_id"], "windows_copied": n, "reader_grids": rg,
                             "reader_reads_whole_root": all(x["reader_fits"] for x in rg)})
need = sum(os.path.getsize(s) for s, _ in plan)
used, limit = int(os.environ["E98_HOME_USED"]), int(os.environ["E98_HOME_LIMIT"])
print(f"{len(plan)} files, {need / 2**30:.2f} GiB to copy; home holds {used / 2**30:.1f} GiB; limit {limit / 2**30:.0f} GiB")
if used + need > limit:
    raise SystemExit(f"refused: {used / 2**30:.1f} + {need / 2**30:.2f} GiB would pass {limit / 2**30:.0f} GiB of the "
                     "100 GiB home quota; keep only the output and SCL (unset E98_KEEP_IMAGERY) or free space")

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
# records: everything small that says how the run was made; the request geometry stays in the pinned clone (its
# sha256 is in awf_config.sha256; a pilot's clipped geometry, in awf_pilot_config.sha256), so no coordinates are copied
for f in ("versions.json", f"scl_check{sfx}.json", f"predict_check{sfx}.json",
          os.path.basename(os.environ["E98_CFG"]) + ".sha256",
          "freeze.txt", "pip_check.txt", "overrides.txt", "stage_spelling.txt"):
    if os.path.exists(os.path.join(deploy, f)):
        put(os.path.join(deploy, f), os.path.join("records", f))
# the inventory as tracked in exp/out (prepare wrote it without the real root and group names), not scratch's
put(os.path.join(os.environ["E98_OUT"], "e98_prepare_inventory.json"), os.path.join("records", "prepare_inventory.json"))
for f in ("model.yaml", "dataset.json", "olmoearth_run.yaml"):
    put(os.path.join(os.environ["E98_CFG"], f), os.path.join("records", os.path.basename(os.environ["E98_CFG"]), f))
put(os.path.join(deploy, "introspect", "olmoearth_run_grep.txt"), os.path.join("records", "olmoearth_run_grep.txt"))
with open(os.path.join(dest, "MANIFEST.sha256"), "w") as f:
    f.write("\n".join(sorted(lines, key=lambda l: l[66:])) + "\n")
summary.update(files=len(lines), bytes=sum(os.path.getsize(os.path.join(dest, l[66:])) for l in lines),
               manifest_sha256=hashlib.sha256(open(os.path.join(dest, "MANIFEST.sha256"), "rb").read()).hexdigest())
json.dump(summary, open(os.path.join(os.environ["E98_OUT"], "e98_collect.json"), "w"), indent=1)
print(json.dumps(summary))
EOF

echo "== $(date -Is) the copy, verified against its manifest =="
( cd "$DEST.partial" && sha256sum -c --quiet MANIFEST.sha256 ) || { echo "manifest check failed; $DEST.partial left for inspection"; exit 1; }
mv "$DEST.partial" "$DEST"
"$PY" - <<'EOF'
import json, os
s = json.load(open(os.path.join(os.environ["E98_OUT"], "e98_collect.json")))
s["dest"] = os.environ["E98_DEST"]
s["manifest_verified"] = True
json.dump(s, open(os.path.join(os.environ["E98_OUT"], "e98_collect.json"), "w"), indent=1)
EOF
echo "== $(date -Is) done: $DEST ($(du -sh "$DEST" | cut -f1)), $OUT/e98_collect.json =="
"$PY" - <<'EOF'
import json, os
s = json.load(open(os.path.join(os.environ["E98_OUT"], "e98_collect.json")))
print("next, on a cpu job with the repository's own environment (the reader is on the from-olmoearth branch):")
for r in s["roots"]:
    path = os.path.join(os.environ["E98_DEST"], r["root_id"])
    if r["reader_reads_whole_root"]:
        print(f"  oe-inferencex from-olmoearth {path} --out <dir> --conditions --inputs sentinel2_scl")
        continue
    big = max(r["reader_grids"], key=lambda g: g["scores_gib"])
    print(f"  {r['root_id']}: from-olmoearth will REFUSE it as one grid ({big['px'][0]} x {big['px'][1]} px in one CRS, "
          f"{big['scores_gib']} GiB of scores against its 4 GiB MAX_GRID_BYTES, or a CRS at two resolutions). Its own "
          "remedy, --window parts each to its own --out, is what exp/jobs/e98_read.sh does; exp98 reads the part "
          "directories as one map.")
print("the whole read, every root in parts, is exp/jobs/e98_read.sh, chained with afterok on this job")
EOF
