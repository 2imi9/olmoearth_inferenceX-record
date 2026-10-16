#!/bin/bash
# exp111 (docs/plan/blind_transfer.md; the page is the specification): the blind transfer test. Ai2's FT-AWF v1
# checkpoint and the record's v1.2 fine-tune predict through rslearn 0.1.17 DIRECTLY (dataset prepare / ingest /
# materialize, model predict; no olmoearth_run, no olmoearth-runner: every rslearn mode refuses if either imports) on
# 124-px windows that sit on Ai2's deployment lattice, at TimeSync 2017 plots of Kenya and Tanzania no model has seen.
# Helpers: exp/exp111_blind_transfer.py (selection, windows, read, gate, units) and exp/exp111_grade.py (P1-P3), both in
# the record's uv environment (`uv run --frozen`), and exp/exp111_jobs.py (the rslearn side; exp102's venv). Configs:
# exp/jobs/e111_config/dataset.json (exp98's with the time order pinned), exp/jobs/e98_config/scl_dataset.json,
# exp/jobs/e102_config/model.yaml (one file, MODEL_ID).
#
# Phase 1 (allowed before freezing; one chain, exp/jobs/e111_submit.sh phase1 [--from MODE]):
#   E111_MODE=env        CPU, 4 CPUs, 32 GB, 1 h. /home's size and file count (one find, refused when unknown, above
#                        90 GiB or above 450,000 files: the limits are 100 GiB and 500,000 files; the chain's only
#                        refusal on /home, since no later job writes there beyond its Slurm log) and our scratch tree;
#                        the venv (rslearn 0.1.17, olmoearth-pretrain-minimal 0.1.2, torch 2.14.1, lightning 2.6.6,
#                        jsonargparse 4.52.0; E111_BUILD_VENV=1 builds $SCRATCH/e111/venv if exp102's is gone: PyPI
#                        downloads, the owner's call), no runner, both checkpoints' sha256, both loaded STRICTLY into the
#                        module built from model.yaml on CPU (offline base weights); the record's environment (the first
#                        `uv run --frozen` in a fresh clone installs it from uv.lock); the inputs the chain needs: the
#                        TimeSync file, Natural Earth and Ai2's geometry (E111_FETCH=1 fetches a missing one and checks
#                        its pin; otherwise refused), exp89's 1,459 label windows; and the purge-bound ones it would like
#                        (exp98's and exp99's deployment datasets: A' and item identity are skipped without them).
#   E111_MODE=select     CPU, 4 CPUs, 16 GB, 30 min. exp99's 309 (refused unless 309 / 219 / 90 / 47), the pool
#                        (refused unless 3,691 in the page's six cells), the draw (seed 111, the sizes in
#                        exp111_blind_transfer.py's SIZES: part of the commit, like the 720 floor, which is 80% of them;
#                        no option changes them), the windows of gate23, gate17 and blind17 (caps; the drawn plots in a
#                        gate17 cell counted against exp99's 2017 deployment windows when they are on scratch and
#                        readable, else against the gate17 cells, a lower bound; the record says which and why),
#                        rslearn 0.1.17's check of each root, the deployment windows' time ranges against the literal
#                        (windows a partial purge left unreadable counted, the rest checked). A rerun keeps an existing
#                        draw and roots that equal this run's (the imagery stays) and refuses different ones unless
#                        E111_REDO names select, which is refused once the page is frozen. Ids and positions: $PRIV
#                        only.
#   E111_MODE=prepare    CPU, 32 CPUs, 64 GB, 8 h. Downloads Sentinel-2 L2A (and SCL) from Planetary Computer, no key.
#                        A two-window smoke root first (twelve groups, newest first, plausible reflectance), then each
#                        root (E111_BLIND_PREP=0 leaves blind17 for after freezing): prepare, ingest (a no-op) and
#                        materialize, three passes, each under `timeout 3h` (rslearn skips what is done, so a rerun
#                        resumes); a failed share above 2% after pass 1 stops the job (a systematic failure), and after
#                        pass 3 any failure beyond MAX_FAILED_SHARE (the owner's allowance, fixed in this file at the
#                        phase-1 commit; proposed 0, at most 0.005; prepare's and materialize's failures summed). Then,
#                        in this order: the SCL sidecar (only the windows with all twelve groups); its materialize
#                        (--ignore-errors), whose failures are added to pass 3's against the same allowance; precheck
#                        (partial windows and windows whose SCL is incomplete refused, or within the allowance
#                        quarantined into this job's own directory, so that they count as no input); the SCL check
#                        (the windows with input only); the input inventory by root and, for blind17, by country x label
#                        group (counts; a window quarantined earlier and fetched again is not counted as quarantined).
#                        /home's usage recorded at the end, a note only. A watchdog stops the job above E111_MAX_GB (25)
#                        under $PRIV or if a window count changes.
#   E111_MODE=gate       GPU, b200-devel only (the blind predict must run on the same device), 1 GPU, 8 CPUs, 64 GB,
#                        2 h 30 (GATE_LIMIT_S). First removes the records of any earlier gate (the device record, the
#                        four output checks, e111_gate.json). `rslearn model predict` for v1 and v1.2 on gate23 and
#                        gate17 (layers output_v1, output_v12), each checked (the record names this job); only when all
#                        four pass, the GPU's name and this job's commit are recorded (e111_device_gate.json: the gate's
#                        predict_commit, and its job). Then, report-
#                        only and never failing the job, diagnostic A' (E111_APRIME=1, default, when exp98's dataset is
#                        on scratch): v1 on exp98's own deployment windows that hold the 259 points, layer
#                        output_e111_v1, under a timeout that leaves 15 minutes of the job, read into $PRIV; the layer is
#                        removed by an EXIT trap here, again in gatecheck and in pack1.
#   E111_MODE=gatecheck  CPU, 8 CPUs, 32 GB, 1 h. Refused unless the device record and the four output checks are one
#                        gate job's, all passing (exp111_jobs.py gate-records), and refused if a GUARDED file changed
#                        between that job's commit and this one's (a --from gatecheck resubmission at a later commit:
#                        rerun --from gate). Read gate23;
#                        item identity; gate A; only if A passed: read gate17 and gate B. Writes $OUT/e111_gate.json
#                        (the gate's GPU, its predict_commit and this commit); exits 4 when the gate fails (pack runs).
#   E111_MODE=pack1      CPU, 1 CPU, 4 GB, 15 min, Slurm mail at its end. Removes the A' layer if it is still there;
#                        $OUT/fetch/e111_phase1.tar.gz and .sha256: records, the gate file, this chain's masked logs
#                        (E111_JOBS lists the job ids). Counts only: exp111_jobs.py pack-check, the function every
#                        record writer uses, refuses a JSON key naming an id or a position.
# Phase 2 (after the owner freezes the page with the draw's sha256; exp/jobs/e111_submit.sh phase2 [--from MODE]).
# Every phase-2 mode refuses unless the page is frozen at E111_SHA, e111_gate.json passed, both of the gate's commits
# (predict_commit, the gate job's; commit, gatecheck's) are ancestors of E111_SHA, none of GUARDED changed since either,
# and draw.json is the page's draw:
#   E111_MODE=predict    GPU, b200-devel, 1 h. Refused on another GPU than the gate's. An earlier predict's output
#                        checks removed first; v1 and v1.2 on blind17, each checked (prepared in phase 1 unless
#                        E111_BLIND_PREP=0: then prepare first).
#   E111_MODE=read       CPU, 8 CPUs, 32 GB, 1 h. The blind read of both models into $PRIV.
#   E111_MODE=units      CPU, 4 CPUs, 16 GB, 30 min. exp111_units_v1.npz and _v12.npz (exp99's schema, shuffled rows)
#                        and their record into $OUT/units/.
#   E111_MODE=grade      CPU, 4 CPUs, 16 GB, 30 min. exp/exp111_grade.py --smoke, then the grade: $OUT/grade/
#                        exp111_summary.json (aggregates).
#   E111_MODE=pack2      CPU, Slurm mail at its end. $OUT/fetch/e111_phase2.tar.gz and .sha256 (records, units, the
#                        summary; no id, no position).
#
# Every mode needs E111_SHA (full, pushed to the record repo): its own clone $SCRATCH/e111_code of
# github.com/2imi9/olmoearth_inferenceX-record, checked out detached; never a home checkout. Writes: $PRIV
# ($SCRATCH/e111_private, mode 700: the draw, the units maps, the three roots and their SCL sidecars, the reads,
# rslearn's logs and temp files), $OUT ($SCRATCH/e111_out: records, masked logs, trainer dirs, the fetch tarballs, the
# units, the grade), the clone's .venv (uv), $SCRATCH/e111_data (fetched public inputs, only with E111_FETCH=1),
# $SCRATCH/hf (checkpoints, only with E111_HF_OFFLINE=0), the Slurm log in /home/qi_zim_neu/olmoearth_inferenceX/slurm
# (KB). A' writes the layer output_e111_v1 into exp98's deployment dataset on scratch (removed as above). Nothing echoed
# holds a plot id, a window bound, a coordinate, an item or tile name.
#
# Expected (estimates, not measured): env 10 to 40 min; select 5 to 15 min; prepare 1.5 to 5 h for 1,468 windows
# (exp104: 696 windows x 8 groups in about 40 min at 32 CPUs); gate 30 to 90 min (A' up to 1 h of it); gatecheck 10 to
# 30 min; predict about 15 min; read 10 to 30 min; units and grade a few minutes. Scratch: about 9 GB in $PRIV
# (watchdog at 25 GB), A' up to 11 GB transient; /home: Slurm logs only. Network: Planetary Computer, tens of GB read,
# not stored. Purge: scratch after 30 days; phase 2 must run within 30 days of prepare. Maintenance: third Tuesday,
# 05:00-17:00 ET (20 Oct, 17 Nov).
#
# Reruns: E111_REDO is a comma list of the modes allowed to replace their own earlier output, and only theirs:
# select (a different draw, and the roots: their imagery is fetched again), gate (the gate roots' output layers),
# gatecheck (the gate file, the gate reads), predict (blind17's output layers), read, units, grade. The submit script
# adds gate,gatecheck on every phase-1 submission. Options: E111_BLIND_PREP (1), E111_APRIME (1), E111_MAX_GB (25),
# E111_FETCH (0), E111_HF_OFFLINE (1), E111_BUILD_VENV (0), E111_VENV (exp102's), E111_WORKERS (8, predict's data
# loader), E111_JOBS (pack: job ids). Not options, part of the commit the gate passes with: MAX_FAILED_SHARE (below)
# and the draw's sizes (exp111_blind_transfer.py SIZES; E111_SIZES is refused).
#SBATCH -A p2026_0089_neu
#SBATCH -J e111
#SBATCH -o /home/qi_zim_neu/olmoearth_inferenceX/slurm/%x-%j.out
set -euo pipefail
umask 077
# the soft limit on open files is 1,024 on the cpu nodes (hard 131,072); worker pools exhaust it (job 1243542)
ulimit -n 65536
T_START=$(date +%s)
MODE=${E111_MODE:-}
case "$MODE" in env|select|prepare|gate|gatecheck|pack1|predict|read|units|grade|pack2) ;;
  *) echo "E111_MODE must be env, select, prepare, gate, gatecheck, pack1, predict, read, units, grade or pack2 (got '$MODE')"; exit 2 ;; esac
[[ "${E111_SHA:-}" =~ ^[0-9a-f]{40}$ ]] || { echo "E111_SHA must be the full 40-character commit (got '${E111_SHA:-}')"; exit 2; }
[ -z "${E111_DROP_PARTIAL:-}" ] || { echo "E111_DROP_PARTIAL is gone: MAX_FAILED_SHARE in this file is the failure allowance"; exit 2; }
[ -z "${E111_SIZES:-}" ] || { echo "E111_SIZES is gone: the draw's sizes are SIZES in exp/exp111_blind_transfer.py, part of the commit (the 720 floor is 80% of them)"; exit 2; }
for v in E111_FETCH E111_HF_OFFLINE E111_BUILD_VENV E111_APRIME E111_BLIND_PREP; do
  case "${!v:-}" in ""|0|1) ;; *) echo "$v must be 0 or 1 (got '${!v}')"; exit 2 ;; esac
done
REDO=${E111_REDO:-}
for r in ${REDO//,/ }; do
  case "$r" in select|gate|gatecheck|predict|read|units|grade) ;;
    *) echo "E111_REDO is a comma list of select, gate, gatecheck, predict, read, units, grade (got '$REDO')"; exit 2 ;; esac
done
redo_for() { [[ ",$REDO," == *",$1,"* ]]; }
MAX_GB=${E111_MAX_GB:-25}
[[ "$MAX_GB" =~ ^[0-9]+$ ]] || { echo "E111_MAX_GB must be an integer"; exit 2; }
WORKERS=${E111_WORKERS:-8}
[[ "$WORKERS" =~ ^[0-9]+$ ]] || { echo "E111_WORKERS must be an integer"; exit 2; }
GATE_LIMIT_S=9000                                    # the gate job's -t 02:30:00 in e111_submit.sh (the smoke compares)
# The failure allowance after prepare's third pass (PROPOSED on the page: 0; the owner may set at most 0.005 before the
# phase-1 commit; a later change is a change to this guarded file, so the gate is rerun): the share of a root's windows
# that may still fail (prepare's and materialize's failures summed) or be left partial; those windows' input is
# quarantined and they count as no input, by cell, in the inventory.
MAX_FAILED_SHARE=0
[[ "$MAX_FAILED_SHARE" =~ ^0(\.[0-9]+)?$ ]] && awk -v s="$MAX_FAILED_SHARE" 'BEGIN { exit !(s <= 0.005) }' \
  || { echo "MAX_FAILED_SHARE must be between 0 and 0.005 (got $MAX_FAILED_SHARE)"; exit 2; }
HOMEDIR=/home/qi_zim_neu
HOME_REPO=$HOMEDIR/olmoearth_inferenceX
SCRATCH=/scratch/qi_zim_neu/olmoearth_inferenceX
RECORD_URL=https://github.com/2imi9/olmoearth_inferenceX-record.git
CODE=$SCRATCH/e111_code
PRIV=$SCRATCH/e111_private
OUT=$SCRATCH/e111_out
REC=$OUT/records
DEPLOY=$SCRATCH/deploy
VENV=${E111_VENV:-$SCRATCH/e102/venv}
VENV_NEW=$SCRATCH/e111/venv
BREADTH=$HOME_REPO/data/breadth                      # exp99's fetched and verified copies (home: not purged)
BREADTH_FETCH=$SCRATCH/e111_data/breadth             # fetched by env with E111_FETCH=1
AI2_GEOMETRY=$DEPLOY/olmoearth_projects/olmoearth_run_data/awf/prediction_request_geometry.geojson
AI2_GEOMETRY_FETCH=$SCRATCH/e111_data/prediction_request_geometry.geojson
AI2_GEOMETRY_URL=https://raw.githubusercontent.com/allenai/olmoearth_projects/f3c9b0c89c7670b3525647dda4c14b76245d682d/olmoearth_run_data/awf/prediction_request_geometry.geojson
AI2_GEOMETRY_SHA256=fee3ce01008d26c6f3c239e68be70851035eb157d5da70bd72cb46dbc8623784
LABELS=$SCRATCH/data/exp89/awf/dataset/windows
INV23=$DEPLOY/prepare_inventory.json; RUN23=$DEPLOY/awf_run
INV17=$DEPLOY/prepare_inventory_e99.json; RUN17=$DEPLOY/awf2017_run
V1_REPO=allenai/OlmoEarth-v1-FT-AWF-Base
V1_REV=a347b1546ab881c92aa125400ed5acc8126394ca
V1_SHA256=1036b76bf7dfde819479dfa02507634f59a5637f14d4071bc3bf18f4d3d4bab2
V12_SCRATCH=$SCRATCH/e102/runs/v1_2_s0/epoch=60-step=17019.ckpt
V12_REPO=2imi9/OlmoEarth-v1_2-FT-AWF-Base
V12_REV=7a1db147cc68d4bde8dff3739d85dc793723619e
V12_SHA256=5ce8882718f8b3c5f772fd310fb49affe0aadffaf88cec6474af1442b7324816
declare -A MODEL_IDS=([v1]=OLMOEARTH_V1_BASE [v12]=OLMOEARTH_V1_2_BASE)
# what the phase-2 modes refuse to run over if it changed since the gate's commit: the exp111 code and configs, the
# record modules they import, the grader, and the record environment's pins
# (the smoke derives the record modules from the exp111 modules' import graph and refuses a GUARDED that lacks one)
GUARDED=(exp/exp111_blind_transfer.py exp/exp111_jobs.py exp/exp111_grade.py exp/jobs/e111.sh exp/jobs/e111_submit.sh
  exp/jobs/e111_config exp/jobs/e102_config/model.yaml exp/jobs/e98_config exp/exp98_awf_deployment.py
  exp/exp98_burned.py exp/exp99_transfer.py exp/timesync_awf_crosswalk.py exp/exp89_finetuned_checkpoints.py
  exp/exp78_error_rate_estimation.py pyproject.toml uv.lock)
JOB=${SLURM_JOB_ID:-none}
NCPU=${SLURM_CPUS_PER_TASK:-4}
export HF_HOME=$SCRATCH/hf UV_CACHE_DIR=$SCRATCH/uv-cache PYTHONUNBUFFERED=1 WANDB_MODE=disabled
export HF_HUB_OFFLINE=${E111_HF_OFFLINE:-1}
export PATH="$HOME/.local/bin:$PATH"
mkdir -p "$PRIV/logs" "$PRIV/tmp"
chmod 700 "$PRIV"
# rslearn starts its pools with forkserver, whose socket lives under TMPDIR (pymp-XXXXXXXX/listener-XXXXXXXX: 32 more
# characters), and a Unix socket path holds at most 107. $PRIV/tmp/<tag>-<job> came to about 106 and prepare 1364257
# died on it (OSError: AF_UNIX path too long), so every TMPDIR is a short directory under TSHORT, inside $PRIV.
TSHORT=$PRIV/t
mkdir -p "$TSHORT"
TMPN=0
mkdir -p "$REC" "$OUT/logs" "$OUT/fetch"

if [ ! -d "$CODE/.git" ]; then
  mkdir -p "$(dirname "$CODE")"
  git clone -q "$RECORD_URL" "$CODE"
fi
cd "$CODE"
git fetch -q origin main
git cat-file -e "$E111_SHA^{commit}" 2>/dev/null || git fetch -q origin "$E111_SHA" \
  || { echo "E111_SHA=$E111_SHA is not a commit on origin"; exit 2; }
git checkout -q --force --detach "$E111_SHA"
[ "$(git rev-parse HEAD)" = "$E111_SHA" ] || { echo "checkout is $(git rev-parse HEAD), not E111_SHA"; exit 2; }
test -f exp/exp111_blind_transfer.py && test -f exp/exp111_jobs.py && test -f exp/exp111_grade.py \
  && test -f exp/jobs/e111_config/dataset.json || { echo "E111_SHA predates exp111"; exit 2; }
FROZEN=0
grep -q '^\*\*Status: frozen' docs/plan/blind_transfer.md 2>/dev/null && FROZEN=1
PY=(uv run --frozen python exp/exp111_blind_transfer.py)
GPY=(uv run --frozen python exp/exp111_grade.py)
JPY=("$VENV/bin/python" exp/exp111_jobs.py)
CONFIG=$CODE/exp/jobs/e102_config/model.yaml
GATE=$OUT/e111_gate.json
echo "== $(date -Is) job $JOB on $(hostname): exp111 $MODE, commit $E111_SHA, page $([ $FROZEN = 1 ] && echo frozen || echo draft) =="

# shown output is masked: the private directory, window names, item and tile names, numbers of 4+ digits and decimals
masked() { sed -E "s#$PRIV#<private>#g; s#$DEPLOY#<deploy>#g; s/\b[abg][0-9]{4}\b/<window>/g; s/S2[ABC]_MSIL2A_[A-Za-z0-9_.]+/<item>/g; \
  s/\bT[0-9]{2}[A-Z]{3}\b/<tile>/g; s/-?[0-9]+\.[0-9]{2,}/<n>/g; s/-?[0-9]{4,}/<n>/g"; }

rec_ok() {   # rec_ok <record>: exists and says "ok": true
  test -s "$1" && python3 -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get("ok") is True else 1)' "$1"
}

run_logged() {   # run_logged <tag> <timeout> <command...>: output to $PRIV/logs/<tag>-$JOB.log; masked tail on failure
  local tag=$1 lim=$2; shift 2
  local log=$PRIV/logs/$tag-$JOB.log t0 rc=0
  t0=$(date +%s)
  timeout --kill-after=120 "$lim" "$@" > "$log" 2>&1 || rc=$?
  if [ "$rc" != 0 ]; then
    { [ "$rc" = 124 ] || [ "$rc" = 137 ]; } && echo "$tag hit its time limit ($lim)"
    echo "$tag failed (exit $rc) after $(( $(date +%s) - t0 )) s; the last lines of its log, masked:"
    tail -n 40 "$log" | masked
    exit 1
  fi
  echo "$tag: $(( $(date +%s) - t0 )) s"
}

budget_s() {   # budget_s <margin s>: the seconds left before this job's time limit, less the margin
  local end=${SLURM_JOB_END_TIME:-}
  [[ "$end" =~ ^[0-9]+$ ]] || end=$(( T_START + GATE_LIMIT_S ))
  echo $(( end - $(date +%s) - $1 ))
}

need_frozen() {
  [ "$FROZEN" = 1 ] || { echo "refused: docs/plan/blind_transfer.md is not frozen at E111_SHA; no blind plot is predicted or read before it is"; exit 2; }
}

same_guarded() {   # same_guarded <from> <to>: <from> is an ancestor of <to> and no GUARDED file differs between them
  local from=$1 to=$2
  [[ "$from" =~ ^[0-9a-f]{40}$ ]] || return 1
  git cat-file -e "$from^{commit}" 2>/dev/null || git fetch -q origin "$from" 2>/dev/null || return 1
  git merge-base --is-ancestor "$from" "$to" || return 1
  git diff --quiet "$from" "$to" -- "${GUARDED[@]}"
}

guard_phase2() {   # every phase-2 mode: frozen, the gate passed, the code it predicted and passed with, the page's draw
  local c
  need_frozen
  test -s "$GATE" || { echo "no e111_gate.json: run phase 1 first"; exit 1; }
  G_COMMITS=$(python3 -c 'import json,sys; g=json.load(open(sys.argv[1])); assert g.get("pass") is True, "the gate did not pass"; print(g["predict_commit"], g["commit"])' "$GATE") \
    || { echo "refused: the known-answer gate did not pass"; exit 2; }
  for c in $G_COMMITS; do
    same_guarded "$c" "$E111_SHA" \
      || { echo "refused: the gate predicted or passed at ${c:0:12}, which is not an ancestor of E111_SHA, or the exp111 code, configs, imported modules or the environment's pins changed since; rerun the gate"; exit 2; }
  done
  "${PY[@]}" --check-draw --private-dir "$PRIV" || { echo "refused: draw.json is not the draw the frozen page holds"; exit 2; }
}

usage() {   # usage <tag> [note]: /home's size and file count (one find) and our scratch tree, printed and recorded;
  # refused when unknown or near a limit, or with "note" (the end of prepare, which writes nothing to /home: a refusal
  # there would cancel the chain after hours of work and protect nothing) printed as a note and the job goes on
  local tmp=$PRIV/tmp/usage-$1-$JOB.txt rc=0 hk="" hn="" partial=0 s mode=${2:-refuse}
  timeout 900 find "$HOMEDIR" -xdev -printf '%k\n' > "$tmp" 2>/dev/null || rc=$?
  if [ "$rc" = 0 ] || [ "$rc" = 1 ]; then
    read -r hk hn < <(awk '{s += $1} END {print s + 0, NR}' "$tmp")
    [ "$rc" = 1 ] && partial=1
  fi
  rm -f "$tmp"
  [[ "$hn" =~ ^[0-9]+$ ]] && [ "$hn" -gt 0 ] || { hk=""; hn=""; }
  s=$( (timeout 900 du -sk "$SCRATCH" 2>/dev/null || true) | tail -n 1 | cut -f1)
  [[ "$s" =~ ^[0-9]+$ ]] || s=""
  python3 -c 'import json,sys; v=lambda x: int(x) if x else None; json.dump({"home_kib": v(sys.argv[1]), "home_files": v(sys.argv[2]), "home_partial": sys.argv[3] == "1", "scratch_kib": v(sys.argv[4]), "home_limits": {"gib": 100, "files": 500000}, "refused_at": {"gib": 90, "files": 450000}}, open(sys.argv[5], "w"))' \
    "$hk" "$hn" "$partial" "$s" "$REC/e111_usage_$1.json"
  if [ -z "$hk" ]; then
    [ "$mode" = note ] && { echo "note: /home's usage is unknown (find exit $rc); recorded as unknown, the job goes on"; return 0; }
    echo "refused: /home's usage is unknown (find exit $rc; 124 is its 15-minute timeout): nothing runs until it is known"; exit 1
  fi
  echo "usage: /home $(( hk / 1048576 )) GiB in $hn files$([ "$partial" = 1 ] && echo ' (some entries unreadable: a lower bound)'), our scratch tree $([ -n "$s" ] && echo "$(( s / 1048576 )) GiB" || echo unknown)"
  if [ "$mode" = note ]; then
    [ "$hk" -lt $(( 90 * 1048576 )) ] || echo "note: /home is above 90 GiB; clean it before the gate's logs land (the 100 GiB hard limit locked the account on 9 October)"
    [ "$hn" -lt 450000 ] || echo "note: /home holds $hn files, above 450,000 (the hard limit is 500,000 files)"
    return 0
  fi
  [ "$hk" -lt $(( 90 * 1048576 )) ] || { echo "refused: /home is above 90 GiB (the 100 GiB hard limit locked the account on 9 October)"; exit 1; }
  [ "$hn" -lt 450000 ] || { echo "refused: /home holds $hn files, above 450,000 (the hard limit is 500,000 files)"; exit 1; }
}

need_venv() {
  test -x "$VENV/bin/rslearn" || { echo "no rslearn venv at $VENV (exp102's, purged?): run E111_MODE=env with E111_BUILD_VENV=1, then E111_VENV=$VENV_NEW"; exit 1; }
  "${JPY[@]}" versions --record "$REC/e111_versions_$MODE.json" > /dev/null || { echo "the venv's versions are not the pins"; exit 1; }
  "${JPY[@]}" no-runner
}

sha_ok() { [ -s "$1" ] && echo "$2  $1" | sha256sum -c --quiet - > /dev/null 2>&1; }

resolve_inputs() {   # BREADTH and AI2_GEOMETRY to verified copies (home or the pinned file, else the fetched ones); 1 if one is missing
  local miss=0
  if ! uv run --frozen python exp/exp103_jobs.py verify-breadth --dir "$BREADTH" > /dev/null 2>&1; then
    if uv run --frozen python exp/exp103_jobs.py verify-breadth --dir "$BREADTH_FETCH" > /dev/null 2>&1; then
      BREADTH=$BREADTH_FETCH
    else miss=1; fi
  fi
  if ! sha_ok "$AI2_GEOMETRY" "$AI2_GEOMETRY_SHA256"; then
    if sha_ok "$AI2_GEOMETRY_FETCH" "$AI2_GEOMETRY_SHA256"; then AI2_GEOMETRY=$AI2_GEOMETRY_FETCH; else miss=1; fi
  fi
  return $miss
}

ckpts() {   # sets CKPT[v1], CKPT[v12], sha256 checked; downloads only with E111_HF_OFFLINE=0
  declare -gA CKPT
  CKPT[v1]=$("$VENV/bin/python" -c "from huggingface_hub import hf_hub_download as d; print(d('$V1_REPO', 'model.ckpt', revision='$V1_REV'))") \
    || { echo "v1's checkpoint is not in \$SCRATCH/hf (E111_HF_OFFLINE=0 fetches it, 1.04 GB; the owner's call)"; exit 1; }
  if [ -s "$V12_SCRATCH" ]; then CKPT[v12]=$V12_SCRATCH; else
    CKPT[v12]=$("$VENV/bin/python" -c "from huggingface_hub import hf_hub_download as d; print(d('$V12_REPO', 'model.ckpt', revision='$V12_REV'))") \
      || { echo "v1.2's checkpoint is gone from scratch and not cached (E111_HF_OFFLINE=0 fetches the public copy, 1.16 GB)"; exit 1; }
  fi
  echo "$V1_SHA256  ${CKPT[v1]}" | sha256sum -c --quiet - || { echo "v1's sha256 is not the pinned one"; exit 1; }
  echo "$V12_SHA256  ${CKPT[v12]}" | sha256sum -c --quiet - || { echo "v1.2's sha256 is not exp102's"; exit 1; }
  echo "checkpoints: v1 $V1_REPO @ ${V1_REV:0:8} and v1.2 (exp102), sha256 verified"
}

gpu_prep() {   # gpu_prep <device record>: the GPU, recorded (name, capability, torch's kernels for it; this job and its
  # commit), the venv, the checkpoints. The gate gives a private path and moves it to $REC only after its four
  # predictions are checked: the device record is the gate's predict_commit, and must not exist for a gate that died
  local dev=$1
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || { echo "no GPU: submit with --gpus=1"; exit 2; }
  need_venv
  "${JPY[@]}" device --commit "$E111_SHA" --record "$dev" > /dev/null || { echo "torch sees no usable GPU"; exit 2; }
  python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); print("device:", r["name"], r["arch"], "torch", r["torch"])' "$dev"
  ckpts
}

run_predict() {   # run_predict <dataset root> <model v1|v12> <layer> <config> <tag> [timeout]
  local ds=$1 m=$2 layer=$3 cfg=$4 tag=$5 lim=${6:-3h} work
  export DATASET_PATH=$ds PREDICTION_OUTPUT_LAYER=$layer MODEL_ID=${MODEL_IDS[$m]} NUM_WORKERS=$WORKERS
  export TRAINER_DATA_PATH=$OUT/trainer/$tag-$JOB
  # rslearn's ModelDataset writes every window's name, bounds and projection to tempfile.gettempdir()
  # (rslearn/train/dataset.py:1007-1019 at v0.1.17): keep them in a private directory (TSHORT, mode 700)
  set_tmp
  work=$PRIV/tmp/work-$tag-$JOB
  mkdir -p "$TRAINER_DATA_PATH" "$TMPDIR" "$work"
  echo "== $(date -Is) rslearn model predict ($tag): $m, layer $layer, $WORKERS workers, limit $lim =="
  ( cd "$work" && run_logged "$tag" "$lim" "$VENV/bin/rslearn" model predict --config "$cfg" --ckpt_path "${CKPT[$m]}" ) || return 1
  { grep -E "examples in split predict|Skipped [0-9]+ windows|Loaded [0-9]+ windows" "$PRIV/logs/$tag-$JOB.log" \
      || echo "(no count line found)"; } | masked | sed 's/^/  /'
}

fresh_layer() {   # fresh_layer <root> <layer>: refuse an existing layer unless E111_REDO names this mode (then removed)
  local n
  n=$(find "$1/windows" -mindepth 4 -maxdepth 4 -type d -name "$2" -path '*/layers/*' | wc -l)
  if [ "$n" -gt 0 ]; then
    redo_for "$MODE" || { echo "$n window(s) already hold layers/$2/; E111_REDO=$MODE removes them first"; exit 2; }
    find "$1/windows" -mindepth 4 -maxdepth 4 -type d -name "$2" -path '*/layers/*' -exec rm -rf {} +
  fi
}

remove_aprime() {   # A''s layer out of exp98's deployment dataset: an EXIT trap in gate and gatecheck, and in pack1
  if [ -s "$PRIV/aprime/aprime_plan.json" ] && [ -x "$VENV/bin/python" ]; then
    "$VENV/bin/python" "$CODE/exp/exp111_jobs.py" remove-layer --plan "$PRIV/aprime/aprime_plan.json" --layer output_e111_v1 \
      || echo "note: A''s layer could not be removed here; pack1 tries again"
  fi
}

set_tmp() {   # export TMPDIR as a short private directory, numbered within the job; refuse one a socket cannot sit under
  TMPN=$((TMPN + 1))
  export TMPDIR=$TSHORT/$JOB.$TMPN
  [ $(( ${#TMPDIR} + 32 )) -le 100 ] || { echo "TMPDIR $TMPDIR is too long for rslearn's forkserver socket"; exit 2; }
  mkdir -p "$TMPDIR"
}

prepare_root() {   # prepare_root <name>: three passes of prepare/ingest/materialize, then the SCL sidecar (windows with
  # all twelve groups only), its materialize (failures added to pass 3's), precheck (quarantine within the allowance),
  # the SCL check (windows with input only), inventory. The local smoke runs this function itself, rslearn stood in for
  local name=$1 ds=$PRIV/roots/$1 pass maxf
  local -a plus
  set_tmp
  for pass in 1 2 3; do
    # pass 1 stops a systematic failure (above 2% of the windows); pass 2 retries; after pass 3 at most MAX_FAILED_SHARE
    # of the windows may fail, prepare's and materialize's failures summed (the owner's allowance; proposed 0)
    case $pass in 1) maxf=0.02 ;; 2) maxf=1 ;; 3) maxf=$MAX_FAILED_SHARE ;; esac
    plus=(); [ "$pass" = 3 ] && plus=(--plus "$REC/e111_${name}_prepare3.json")
    run_logged "$name-prepare$pass" 3h "$VENV/bin/rslearn" dataset prepare --root "$ds" --workers "$NCPU" \
      --retry-max-attempts 5 --retry-backoff-seconds 10 --ignore-errors
    "${JPY[@]}" prepare-summary --log "$PRIV/logs/$name-prepare$pass-$JOB.log" --max-failed-share "$maxf" \
      --record "$REC/e111_${name}_prepare$pass.json" | masked \
      || { echo "$name: prepare failed at more than $maxf of the windows in pass $pass: stopped (resubmit with --from prepare to resume)"; exit 1; }
    run_logged "$name-ingest$pass" 1h "$VENV/bin/rslearn" dataset ingest --root "$ds" --workers "$NCPU"
    run_logged "$name-materialize$pass" 3h "$VENV/bin/rslearn" dataset materialize --root "$ds" --workers "$NCPU" \
      --retry-max-attempts 5 --retry-backoff-seconds 10 --ignore-errors
    "${JPY[@]}" prepare-summary --log "$PRIV/logs/$name-materialize$pass-$JOB.log" --max-failed-share "$maxf" \
      ${plus[@]+"${plus[@]}"} --record "$REC/e111_${name}_materialize$pass.json" | masked \
      || { echo "$name: materialize failed at more than $maxf of the windows in pass $pass: stopped (resubmit with --from prepare to resume)"; exit 1; }
  done
  # the SCL sidecar of the windows with all twelve groups (a window that failed within the allowance gets none)
  "${JPY[@]}" scl-sidecar --root "$ds" --scl "$PRIV/scl/$name"
  run_logged "$name-scl" 2h "$VENV/bin/rslearn" dataset materialize --root "$PRIV/scl/$name" --workers "$NCPU" \
    --retry-max-attempts 5 --retry-backoff-seconds 60 --ignore-errors
  "${JPY[@]}" prepare-summary --log "$PRIV/logs/$name-scl-$JOB.log" --layer sentinel2_scl --max-failed-share "$MAX_FAILED_SHARE" \
    --plus "$REC/e111_${name}_materialize3.json" --record "$REC/e111_${name}_scl_materialize.json" | masked \
    || { echo "$name: the SCL materialize failed at windows that, with pass 3's failures, exceed the allowance $MAX_FAILED_SHARE: stopped (resubmit with --from prepare to resume)"; exit 1; }
  # partial windows and windows whose SCL is incomplete: refused, or within the allowance quarantined (this job's own
  # directory), before anything compares the sidecar with the root
  "${JPY[@]}" precheck --root "$ds" --units "$PRIV/units_$name.json" --config "$CONFIG" --workers "$NCPU" \
    --scl "$PRIV/scl/$name" --max-drop-share "$MAX_FAILED_SHARE" --quarantine "$PRIV/quarantine" --run-tag "$JOB" \
    --record "$REC/e111_${name}_precheck.json" | masked
  "${JPY[@]}" scl-check --root "$ds" --scl "$PRIV/scl/$name" --record "$REC/e111_${name}_scl_check.json" | masked
  "${JPY[@]}" inventory --root "$ds" --units "$PRIV/units_$name.json" --quarantine "$PRIV/quarantine" \
    --record "$REC/e111_${name}_inventory.json"
}

record_inputs() {   # record_inputs <inputs ok 0|1>: $REC/e111_inputs.json (counts and flags only); 1 if a required input is missing
    python3 - "$BREADTH" "$AI2_GEOMETRY" "$AI2_GEOMETRY_SHA256" "$LABELS" "$INV23" "$RUN23" "$INV17" "$RUN17" "$1" "$REC/e111_inputs.json" <<'EOF'
import hashlib, json, os, sys
breadth, geom, gsha, labels, inv23, run23, inv17, run17, inputs_ok, out = sys.argv[1:]
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()
lab = os.path.join(labels, "spatial_split")
r = {"timesync_and_natural_earth_verified": inputs_ok == "1" or None,
     "breadth_copy": "home" if breadth.startswith("/home/") else "fetched to scratch",
     "ai2_geometry_verified": os.path.isfile(geom) and sha(geom) == gsha,
     "ai2_geometry_copy": "deploy (purge-bound)" if "/deploy/" in geom else "fetched to scratch",
     "label_windows": len(os.listdir(lab)) if os.path.isdir(lab) else 0,
     "deployment_2023": os.path.isfile(inv23) and os.path.isdir(run23),
     "deployment_2017": os.path.isfile(inv17) and os.path.isdir(run17)}
r["required_ok"] = inputs_ok == "1" and r["ai2_geometry_verified"] and r["label_windows"] == 1459
r["ok"] = r["required_ok"]
json.dump(r, open(out, "w"), indent=1)
print(json.dumps(r))
if not r["deployment_2023"] or not r["deployment_2017"]:
    print("note: a deployment dataset is gone (the scratch purge): A' and the item-identity diagnostics will be skipped")
if not r["required_ok"]:
    print("missing: the TimeSync file, Natural Earth or Ai2's geometry (E111_FETCH=1 fetches and checks them: 26 MB, "
          "13 MB, KB), or exp89's label windows (exp/jobs/e89.sh, E89_MODE=inv E89_ARM=awf: Ai2's 1.87 GB tar)")
    sys.exit(1)
EOF
}

case "$MODE" in
  env)
    usage env
    if [ ! -x "$VENV/bin/rslearn" ]; then
      [ "${E111_BUILD_VENV:-0}" = 1 ] || { echo "exp102's venv is gone; E111_BUILD_VENV=1 builds $VENV_NEW (PyPI, a few GB with torch: the owner's call)"; exit 1; }
      [ "$VENV" = "$VENV_NEW" ] || { echo "with E111_BUILD_VENV=1 set E111_VENV=$VENV_NEW too"; exit 2; }
      uv venv -q --python 3.12 "$VENV"
      # the versions exp103 recorded in exp102's venv (e103_predict17_env.json)
      uv pip install -q --python "$VENV/bin/python" --exclude-newer 2026-10-11T00:00:00Z "rslearn[extra]==0.1.17" \
        "olmoearth-pretrain-minimal==0.1.2" "torch==2.14.1" "lightning==2.6.6" "jsonargparse==4.52.0"
      uv pip freeze --python "$VENV/bin/python" > "$REC/e111_venv_freeze.txt"
    fi
    need_venv
    "${JPY[@]}" check-configs --record "$REC/e111_configs.json" > /dev/null || { echo "the configs are not as pinned"; exit 1; }
    ckpts
    D=$PRIV/tmp/env-ds-$JOB
    mkdir -p "$D"
    cp exp/jobs/e111_config/dataset.json "$D/config.json"
    export DATASET_PATH=$D PREDICTION_OUTPUT_LAYER=output_v1 NUM_WORKERS=0 TRAINER_DATA_PATH=$OUT/trainer/env-$JOB
    set_tmp
    "${JPY[@]}" parse-model --record "$REC/e111_parse_model.json" > /dev/null || { echo "model.yaml does not parse as expected"; exit 1; }
    for m in v1 v12; do
      echo "== strict load of $m into the module model.yaml builds (CPU) =="
      run_logged "loadcheck-$m" 30m "${JPY[@]}" load-check --config "$CONFIG" --ckpt "${CKPT[$m]}" \
        --model-id "${MODEL_IDS[$m]}" --record "$REC/e111_loadcheck_$m.json"
      python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); print({k: r.get(k) for k in ("model_id","load","keys_in_checkpoint","missing","unexpected","shape_mismatch","non_tensor","strict_load","load_error","accepted_unexpected","legacy_table_max_abs_diff","legacy_table_is_the_formula","legacy_max","plain_weights_only","ok")})' "$REC/e111_loadcheck_$m.json"
    done
    rm -rf "$D"
    echo "== the record's environment (uv.lock; the first run in a fresh clone installs it from PyPI) =="
    run_logged uv-sync 30m uv sync --frozen
    if [ "${E111_FETCH:-0}" = 1 ]; then
      echo "== E111_FETCH=1: a missing or unverified TimeSync file, Natural Earth or Ai2 geometry is fetched and checked =="
      if ! uv run --frozen python exp/exp103_jobs.py verify-breadth --dir "$BREADTH" > /dev/null 2>&1; then
        run_logged fetch-breadth 10m uv run --frozen python exp/timesync_awf_crosswalk.py fetch --dest "$BREADTH_FETCH"
      fi
      if ! sha_ok "$AI2_GEOMETRY" "$AI2_GEOMETRY_SHA256" && ! sha_ok "$AI2_GEOMETRY_FETCH" "$AI2_GEOMETRY_SHA256"; then
        mkdir -p "$(dirname "$AI2_GEOMETRY_FETCH")"
        curl -fsS --max-time 120 -o "$AI2_GEOMETRY_FETCH.partial" "$AI2_GEOMETRY_URL"
        sha_ok "$AI2_GEOMETRY_FETCH.partial" "$AI2_GEOMETRY_SHA256" || { rm -f "$AI2_GEOMETRY_FETCH.partial"; echo "Ai2's geometry as fetched is not the pinned file (sha256)"; exit 1; }
        mv "$AI2_GEOMETRY_FETCH.partial" "$AI2_GEOMETRY_FETCH"
      fi
    fi
    INPUTS_OK=1
    resolve_inputs || INPUTS_OK=0
    record_inputs "$INPUTS_OK"
    echo "== $(date -Is) env done: records/e111_versions_env.json, e111_loadcheck_v1/v12.json, e111_inputs.json ==" ;;

  select)
    need_venv
    resolve_inputs || { echo "the TimeSync file, Natural Earth or Ai2's geometry is missing or not as pinned: run env with E111_FETCH=1"; exit 1; }
    echo "== exp99's 309, the pool, the draw (seed 111, the sizes of the commit: SIZES); an existing equal draw is kept =="
    S_ARGS=(--select --private-dir "$PRIV" --timesync "$BREADTH/eastafrica_yearly_point_data.csv"
            --request-geometry "$AI2_GEOMETRY" --boundaries "$BREADTH/ne_10m_admin_0_countries.geojson"
            --record "$REC/e111_select.json")
    W_ARGS=(--windows --private-dir "$PRIV" --labels "$LABELS" --record "$REC/e111_windows.json")
    if redo_for select; then
      [ "$FROZEN" = 0 ] || { echo "refused: E111_REDO=select after the page is frozen (it would replace the frozen draw, or remove blind17 with phase 2's outputs)"; exit 2; }
      S_ARGS+=(--redo); W_ARGS+=(--redo)
    fi
    # the drawn plots at which exp99 (v1) or exp103 (v1.2) wrote an output: exp99's own 2017 windows while they are
    # on scratch (purged around 8 to 9 November) and readable whole, else the gate17 cells alone, a lower bound (the
    # record says which, and why: a partially purged deployment falls back instead of failing select)
    if [ -s "$INV17" ] && [ -d "$RUN17" ]; then W_ARGS+=(--deploy-inv "$INV17" --deploy-run "$RUN17"); fi
    "${PY[@]}" "${S_ARGS[@]}" | masked
    echo "== the windows of gate23, gate17 and blind17 (roots equal to this run's are kept with their imagery) =="
    "${PY[@]}" "${W_ARGS[@]}" | masked
    "${JPY[@]}" check-configs --record "$REC/e111_configs.json" > /dev/null
    for r in gate23 gate17 blind17; do
      "${JPY[@]}" check-dataset --root "$PRIV/roots/$r" --units "$PRIV/units_$r.json" --workers "$NCPU" \
        --record "$REC/e111_${r}_dataset_check.json" | python3 -c 'import json,sys; r=json.loads(sys.stdin.read().strip().splitlines()[-1]); print(json.dumps({k: r.get(k) for k in ("root","windows","sizes","epsg","bad","periods","min_margin_to_unshared_edge_px","ok")})); sys.exit(0 if r["ok"] else 1)'
    done
    for y in 23 17; do
      if [ "$y" = 23 ]; then I=$INV23; R=$RUN23; else I=$INV17; R=$RUN17; fi
      if [ -s "$I" ] && [ -d "$R" ]; then
        "${JPY[@]}" deploy-time-range --inv "$I" --run "$R" --year "20$y" --record "$REC/e111_deploy_time_range$y.json" \
          | python3 -c 'import json,sys; r=json.loads(sys.stdin.read()); print({k: r[k] for k in ("year","distinct","windows","unreadable","checked","equals_literal")}); sys.exit(0 if r["ok"] else 1)' \
          || { echo "the deployment's 20$y windows do not hold the literal time range: the windows would read other periods"; exit 1; }
      else
        echo "note: the 20$y deployment dataset is gone; the literal time range is used unchecked"
      fi
    done
    echo "== $(date -Is) done: records/e111_select.json, e111_windows.json, e111_*_dataset_check.json ==" ;;

  prepare)
    need_venv
    for r in gate23 gate17 blind17; do
      rec_ok "$REC/e111_${r}_dataset_check.json" || { echo "no passing dataset check for $r: run E111_MODE=select first"; exit 1; }
    done
    ROOTS=(gate23 gate17)
    [ "${E111_BLIND_PREP:-1}" = 1 ] && ROOTS+=(blind17)
    WD_ROOTS=()
    for r in gate23 gate17 blind17; do WD_ROOTS+=("$PRIV/roots/$r"); done
    PGID=$(ps -o pgid= -p $$ | tr -d ' ')
    "${JPY[@]}" watchdog --dir "$PRIV" --roots "${WD_ROOTS[@]}" --max-gb "$MAX_GB" --interval 60 --pgid "$PGID" \
      --scancel "$JOB" --parent $$ --reason "$OUT/logs/watchdog-$JOB.txt" &
    WD=$!
    trap 'kill $WD 2>/dev/null || true' EXIT
    echo "== the smoke root: two of gate23's windows, prepared and materialized first =="
    SM=$PRIV/roots/smoke2
    rm -rf "$SM"
    mkdir -p "$SM/windows/e111"
    cp "$PRIV/roots/gate23/config.json" "$SM/config.json"
    for n in a0000 a0001; do mkdir -p "$SM/windows/e111/$n"; cp "$PRIV/roots/gate23/windows/e111/$n/metadata.json" "$SM/windows/e111/$n/"; done
    set_tmp
    run_logged smoke2-prepare 30m "$VENV/bin/rslearn" dataset prepare --root "$SM" --workers 2 --retry-max-attempts 5 --retry-backoff-seconds 10
    run_logged smoke2-materialize 30m "$VENV/bin/rslearn" dataset materialize --root "$SM" --workers 2 --retry-max-attempts 5 --retry-backoff-seconds 10
    "${JPY[@]}" check-groups --root "$SM" --record "$REC/e111_smoke2_groups.json" | masked \
      || { echo "the smoke root's item groups or reflectance are not as expected: stopped before the full build"; exit 1; }
    rm -rf "$SM"
    for r in "${ROOTS[@]}"; do
      echo "== $(date -Is) $r =="
      prepare_root "$r"
    done
    if [ -s "$OUT/logs/watchdog-$JOB.txt" ]; then cat "$OUT/logs/watchdog-$JOB.txt"; exit 1; fi
    du -sh "$PRIV" | sed "s#$PRIV#<private>#"
    usage prepare note
    echo "== $(date -Is) done: records/e111_*_inventory.json (the input loss by root and, for blind17, by cell) ==" ;;

  gate)
    trap remove_aprime EXIT
    # a gate job replaces every record of an earlier gate before it predicts: a gate that dies part way then leaves no
    # device record (gatecheck refuses), never an earlier gate's output checks beside a newer commit
    rm -f "$REC/e111_device_gate.json" "$GATE" "$REC/e111_aprime_config.json" "$REC/e111_read_aprime.json"
    for r in gate23 gate17; do for m in v1 v12; do rm -f "$REC/e111_${r}_${m}_output.json"; done; done
    DEV_PENDING=$PRIV/tmp/device-gate-$JOB.json
    gpu_prep "$DEV_PENDING"
    for r in gate23 gate17; do
      rec_ok "$REC/e111_${r}_precheck.json" || { echo "no passing precheck for $r: run E111_MODE=prepare first"; exit 1; }
      rec_ok "$REC/e111_${r}_scl_check.json" || { echo "no passing SCL check for $r: run E111_MODE=prepare first"; exit 1; }
    done
    for r in gate23 gate17; do
      for m in v1 v12; do
        fresh_layer "$PRIV/roots/$r" "output_$m"
        run_predict "$PRIV/roots/$r" "$m" "output_$m" "$CONFIG" "pred-$r-$m"
        "${JPY[@]}" check-output --root "$PRIV/roots/$r" --layer "output_$m" --record "$REC/e111_${r}_${m}_output.json" \
          | masked || { echo "the $r $m output check failed"; exit 1; }
      done
    done
    # all four predictions checked: now, and only now, the gate's device record (its GPU, predict_commit and job)
    mv "$DEV_PENDING" "$REC/e111_device_gate.json"
    echo "== $(date -Is) gate predictions done =="
    rm -f "$PRIV/read/aprime_v1.npz"
    if [ "${E111_APRIME:-1}" = 1 ] && [ -s "$INV23" ] && [ -d "$RUN23" ]; then
      LEFT=$(budget_s 900)
      if [ "$LEFT" -lt 1200 ]; then
        echo "note: A' skipped: $LEFT s left before the job's limit after a 15-minute margin"
      else
        echo "== diagnostic A' (report-only; its failure is a note, never the gate's): v1 under rslearn 0.1.17 on exp98's own windows that hold the 259 points, at most $LEFT s =="
        ( "${JPY[@]}" aprime-config --units "$PRIV/units_gate23.json" --inv "$INV23" --run "$RUN23" --config "$CONFIG" \
              --out-dir "$PRIV/aprime" --record "$REC/e111_aprime_config.json" || exit 1
          "${JPY[@]}" remove-layer --plan "$PRIV/aprime/aprime_plan.json" --layer output_e111_v1 || exit 1
          for C in "$PRIV"/aprime/model_aprime_r*.yaml; do
            RI=$(basename "$C" .yaml); RI=${RI#model_aprime_r}
            ROOTP=$(python3 -c 'import json,sys; print([p["root"] for p in json.load(open(sys.argv[1])) if p["root_index"] == int(sys.argv[2])][0])' "$PRIV/aprime/aprime_plan.json" "$RI") || exit 1
            L=$(budget_s 900)
            [ "$L" -gt 300 ] || { echo "A' ran out of time before root $RI"; exit 1; }
            run_predict "$ROOTP" v1 output_e111_v1 "$C" "aprime-r$RI" "${L}s" || exit 1
          done
          "${PY[@]}" --read-aprime --private-dir "$PRIV" --aprime-plan "$PRIV/aprime/aprime_plan.json" \
            --record "$REC/e111_read_aprime.json" | masked || exit 1
        ) || echo "note: diagnostic A' did not complete (report-only); the gate's predictions above stand"
      fi
    else
      echo "note: A' skipped (E111_APRIME=0 or exp98's deployment dataset is gone)"
    fi
    echo "== $(date -Is) gate done ==" ;;

  gatecheck)
    trap remove_aprime EXIT
    # the device record and the four output checks: present, passing, and one gate job's (the device record is written
    # only after that job's four checks passed, and a gate job removes all five when it starts)
    need_venv
    "${JPY[@]}" gate-records --rec "$REC" || exit 2
    # the gate's predictions ran at the gate job's commit; a --from gatecheck resubmission at a later commit must not
    # pass a gate whose guarded code changed after the predictions
    P_COMMIT=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("commit") or "")' "$REC/e111_device_gate.json")
    same_guarded "$P_COMMIT" "$E111_SHA" \
      || { echo "refused: the gate's predictions ran at ${P_COMMIT:0:12}; it is not an ancestor of E111_SHA or a guarded file changed since: resubmit with --from gate"; exit 2; }
    [ ! -e "$GATE" ] || redo_for gatecheck || { echo "e111_gate.json exists; E111_REDO=gatecheck replaces it"; exit 2; }
    rm -f "$GATE"
    RD=(); redo_for gatecheck && RD=(--redo)
    for m in v1 v12; do
      "${PY[@]}" --read gate23 "$m" --private-dir "$PRIV" --record "$REC/e111_read_gate23_$m.json" ${RD[@]+"${RD[@]}"} | masked
    done
    mkdir -p "$PRIV/identity"
    rm -f "$PRIV/identity/identity_gate23.json" "$PRIV/identity/identity_gate17.json"   # never an earlier run's flags
    if [ -s "$INV23" ] && [ -d "$RUN23" ]; then
      "${JPY[@]}" item-identity --root "$PRIV/roots/gate23" --units "$PRIV/units_gate23.json" --inv "$INV23" --run "$RUN23" \
        --flags-out "$PRIV/identity/identity_gate23.json" --record "$REC/e111_identity_gate23.json"
    fi
    A_RC=0
    "${PY[@]}" --gate A --private-dir "$PRIV" --gate-file "$GATE" --identity-dir "$PRIV/identity" \
      --device-file "$REC/e111_device_gate.json" || A_RC=$?
    if [ "$A_RC" != 0 ]; then echo "gate A did not pass (exit $A_RC): gate B is not read; nothing blind is predicted"; exit 4; fi
    for m in v1 v12; do
      "${PY[@]}" --read gate17 "$m" --private-dir "$PRIV" --record "$REC/e111_read_gate17_$m.json" ${RD[@]+"${RD[@]}"} | masked
    done
    if [ -s "$INV17" ] && [ -d "$RUN17" ]; then
      "${JPY[@]}" item-identity --root "$PRIV/roots/gate17" --units "$PRIV/units_gate17.json" --inv "$INV17" --run "$RUN17" \
        --flags-out "$PRIV/identity/identity_gate17.json" --record "$REC/e111_identity_gate17.json"
    fi
    B_RC=0
    "${PY[@]}" --gate B --private-dir "$PRIV" --gate-file "$GATE" || B_RC=$?
    [ "$B_RC" = 0 ] || { echo "gate B did not pass (exit $B_RC): nothing blind is predicted"; exit 4; }
    echo "== $(date -Is) the gate passed: e111_out/e111_gate.json ==" ;;

  pack1|pack2)
    N=${MODE#pack}
    [ "$N" = 1 ] && remove_aprime
    T=$OUT/fetch/e111_phase$N
    rm -rf "$T" "$T.tar.gz" "$T.tar.gz.sha256"
    mkdir -p "$T/slurm"
    cp -p "$REC"/*.json "$T/" 2>/dev/null || true
    cp -p "$GATE" "$T/" 2>/dev/null || true
    if [ "$N" = 2 ]; then cp -p "$OUT"/units/* "$OUT"/grade/* "$T/" 2>/dev/null || true; fi
    for j in ${E111_JOBS:-}; do
      [[ "$j" =~ ^[0-9]+$ ]] || continue
      for f in "$HOME_REPO"/slurm/e111*-"$j".out; do
        if [ -s "$f" ]; then masked < "$f" > "$T/slurm/$(basename "$f")"; fi
      done
    done
    # the one key check (exp111_jobs.RECORD_FORBIDDEN_KEYS, the record writers' own): standard library only, so any
    # Python left on the node runs it
    PACK_PY=python3
    for c in "$VENV/bin/python" "$CODE/.venv/bin/python"; do [ -x "$c" ] && { PACK_PY=$c; break; }; done
    "$PACK_PY" "$CODE/exp/exp111_jobs.py" pack-check --dir "$T"
    tar -C "$OUT/fetch" -czf "$T.tar.gz" "e111_phase$N"
    ( cd "$OUT/fetch" && sha256sum "e111_phase$N.tar.gz" > "e111_phase$N.tar.gz.sha256" )
    rm -rf "$T"
    ls -la "$OUT/fetch" | sed "s#$OUT#<out>#"
    echo "== fetch from the Mac: scp aicr:$OUT/fetch/e111_phase$N.tar.gz* <scratchpad>/exp111/fetch/ (one Duo push) ==" ;;

  predict)
    guard_phase2
    gpu_prep "$REC/e111_device_predict.json"
    "${JPY[@]}" device --expect-gate "$GATE" > /dev/null \
      || { echo "refused: this GPU is not the one the gate ran on (e111_gate.json's device); predict on the gate's device"; exit 2; }
    if [ ! -s "$REC/e111_blind17_precheck.json" ]; then
      echo "blind17 was not prepared in phase 1 (E111_BLIND_PREP=0): prepare it in a CPU job first (E111_MODE=prepare)"; exit 1
    fi
    rec_ok "$REC/e111_blind17_precheck.json" || { echo "blind17's precheck did not pass"; exit 1; }
    rec_ok "$REC/e111_blind17_scl_check.json" || { echo "blind17's SCL check did not pass"; exit 1; }
    # both layers checked first, so a refusal leaves the earlier records as they were; then, as the gate, an earlier
    # predict's output checks go, so a predict that dies part way leaves the read refusing
    for m in v1 v12; do fresh_layer "$PRIV/roots/blind17" "output_$m"; done
    for m in v1 v12; do rm -f "$REC/e111_blind17_${m}_output.json"; done
    for m in v1 v12; do
      run_predict "$PRIV/roots/blind17" "$m" "output_$m" "$CONFIG" "pred-blind17-$m"
      "${JPY[@]}" check-output --root "$PRIV/roots/blind17" --layer "output_$m" --record "$REC/e111_blind17_${m}_output.json" \
        | masked || { echo "the blind17 $m output check failed"; exit 1; }
    done
    echo "== $(date -Is) done: records/e111_blind17_*_output.json ==" ;;

  read)
    guard_phase2
    RD=(); redo_for read && RD=(--redo)
    for m in v1 v12; do
      rec_ok "$REC/e111_blind17_${m}_output.json" || { echo "no passing blind output check for $m: run E111_MODE=predict first"; exit 1; }
      "${PY[@]}" --read blind17 "$m" --private-dir "$PRIV" --record "$REC/e111_read_blind17_$m.json" ${RD[@]+"${RD[@]}"} | masked
    done ;;

  units)
    guard_phase2
    U=$OUT/units
    [ ! -e "$U" ] || redo_for units || { echo "e111_out/units exists; E111_REDO=units replaces it"; exit 2; }
    rm -rf "$U"
    "${PY[@]}" --units --private-dir "$PRIV" --gate-file "$GATE" --out-dir "$U"
    ls -la "$U" | sed "s#$OUT#<out>#" ;;

  grade)
    guard_phase2
    G=$OUT/grade
    [ ! -e "$G" ] || redo_for grade || { echo "e111_out/grade exists; E111_REDO=grade replaces it"; exit 2; }
    rm -rf "$G"
    run_logged grade-smoke 20m "${GPY[@]}" --smoke --out-dir "$OUT/logs/grade_smoke"   # synthetic: never packed
    "${GPY[@]}" --units-dir "$OUT/units" --out-dir "$G" | sed "s#$OUT#<out>#"
    echo "== $(date -Is) done: e111_out/grade/exp111_summary.json ==" ;;
esac
du -sh "$OUT" "$PRIV" 2>/dev/null | sed "s#$PRIV#<private>#; s#$OUT#<out>#; s/^/on scratch: /" || true
