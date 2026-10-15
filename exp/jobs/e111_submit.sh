#!/bin/bash
# exp111's submission, run by the OWNER from the Mac in the record repo after commit and push (never by an agent):
#   bash exp/jobs/e111_submit.sh phase1 [--from env|select|prepare|gate|gatecheck]   # ... -> gate -> gatecheck -> pack1
#   bash exp/jobs/e111_submit.sh phase2 [--from predict|read|units|grade]            # after freezing: ... -> pack2
# One ssh connection for the whole chain (ControlMaster: one Duo push; the socket lives 10 minutes in ~/.ssh), one
# sbatch per line on the login node and nothing else there; every dependent job has --kill-on-invalid-dep=yes, so a
# failed step cancels the rest instead of leaving them pending; pack runs afterany, so the records of a failed chain
# still come back.
#
# --from resubmits a chain from a later step after a failure, at the same one Duo push: the steps before it are not
# run again (select keeps an existing equal draw and roots; prepare resumes, rslearn skipping what is done). Every
# phase-1 submission sets E111_REDO to include gate and gatecheck, which replace only their own outputs (the gate roots'
# output layers, the gate file); select's own redo (a different draw, the roots' imagery fetched again) and every
# phase-2 redo stay the owner's explicit E111_REDO (a comma list; see exp/jobs/e111.sh). Any phase-2 rerun is reported.
#
# The gate and the blind predict both run on b200-devel only: on B200 GPUs, as exp102 and exp103 did (they ran on
# b200-batch; the device check compares the GPU, not the partition); predict also refuses another GPU than the one
# e111_gate.json records. A --from gatecheck resubmission at a later commit is refused by gatecheck itself when a
# guarded file changed since the gate job's commit (its predictions), or when the gate job's records are not one job's
# (a gate that died part way): resubmit --from gate then. The pack jobs ask for Slurm mail at their end
# (--mail-type=END,FAIL, which aicr's Slurm Basics page lists; no --mail-user is written here, so the mail goes to the
# account's default address).
# Nothing here polls: fetch afterwards with ONE scp (one more Duo push), once the mail has come or at the planned time:
#   mkdir -p <scratchpad>/exp111/fetch && scp 'aicr:/scratch/qi_zim_neu/olmoearth_inferenceX/e111_out/fetch/e111_phase1.tar.gz*' <scratchpad>/exp111/fetch/
#   (cd <scratchpad>/exp111/fetch && shasum -a 256 -c e111_phase1.tar.gz.sha256)
# Options pass through: E111_BLIND_PREP, E111_APRIME, E111_MAX_GB, E111_FETCH, E111_HF_OFFLINE, E111_BUILD_VENV,
# E111_VENV, E111_REDO (each the owner's call; see exp/jobs/e111.sh). Not options, part of the commit: the failure
# allowance (MAX_FAILED_SHARE in exp/jobs/e111.sh) and the draw's sizes (SIZES in exp/exp111_blind_transfer.py, with the
# 720 floor in exp/exp111_grade.py); E111_SIZES is refused.
# E111_DRY_RUN=1 prints the sbatch lines and connects to nothing (the local smoke uses it; no git check either).
set -euo pipefail
DRY=${E111_DRY_RUN:-0}
PHASE=${1:-}
FROM=
case "$PHASE" in phase1|phase2) ;; *) echo "usage: bash exp/jobs/e111_submit.sh phase1|phase2 [--from MODE]"; exit 2 ;; esac
if [ "${2:-}" = "--from" ]; then FROM=${3:-}; elif [ -n "${2:-}" ]; then echo "unknown argument: $2"; exit 2; fi
if [ "$PHASE" = phase1 ]; then STEPS=(env select prepare gate gatecheck); else STEPS=(predict read units grade); fi
if [ -n "$FROM" ]; then
  found=0; for s in "${STEPS[@]}"; do [ "$s" = "$FROM" ] && found=1; done
  [ "$found" = 1 ] || { echo "--from takes one of: ${STEPS[*]} (got '$FROM')"; exit 2; }
fi
if [ "$DRY" = 1 ]; then
  SHA=0000000000000000000000000000000000000000
else
  cd "$(git rev-parse --show-toplevel)"
  SHA=$(git rev-parse HEAD)
  git diff --quiet && git diff --cached --quiet || { echo "uncommitted changes: commit and push first"; exit 2; }
  git fetch -q origin main
  git merge-base --is-ancestor "$SHA" origin/main || { echo "$SHA is not on origin/main: push first"; exit 2; }
fi
if [ "$PHASE" = phase2 ] && [ "$DRY" != 1 ]; then
  grep -q '^\*\*Status: frozen' docs/plan/blind_transfer.md || { echo "docs/plan/blind_transfer.md is not frozen"; exit 2; }
  grep -qiE '^\*\*Draw sha256[^*]*\*\*[[:space:]]*`[0-9a-f]{64}`' docs/plan/blind_transfer.md \
    || { echo "docs/plan/blind_transfer.md holds no frozen draw sha256"; exit 2; }
fi
if [ "$PHASE" = phase1 ]; then
  E111_REDO=${E111_REDO:+$E111_REDO,}gate,gatecheck
fi
PASS=()
[ -z "${E111_DROP_PARTIAL:-}" ] || { echo "E111_DROP_PARTIAL is gone: MAX_FAILED_SHARE in exp/jobs/e111.sh is the failure allowance"; exit 2; }
[ -z "${E111_SIZES:-}" ] || { echo "E111_SIZES is gone: the draw's sizes are SIZES in exp/exp111_blind_transfer.py, part of the commit (the 720 floor is 80% of them)"; exit 2; }
for v in E111_BLIND_PREP E111_APRIME E111_MAX_GB E111_FETCH E111_HF_OFFLINE \
         E111_BUILD_VENV E111_VENV E111_REDO; do
  if [ -n "${!v:-}" ]; then
    [[ "${!v}" =~ ^[A-Za-z0-9_./,-]+$ ]] || { echo "$v holds characters this script will not pass"; exit 2; }
    PASS+=("$v=${!v}")
  fi
done
CM=(-o ControlMaster=auto -o "ControlPath=$HOME/.ssh/cm-e111-%r@%h:%p" -o ControlPersist=10m)
remote() {   # remote <command line>: on the login node over the shared connection, the job script on stdin
  if [ "$DRY" = 1 ]; then echo "DRY: $1" >&2; echo "$((1001 + ${#JOBS[@]}))"; return 0; fi
  ssh "${CM[@]}" aicr "$1" < exp/jobs/e111.sh
}
JOBS=()
LAST=
STARTED=0
[ -z "$FROM" ] && STARTED=1
sub() {   # sub <mode> <sbatch options...>: one sbatch on the login node, the job script on stdin; sets LAST
  local mode=$1; shift
  local id dep=()
  if [ "$STARTED" = 0 ]; then
    [ "$mode" = "$FROM" ] && STARTED=1 || return 0
  fi
  if [ -n "$LAST" ]; then dep=(--dependency=afterok:"$LAST" --kill-on-invalid-dep=yes); fi
  id=$(remote "E111_MODE=$mode E111_SHA=$SHA ${PASS[*]:-} E111_JOBS='${JOBS[*]:-}' sbatch --parsable $* ${dep[*]:-}")
  [[ "$id" =~ ^[0-9]+$ ]] || { echo "sbatch did not return a job id for $mode: $id"; exit 1; }
  JOBS+=("$id")
  LAST=$id
  echo "$mode: $id"
}
pack() {   # pack <mode>: after the last job, whatever its outcome; Slurm mail at its end
  local id
  id=$(remote "E111_MODE=$1 E111_SHA=$SHA ${PASS[*]:-} E111_JOBS='${JOBS[*]:-}' sbatch --parsable -J e111$1 -p cpu -c 1 --mem=4G -t 00:15:00 --mail-type=END,FAIL --dependency=afterany:$LAST")
  [[ "$id" =~ ^[0-9]+$ ]] || { echo "sbatch did not return a job id for $1: $id"; exit 1; }
  JOBS+=("$id")
  echo "$1: $id"
}
echo "submitting $PHASE${FROM:+ from $FROM} at $SHA (one Duo push for the connection)"
if [ "$PHASE" = phase1 ]; then
  sub env -J e111env -p cpu -c 4 --mem=32G -t 01:00:00
  sub select -J e111sel -p cpu -c 4 --mem=16G -t 00:30:00
  sub prepare -J e111prep -p cpu -c 32 --mem=64G -t 08:00:00
  sub gate -J e111gate -p b200-devel -q interactive --gpus=1 -c 8 --mem=64G -t 02:30:00
  sub gatecheck -J e111gchk -p cpu -c 8 --mem=32G -t 01:00:00
  pack pack1
else
  sub predict -J e111pred -p b200-devel -q interactive --gpus=1 -c 8 --mem=64G -t 01:00:00
  sub read -J e111read -p cpu -c 8 --mem=32G -t 01:00:00
  sub units -J e111units -p cpu -c 4 --mem=16G -t 00:30:00
  sub grade -J e111grade -p cpu -c 4 --mem=16G -t 00:30:00
  pack pack2
fi
[ "$DRY" = 1 ] || ssh "${CM[@]}" -O exit aicr 2>/dev/null || true
echo "submitted: ${JOBS[*]}; no polling. The pack job writes e111_out/fetch/e111_$PHASE.tar.gz and mails at its end; fetch it with one scp."
