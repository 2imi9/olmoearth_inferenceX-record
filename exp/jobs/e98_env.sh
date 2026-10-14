#!/bin/bash
#SBATCH -A p2026_0089_neu
#SBATCH -p cpu
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -J e98env
#SBATCH -o /home/qi_zim_neu/olmoearth_inferenceX/slurm/%x-%j.out
# exp98 stage 1 of 4 (exp/jobs/E98_README.md): the deployment environment for Ai2's FT-AWF model under olmoearth_run,
# kept apart from the repository's .venv, under /scratch/qi_zim_neu/olmoearth_inferenceX/deploy/:
#   python/                  CPython 3.11 from uv (olmoearth-runner 0.1.14 requires >=3.11,<3.12, PyPI metadata)
#   venv/                    olmoearth_projects (editable, pinned clone) + olmoearth-runner 0.1.14 and its pins
#   olmoearth_projects/      allenai/olmoearth_projects at f3c9b0c8, checked out detached
#   overrides.txt            the torch CUDA 12.8 build (see below)
#   checkpoint_path.txt      the FT-AWF model.ckpt in HF_HOME, the file olmoearth_run takes as --checkpoint_path
#   stage_spelling.txt       how one_stage's --stage names its values (read from its --help)
#   introspect/              olmoearth_run's own source, grepped for the env vars and paths it uses (not public on GitHub)
#   smoke/                   the synthetic writer smoke's dataset
#   versions.json            python, package versions, git shas, checkpoint sha256; also copied to exp/out/exp98/
# Submit from the Mac (the script travels on stdin; the job resets the checkout to origin/main, so push first):
#   ssh aicr 'sbatch --parsable' < exp/jobs/e98_env.sh
# E98_SHA=<full sha> checks out that commit (detached) instead of origin/main, so a push while the chain waits does not
# change what it runs; E98_REBUILD=1 rebuilds the venv.
#
# Versions, and the two places they differ from Ai2's uv.lock at f3c9b0c8 (olmoearth-runner 0.1.12, rslearn 0.0.23,
# torch 2.7.1 from PyPI):
#  - olmoearth-runner 0.1.14, which pins rslearn==0.0.27, olmoearth-pretrain==0.0.2, torch==2.7.1, numpy==1.26.4
#    (its requires_dist on PyPI). rslearn 0.0.23 has no SegmentationTask.output_probs; 0.0.27 has it
#    (rslearn/train/tasks/segmentation.py:44). Between the two, rslearn/models/olmoearth_pretrain/model.py changed only
#    a warning's text (GitHub compare v0.0.23...v0.0.27), and per_period_mosaic_reverse_time_order defaults to True,
#    the old order (rslearn/config/dataset.py:351 @ v0.0.27), so the 12 timesteps reach the model in the same order.
#  - torch 2.7.1+cu128 instead of PyPI's 2.7.1, whose CUDA 12.6 build (the runner pins nvidia-*-cu12 12.6.x) has no
#    kernels for the Blackwell GPUs of rtx-batch (RTX PRO 6000) and b200-batch (B200). Same torch release, so the
#    nvidia-* overrides are the set torch 2.7.1+cu128 resolves to in this repository's uv.lock. e98_predict.sh checks
#    that the GPU's compute capability is in torch.cuda.get_arch_list() before it runs anything.
set -euo pipefail
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
OUT=$REPO/exp/out/exp98
export HF_HOME=$SCRATCH/hf
export UV_CACHE_DIR=$SCRATCH/uv-cache
export UV_PYTHON_INSTALL_DIR=$DEPLOY/python
export UV_PYTHON_BIN_DIR=$DEPLOY/python/bin   # keeps uv from linking python3.11 into the shared ~/.local/bin
export PYTHONUNBUFFERED=1
export PATH="$HOME/.local/bin:$PATH"
export WANDB_MODE=disabled

OEP_URL=https://github.com/allenai/olmoearth_projects.git
OEP_SHA=f3c9b0c89c7670b3525647dda4c14b76245d682d
RUNNER_VERSION=0.1.14
RSLEARN_VERSION=0.0.27      # olmoearth-runner 0.1.14 pins it
PRETRAIN_VERSION=0.0.2      # idem
TORCH_VERSION=2.7.1+cu128
TORCHVISION_VERSION=0.22.1+cu128
CKPT_REPO=allenai/OlmoEarth-v1-FT-AWF-Base
CKPT_REV=a347b1546ab881c92aa125400ed5acc8126394ca   # the revision exp89 pinned (exp/exp89_finetuned_checkpoints.py:284)
CKPT_FILE=model.ckpt
CKPT_SHA256=1036b76bf7dfde819479dfa02507634f59a5637f14d4071bc3bf18f4d3d4bab2
CKPT_BYTES=1041443421
export E98_DEPLOY=$DEPLOY E98_OEP_SHA=$OEP_SHA E98_CKPT_REPO=$CKPT_REPO E98_CKPT_REV=$CKPT_REV \
  E98_CKPT_FILE=$CKPT_FILE E98_CKPT_SHA256=$CKPT_SHA256 E98_CKPT_BYTES=$CKPT_BYTES E98_REPO=$REPO

echo "== $(date -Is) job ${SLURM_JOB_ID:-none} on $(hostname): exp98 env, commit $(git rev-parse HEAD) =="
mkdir -p "$DEPLOY" "$OUT" "$DEPLOY/introspect"
command -v uv >/dev/null || { echo "uv not found at ~/.local/bin"; exit 1; }
uv --version

echo "== olmoearth_projects at $OEP_SHA =="
CLONE=$DEPLOY/olmoearth_projects
if [ -d "$CLONE/.git" ]; then
  [ "$(git -C "$CLONE" rev-parse HEAD)" = "$OEP_SHA" ] || { echo "$CLONE is not at $OEP_SHA; remove it and resubmit"; exit 1; }
  git -C "$CLONE" diff --quiet || { echo "$CLONE has local changes; remove it and resubmit"; exit 1; }
else
  rm -rf "$CLONE"
  git init -q "$CLONE"
  git -C "$CLONE" fetch -q --depth 1 "$OEP_URL" "$OEP_SHA"
  git -C "$CLONE" checkout -q --detach FETCH_HEAD
fi
[ "$(git -C "$CLONE" rev-parse HEAD)" = "$OEP_SHA" ] || { echo "clone is not at $OEP_SHA"; exit 1; }

echo "== the AWF originals, against their sha256 at $OEP_SHA =="
AWF=$CLONE/olmoearth_run_data/awf
sha256sum -c <<EOF
7da20e92b51741f3a2c61137e305bd5a8b59938a107213372099886897751b4b  $AWF/dataset.json
08e0432044c5c67c683372c85c632eb31085b6ce8b4dae49b740475e5b37a7ea  $AWF/model.yaml
1e1c8f709aadd86f84795723d35e07fc0aa79cef6e076c9a75d1ebec36d705ba  $AWF/olmoearth_run.yaml
fee3ce01008d26c6f3c239e68be70851035eb157d5da70bd72cb46dbc8623784  $AWF/prediction_request_geometry.geojson
EOF

echo "== no .env on main.py's path =="
# olmoearth_projects/main.py:35 calls dotenv.load_dotenv(), which looks for a .env from main.py's directory up to /.
# The run must not pick one up; the job only tests that none exists and never reads one.
d=$CLONE/olmoearth_projects
while :; do
  if [ -e "$d/.env" ]; then echo "a .env exists in $d, which main.py's load_dotenv would read; move it away"; exit 2; fi
  [ "$d" = / ] && break
  d=$(dirname "$d")
done

echo "== Python 3.11 and the venv =="
uv python install 3.11
PY=$DEPLOY/venv/bin/python
if [ "${E98_REBUILD:-0}" = 1 ]; then rm -rf "$DEPLOY/venv"; fi
if [ ! -x "$PY" ]; then uv venv -q --python 3.11 "$DEPLOY/venv"; fi
"$PY" -c 'import sys; assert sys.version_info[:2] == (3, 11), sys.version; print(sys.version)'

cat > "$DEPLOY/overrides.txt" <<EOF
torch==$TORCH_VERSION
torchvision==$TORCHVISION_VERSION
nvidia-cublas-cu12==12.8.3.14
nvidia-cuda-cupti-cu12==12.8.57
nvidia-cuda-nvrtc-cu12==12.8.61
nvidia-cuda-runtime-cu12==12.8.57
nvidia-cudnn-cu12==9.7.1.26
nvidia-cufft-cu12==11.3.3.41
nvidia-cufile-cu12==1.13.0.11
nvidia-curand-cu12==10.3.9.55
nvidia-cusolver-cu12==11.7.2.55
nvidia-cusparse-cu12==12.5.7.53
nvidia-cusparselt-cu12==0.6.3
nvidia-nccl-cu12==2.26.2
nvidia-nvjitlink-cu12==12.8.61
nvidia-nvtx-cu12==12.8.55
EOF
uv pip install --python "$PY" \
  --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu128 \
  --index-strategy unsafe-best-match --override "$DEPLOY/overrides.txt" \
  -e "$CLONE" "olmoearth-runner==$RUNNER_VERSION"
uv pip freeze --python "$PY" > "$DEPLOY/freeze.txt"
# The overrides contradict the runner's own nvidia-* pins on purpose, so pip check reports them; it is recorded, not fatal.
uv pip check --python "$PY" > "$DEPLOY/pip_check.txt" 2>&1 || true
echo "pip check: $(wc -l < "$DEPLOY/pip_check.txt") lines in $DEPLOY/pip_check.txt"

"$PY" - <<EOF
import importlib.metadata as md, torch
want = {"olmoearth-runner": "$RUNNER_VERSION", "rslearn": "$RSLEARN_VERSION", "olmoearth-pretrain": "$PRETRAIN_VERSION",
        "torch": "$TORCH_VERSION", "torchvision": "$TORCHVISION_VERSION"}
bad = {k: (md.version(k), v) for k, v in want.items() if md.version(k) != v}
assert not bad, f"installed (got, wanted): {bad}"
assert torch.version.cuda == "12.8", torch.version.cuda
print("versions as pinned; torch", torch.__version__, "CUDA", torch.version.cuda, "archs", torch.cuda.get_arch_list())
import olmoearth_run.runner.local.predict_runner, olmoearth_projects.olmoearth_run.olmoearth_run  # noqa: F401
EOF

echo "== rslearn 0.0.27's PlanetaryComputer.get_item_by_name, fixed as rslearn 0.0.29 fixes it =="
# In rslearn 0.0.27 and 0.0.28, PlanetaryComputer(DirectMaterializeDataSource, StacDataSource) finds
# DirectMaterializeDataSource.get_item_by_name, which raises NotImplementedError, before StacDataSource's, so the first
# direct materialize fails (prepare job 1243637, after 34 min of prepare). rslearn 0.0.29 removed that method from
# DirectMaterializeDataSource (GitHub compare v0.0.28...v0.0.29, rslearn/data_sources/direct_materialize_data_source.py).
# olmoearth-runner 0.1.14 pins 0.0.27, and Ai2's own lock (runner 0.1.12, rslearn 0.0.23) predates both the bug and
# output_probs, so the one removal is applied here, in this venv only, at interpreter start through a .pth file, which
# also runs in every worker process olmoearth_run spawns.
SITE=$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
cat > "$SITE/e98_rslearn_fix.py" <<'EOF'
"""exp98: rslearn 0.0.29's fix of PlanetaryComputer.get_item_by_name, applied to rslearn 0.0.27 (see e98_env.sh)."""
import importlib.metadata

if importlib.metadata.version("rslearn") in ("0.0.27", "0.0.28"):
    from rslearn.data_sources import direct_materialize_data_source as _dm

    if "get_item_by_name" in vars(_dm.DirectMaterializeDataSource):
        del _dm.DirectMaterializeDataSource.get_item_by_name
EOF
echo "import e98_rslearn_fix" > "$SITE/e98_rslearn_fix.pth"
"$PY" - <<'EOF'
from rslearn.data_sources.direct_materialize_data_source import DirectMaterializeDataSource
from rslearn.data_sources.planetary_computer import PlanetaryComputer, Sentinel2
from rslearn.data_sources.stac import StacDataSource
assert "get_item_by_name" not in vars(DirectMaterializeDataSource), "the .pth fix did not run"
assert Sentinel2.get_item_by_name is StacDataSource.get_item_by_name, Sentinel2.__mro__
assert PlanetaryComputer.get_item_by_name is StacDataSource.get_item_by_name
print("fix applied: PlanetaryComputer and Sentinel2 use StacDataSource.get_item_by_name")
EOF
cat > "$DEPLOY/e98_fix_workers.py" <<'EOF'
import multiprocessing as m


def f(_):
    from rslearn.data_sources.planetary_computer import Sentinel2
    from rslearn.data_sources.stac import StacDataSource
    return Sentinel2.get_item_by_name is StacDataSource.get_item_by_name


if __name__ == "__main__":
    for method in ("spawn", "forkserver"):
        with m.get_context(method).Pool(2) as p:
            assert all(p.map(f, range(2))), method
    print("fix holds in spawn and forkserver workers")
EOF
"$PY" "$DEPLOY/e98_fix_workers.py"

echo "== the FT-AWF checkpoint =="
"$PY" - <<'EOF'
import hashlib, os
from huggingface_hub import hf_hub_download
p = hf_hub_download(repo_id=os.environ["E98_CKPT_REPO"], filename=os.environ["E98_CKPT_FILE"],
                    revision=os.environ["E98_CKPT_REV"])
real = os.path.realpath(p)
size = os.path.getsize(real)
h = hashlib.sha256()
with open(real, "rb") as f:
    for chunk in iter(lambda: f.read(1 << 24), b""):
        h.update(chunk)
assert size == int(os.environ["E98_CKPT_BYTES"]), f"{real}: {size} bytes"
assert h.hexdigest() == os.environ["E98_CKPT_SHA256"], f"{real}: sha256 {h.hexdigest()}"
with open(os.path.join(os.environ["E98_DEPLOY"], "checkpoint_path.txt"), "w") as f:
    f.write(p + "\n")
print("checkpoint", p, size, "bytes, sha256 as pinned")
EOF

echo "== the checkpoint's head, and OlmoEarth-v1-Base's architecture (fetched into HF_HOME for the encoder) =="
"$PY" - <<'EOF'
import os, torch
from olmoearth_pretrain.model_loader import ModelID, load_model_from_id
p = open(os.path.join(os.environ["E98_DEPLOY"], "checkpoint_path.txt")).read().strip()
ck = torch.load(p, map_location="cpu", weights_only=False)
sd = ck["state_dict"]
heads = {k: tuple(v.shape) for k, v in sd.items() if k.startswith("model.decoders.segment.")}
print("checkpoint:", len(sd), "tensors; decoder", heads)
assert any(s[0] == 10 for s in heads.values()), "no 10-channel decoder tensor"
m = load_model_from_id(ModelID.OLMOEARTH_V1_BASE)
print("OlmoEarth-v1-Base loaded:", sum(x.numel() for x in m.parameters()), "parameters")
EOF

echo "== smoke: the e98 model.yaml's task and writer, parsed and run by the deployment's own packages =="
rm -rf "$DEPLOY/smoke"; mkdir -p "$DEPLOY/smoke"
E98_MODEL_YAML=$REPO/exp/jobs/e98_config/model.yaml E98_DATASET_JSON=$REPO/exp/jobs/e98_config/dataset.json \
E98_SCL_JSON=$REPO/exp/jobs/e98_config/scl_dataset.json "$PY" - <<'EOF'
import json, os, shutil
import jsonargparse, numpy as np, rasterio, torch, yaml
from rasterio.crs import CRS
from rslearn.utils.jsonargparse import init_jsonargparse
init_jsonargparse()
from rslearn.config import DatasetConfig
from rslearn.dataset import Window
from rslearn.train.prediction_writer import RslearnWriter
from rslearn.train.tasks.task import Task
from rslearn.utils.geometry import Projection

cfg = yaml.safe_load(open(os.environ["E98_MODEL_YAML"]))
for name in ("E98_DATASET_JSON", "E98_SCL_JSON"):        # both must validate as rslearn 0.0.27 dataset configs
    DatasetConfig.model_validate(json.load(open(os.environ[name])))

# the task: SegmentationTask with output_probs returns the (10, H, W) softmax as it is
p = jsonargparse.ArgumentParser()
p.add_argument("--task", type=Task)
task = p.instantiate_classes(p.parse_object({"task": cfg["data"]["init_args"]["task"]})).task
seg = task.tasks["segment"]
assert seg.output_probs is True and seg.prob_scales is None and seg.output_class_idx is None
x = torch.softmax(torch.randn(10, 16, 16), dim=0)
out = seg.process_output(x, None)
assert out.shape == (10, 16, 16) and np.allclose(out, x.numpy()), out.shape
print("task: output_probs gives", out.shape, out.dtype)

# the writer: layer_config parsed from YAML, then one synthetic 32 x 32 window written through it
ds = os.path.join(os.environ["E98_DEPLOY"], "smoke", "ds")
os.makedirs(ds)
shutil.copy(os.environ["E98_DATASET_JSON"], os.path.join(ds, "config.json"))
cbs = [c for c in cfg["trainer"]["callbacks"] if c["class_path"] == "rslearn.train.prediction_writer.RslearnWriter"]
assert len(cbs) == 1
init = dict(cbs[0]["init_args"], path=ds, output_layer="output")   # olmoearth_run fills both in the real run
p = jsonargparse.ArgumentParser()
p.add_argument("--writer", type=RslearnWriter)
w = p.instantiate_classes(p.parse_object({"writer": {"class_path": cbs[0]["class_path"], "init_args": init}})).writer
bs = w.layer_config.band_sets
assert len(bs) == 1 and bs[0].bands == [f"p{i}" for i in range(10)] and bs[0].dtype.value == "float32", bs
win = Window(storage=w.dataset_storage, group="smoke", name="w0", projection=Projection(CRS.from_epsg(3857), 10, -10),
             bounds=(0, 0, 32, 32), time_range=None)
win.save()
probs = torch.softmax(torch.randn(10, 32, 32), dim=0).numpy().astype(np.float32)
w.process_output(win, 0, 1, (0, 0, 32, 32), probs)
tif = os.path.join(ds, "windows", "smoke", "w0", "layers", "output", "_".join(bs[0].bands), "geotiff.tif")
with rasterio.open(tif) as src:
    a = src.read()
assert a.shape == (10, 32, 32) and a.dtype == np.float32 and np.array_equal(a, probs), (a.shape, a.dtype)
assert os.path.exists(os.path.join(ds, "windows", "smoke", "w0", "layers", "output", "completed"))
print("writer: 10 float32 bands at layers/output/" + "_".join(bs[0].bands) + "/geotiff.tif, equal to the input")
EOF

echo "== one_stage's stage names =="
cd "$DEPLOY" || exit 1
if ! help=$("$PY" -m olmoearth_projects.main olmoearth_run one_stage --help 2>&1); then
  echo "one_stage --help failed:"; printf '%s\n' "$help" | tail -n 30; exit 1
fi
if grep -q 'BUILD_DATASET' <<<"$help"; then echo BUILD_DATASET RUN_INFERENCE POSTPROCESS COMBINE > "$DEPLOY/stage_spelling.txt"
elif grep -q 'build_dataset' <<<"$help"; then echo build_dataset run_inference postprocess combine > "$DEPLOY/stage_spelling.txt"
else printf '%s\n' "$help" | tail -n 30; echo "one_stage --help names no stage"; exit 1; fi
echo "stage values: $(cat "$DEPLOY/stage_spelling.txt")"
cd "$REPO" || exit 1

echo "== olmoearth_run, introspected (its source is on PyPI only) =="
RUNNER_DIR=$("$PY" -c 'import olmoearth_run, os; print(os.path.dirname(olmoearth_run.__file__))')
find "$RUNNER_DIR" -name '*.py' | sort > "$DEPLOY/introspect/olmoearth_run_files.txt"
grep -rn --include='*.py' -E 'environ|getenv|DATASET_PATH|PREDICTION_OUTPUT_LAYER|NUM_WORKERS|TRAINER_DATA_PATH|EXTRA_FILES_PATH|WANDB|dataset\.json|config\.json|model\.yaml|cpu_count|workers|materialize|ingest|prepare|output_layer|results' \
  "$RUNNER_DIR" > "$DEPLOY/introspect/olmoearth_run_grep.txt" || true
echo "$(wc -l < "$DEPLOY/introspect/olmoearth_run_files.txt") modules; $(wc -l < "$DEPLOY/introspect/olmoearth_run_grep.txt") matching lines in $DEPLOY/introspect/olmoearth_run_grep.txt"

echo "== versions.json =="
"$PY" - <<'EOF'
import hashlib, importlib.metadata as md, json, os, platform, socket, subprocess, sys
from datetime import datetime, timezone
d = os.environ["E98_DEPLOY"]
def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()
def git(path):
    return subprocess.run(["git", "-C", path, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
pkgs = ["olmoearth-runner", "rslearn", "olmoearth-pretrain", "olmoearth_projects", "torch", "torchvision", "lightning",
        "jsonargparse", "numpy", "rasterio", "huggingface-hub", "planetary-computer", "pystac-client", "pydantic"]
ckpt = open(os.path.join(d, "checkpoint_path.txt")).read().strip()
cfg = os.path.join(os.environ["E98_REPO"], "exp", "jobs", "e98_config")
rec = {
    "written": datetime.now(timezone.utc).isoformat(timespec="seconds"), "job": os.environ.get("SLURM_JOB_ID"),
    "host": socket.gethostname(), "python": sys.version.split()[0], "platform": platform.platform(),
    "uv": subprocess.run(["uv", "--version"], capture_output=True, text=True).stdout.strip(),
    "packages": {p: md.version(p) for p in pkgs},
    "git": {"olmoearth_projects": git(os.path.join(d, "olmoearth_projects")), "olmoearth_inferenceX": git(os.environ["E98_REPO"])},
    "overrides": open(os.path.join(d, "overrides.txt")).read().split(),
    "checkpoint": {"repo": os.environ["E98_CKPT_REPO"], "revision": os.environ["E98_CKPT_REV"],
                   "file": os.environ["E98_CKPT_FILE"], "path": ckpt, "bytes": os.path.getsize(os.path.realpath(ckpt)),
                   "sha256": sha(os.path.realpath(ckpt))},
    "e98_config_sha256": {f: sha(os.path.join(cfg, f)) for f in sorted(os.listdir(cfg)) if not f.endswith(".md")},
    "stage_values": open(os.path.join(d, "stage_spelling.txt")).read().split(),
}
with open(os.path.join(d, "versions.json"), "w") as f:
    json.dump(rec, f, indent=1)
print(json.dumps({k: rec[k] for k in ("python", "packages", "git")}, indent=1))
EOF
cp "$DEPLOY/versions.json" "$OUT/e98_versions.json"
echo "== $(date -Is) done: $DEPLOY/versions.json, $OUT/e98_versions.json =="
