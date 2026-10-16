#!/bin/bash
# exp111's local smoke (the Mac; no network, no imagery, no model, no TimeSync file): both environments on synthetic
# inputs, the grader's smoke (synthetic units with known answers, interval cases, a plain-numpy bootstrap and the known
# answer on the committed exp99/exp103 units), and the job scripts' logic (usage, redo, the A' budget, the failure
# allowance, e111.sh's own prepare_root run in its order with rslearn stood in for (a partial window, a failed window,
# an SCL failure, the allowance's sum, the allowance at 0, reruns that quarantine again or fetch whole), e111.sh's own
# gate mode (a gate that dies part way, records of two jobs), a partly purged deployment, the gate's two commits, the
# submit chain in dry run, the rule consistency between e111.sh, e111_submit.sh, the modules' import graph and, with
# PAGE, the page). Every record a step writes goes under the name the cluster's $REC gives it, and the pack jobs' own
# key check (exp111_jobs.py pack-check) runs over all of them at the end. RECORD_PY is the record repo's Python (oe_inferencex, rasterio, shapely, pyproj); RSLEARN_PY a Python with
# rslearn 0.1.17 (an isolated uv venv; nothing is installed by this script). EXP is the exp/ directory holding
# exp111_blind_transfer.py, exp111_grade.py, exp111_jobs.py, jobs/e111.sh, jobs/e111_submit.sh and the record's
# modules; OUT a scratch directory outside every checkout; PAGE (optional) the preregistration page.
#   RECORD_PY=... RSLEARN_PY=... EXP=.../exp OUT=... [PAGE=.../blind_transfer.md] bash exp/jobs/e111_local_smoke.sh
set -euo pipefail
: "${RECORD_PY:?}" "${RSLEARN_PY:?}" "${EXP:?}" "${OUT:?}"
PAGE=${PAGE:-}
J=$EXP/exp111_jobs.py
B=$EXP/exp111_blind_transfer.py
GR=$EXP/exp111_grade.py
E111_SH=$EXP/jobs/e111.sh
SUBMIT_SH=$EXP/jobs/e111_submit.sh
mkdir -p "$OUT"
P=$OUT/smoke_private
R=$OUT/rec                                  # the records, as the cluster's $REC names them
rm -rf "$R"
mkdir -p "$R"
export PYTHONWARNINGS=ignore HF_HUB_OFFLINE=1 WANDB_MODE=disabled HF_HOME=$OUT/hf_empty NUM_WORKERS=0
export TRAINER_DATA_PATH=$OUT/trainer
pass=0; fail=0
step() {   # step <name> <command...>: run, record pass or fail, keep going
  local name=$1; shift
  if "$@" > "$OUT/step_$name.log" 2>&1; then echo "PASS $name"; pass=$((pass + 1)); else echo "FAIL $name (see step_$name.log)"; tail -n 5 "$OUT/step_$name.log"; fail=$((fail + 1)); fi
}
refused() {   # refused <message> <command...>: the command fails and says <message>
  local msg=$1 o
  shift
  if o=$("$@" 2>&1); then return 1; fi
  grep -q -- "$msg" <<< "$o"
}

echo "== record environment: select/draw, reruns, window geometry, deployment windows, synthetic outputs, reader, gate, draw check, units =="
# as a draft page (the page is frozen from 16 October 2026; the module is guarded since the gate, so the status is set here)
step record_smoke "$RECORD_PY" -I -c "import sys; sys.path.insert(0, sys.argv[1]); import exp111_blind_transfer as b
with b._pretend_status('draft'):
    sys.exit(0 if b.smoke(sys.argv[2])['ok'] else 1)" "$EXP" "$OUT"
echo "== the grader: synthetic units (a passing and a failing case for each bar, the interval cases), a plain-numpy bootstrap, the 271 =="
step grade_smoke "$RECORD_PY" -I -W ignore "$GR" --smoke --out-dir "$OUT"

echo "== rslearn 0.1.17 environment =="
step no_runner "$RSLEARN_PY" -I "$J" no-runner
# the Mac's torch is another build: the core pins here; the job checks all five and must refuse this venv
step versions "$RSLEARN_PY" -I "$J" versions --core-only --record "$R/e111_versions_env.json"
step versions_full_refuses_local_torch "$RSLEARN_PY" -I -c "import subprocess,sys; sys.exit(0 if subprocess.run([sys.executable,'-I',sys.argv[1],'versions'],capture_output=True).returncode==1 else 1)" "$J"
SMOKE_SHA=1111111111111111111111111111111111111111
echo '{"device": {"name": "NVIDIA B200"}}' > "$OUT/gate_device.json"
step device_same_accepted "$RSLEARN_PY" -I "$J" device --name "NVIDIA B200" --commit "$SMOKE_SHA" --expect-gate "$OUT/gate_device.json" \
  --record "$R/e111_device_gate.json"
step device_records_commit "$RSLEARN_PY" -I -c "import json,sys; r=json.load(open(sys.argv[1])); sys.exit(0 if r['commit']==sys.argv[2] else 1)" "$R/e111_device_gate.json" "$SMOKE_SHA"
step device_short_commit_refused "$RSLEARN_PY" -I -c "import subprocess,sys; sys.exit(0 if subprocess.run([sys.executable,'-I',sys.argv[1],'device','--name','NVIDIA B200','--commit','abc123'],capture_output=True).returncode==2 else 1)" "$J"
step device_other_refused "$RSLEARN_PY" -I -c "import subprocess,sys; sys.exit(0 if subprocess.run([sys.executable,'-I',sys.argv[1],'device','--name','NVIDIA RTX PRO 6000 Blackwell Server Edition','--expect-gate',sys.argv[2]],capture_output=True).returncode==2 else 1)" "$J" "$OUT/gate_device.json"
step device_without_gpu_refused "$RSLEARN_PY" -I -c "import subprocess,sys; sys.exit(0 if subprocess.run([sys.executable,'-I',sys.argv[1],'device'],capture_output=True).returncode==2 else 1)" "$J"
step check_configs "$RSLEARN_PY" -I "$J" check-configs --record "$R/e111_configs.json"
step crop_lattice "$RSLEARN_PY" -I "$J" crop-lattice --sizes 124 128 1024
for r in gate23 gate17 blind17; do
  step "check_dataset_$r" "$RSLEARN_PY" -I "$J" check-dataset --root "$P/roots/$r" --units "$P/units_$r.json" --workers 2 \
    --record "$R/e111_${r}_dataset_check.json"
  step "precheck_$r" "$RSLEARN_PY" -I "$J" precheck --root "$P/roots/$r" --units "$P/units_$r.json" --workers 2 \
    --record "$R/e111_${r}_precheck.json"
  for v in v1 v12; do
    step "check_output_${r}_$v" "$RSLEARN_PY" -I "$J" check-output --root "$P/roots/$r" --layer "output_$v" \
      --record "$R/e111_${r}_${v}_output.json"
  done
  step "inventory_$r" "$RSLEARN_PY" -I "$J" inventory --root "$P/roots/$r" --units "$P/units_$r.json" \
    --record "$R/e111_${r}_inventory.json"
done
export DATASET_PATH=$P/roots/gate17 PREDICTION_OUTPUT_LAYER=output_v1
step parse_model "$RSLEARN_PY" -I "$J" parse-model --record "$R/e111_parse_model.json"

echo "== synthetic deployment, items, reflectance, prepare logs, a 201-window root: sidecar, item identity, A' config, time range, the failure allowance =="
"$RSLEARN_PY" -I - "$P" "$OUT" "$EXP" <<'EOF'
import json, os, shutil, sys
import numpy as np, rasterio
from rasterio.transform import from_origin
P, OUT, EXP = sys.argv[1], sys.argv[2], sys.argv[3]
units = json.load(open(os.path.join(P, "units_gate23.json")))["units"]
root = os.path.join(P, "roots", "gate23")
dep = os.path.join(OUT, "deploy_run", "rootdir")
items = lambda tag: [{"layer_name": "sentinel2", "serialized_item_groups": [[{"name": f"S2X_{tag}_{g}"}] for g in range(12)],
                      "materialized": True}]
for k, u in enumerate(units):
    wd = os.path.join(root, "windows", "e111", u["name"])
    json.dump(items("same" if k else "other"), open(os.path.join(wd, "items.json"), "w"))
    b = u["bounds"]; H, W = b[3] - b[1], b[2] - b[0]
    for g in range(12):
        gd = os.path.join(wd, "layers", "sentinel2" if g == 0 else f"sentinel2.{g}", "B01_B02")
        os.makedirs(gd, exist_ok=True)
        with rasterio.open(os.path.join(gd, "geotiff.tif"), "w", driver="GTiff", height=H, width=W, count=2,
                           dtype="uint16", crs=f"EPSG:{u['epsg']}", transform=from_origin(b[0] * 10, -b[1] * 10, 10, 10)) as d:
            d.write(np.full((2, H, W), 1000, np.uint16))
    # the deployment's 1,024-px cell window holding the unit (one per distinct cell), with the 'same' items
    X0, Y0 = u["cell"]
    dw = os.path.join(dep, "windows", "group_0", f"cell_{X0}_{Y0}")
    os.makedirs(dw, exist_ok=True)
    json.dump({"projection": {"crs": f"EPSG:{u['epsg']}", "x_resolution": 10, "y_resolution": -10},
               "bounds": [X0, Y0, X0 + 1024, Y0 + 1024], "time_range": ["2023-01-01T00:00:00+00:00", "2023-12-31T00:00:00+00:00"],
               "options": {}}, open(os.path.join(dw, "metadata.json"), "w"))
    json.dump(items("same"), open(os.path.join(dw, "items.json"), "w"))
json.dump({"roots": [{"root_id": "root_0", "root": "rootdir"}]}, open(os.path.join(OUT, "deploy_inventory.json"), "w"))
# rslearn's summary lines: pass 1 (3 of 1,468 failed in prepare, 0.2%), pass 3 within the allowance (2 + 3 = 5 of
# 1,468: 0.34% <= 0.5%) and above it (4 + 5 = 9: 0.61%)
logs = {"prepare.log": "prepared=1440, skipped=0, rejected=25, failed=3",
        "materialize.log": "materialized=1443, failed=0",
        "prepare3_within.log": "prepared=0, skipped=1441, rejected=25, failed=2",
        "materialize3_within.log": "materialized=0, failed=3",
        "prepare3_above.log": "prepared=0, skipped=1439, rejected=25, failed=4",
        "materialize3_above.log": "materialized=0, failed=5"}
for fn, line in logs.items():
    head = "Materialize" if "materialize" in fn else "Prepare"
    open(os.path.join(OUT, fn), "w").write(f"INFO === {head} Summary ===\nINFO Total windows requested: 1468, "
                                           f"duration: 12.0s\nINFO   Layer sentinel2: {line}\n")
# a smoke root as prepare leaves it: twelve groups newest first (group_time_ranges), 12-band uint16 reflectance
from datetime import datetime, timedelta, timezone
sm = os.path.join(OUT, "smoke2")
shutil.rmtree(sm, ignore_errors=True)
end = datetime(2023, 12, 31, tzinfo=timezone.utc)
for u in units[:2]:
    wd = os.path.join(sm, "windows", "e111", u["name"])
    os.makedirs(wd)
    shutil.copyfile(os.path.join(root, "windows", "e111", u["name"], "metadata.json"), os.path.join(wd, "metadata.json"))
    gtr = [[(end - timedelta(days=30 * (g + 1))).isoformat(), (end - timedelta(days=30 * g)).isoformat()] for g in range(12)]
    json.dump([{"layer_name": "sentinel2", "serialized_item_groups": [[{"name": f"S2X_{g}"}] for g in range(12)],
                "materialized": True, "group_time_ranges": gtr}], open(os.path.join(wd, "items.json"), "w"))
    b = u["bounds"]; H, W = b[3] - b[1], b[2] - b[0]
    for g in range(12):
        gd = os.path.join(wd, "layers", "sentinel2" if g == 0 else f"sentinel2.{g}", "B01_B02_B03_B04_B05_B06_B07_B08_B8A_B09_B11_B12")
        os.makedirs(gd)
        open(os.path.join(os.path.dirname(gd), "completed"), "w").close()
        with rasterio.open(os.path.join(gd, "geotiff.tif"), "w", driver="GTiff", height=H, width=W, count=12,
                           dtype="uint16", crs=f"EPSG:{u['epsg']}", transform=from_origin(b[0] * 10, -b[1] * 10, 10, 10)) as d:
            d.write(np.full((12, H, W), 900, np.uint16))
shutil.copyfile(os.path.join(root, "config.json"), os.path.join(sm, "config.json"))
# two 201-window roots for the failure allowance (metadata and completion markers only; precheck reads no pixel): one
# with a single partial window (1/201 = 0.498% <= 0.5%), one with two (0.995%); the quarantine of an earlier run goes
shutil.rmtree(os.path.join(OUT, "allowance"), ignore_errors=True)
for name, n_partial in (("allow1", 1), ("allow2", 2)):
    ar = os.path.join(OUT, "allowance", name)
    shutil.rmtree(ar, ignore_errors=True)
    us = []
    for k in range(201):
        n = f"b{k:04d}"
        wd = os.path.join(ar, "windows", "e111", n)
        os.makedirs(wd)
        x0 = 1024 * (k % 20)
        json.dump({"projection": {"crs": "EPSG:32737", "x_resolution": 10.0, "y_resolution": -10.0},
                   "bounds": [x0, 900000, x0 + 124, 900124],
                   "time_range": ["2017-01-01T00:00:00+00:00", "2017-12-31T00:00:00+00:00"], "options": {}},
                  open(os.path.join(wd, "metadata.json"), "w"))
        groups = 5 if k < n_partial else 12
        for g in range(groups):
            gd = os.path.join(wd, "layers", "sentinel2" if g == 0 else f"sentinel2.{g}")
            os.makedirs(gd)
            open(os.path.join(gd, "completed"), "w").close()
        us.append({"name": n, "country": "kenya" if k % 2 else "tanzania", "stratum": "OWX"[k % 3]})
    shutil.copyfile(os.path.join(EXP, "jobs", "e111_config", "dataset.json"), os.path.join(ar, "config.json"))
    json.dump({"root": name, "year": 2017, "group": "e111", "units": us}, open(os.path.join(OUT, "allowance", f"units_{name}.json"), "w"))
print("synthetic deployment, logs and roots written")
EOF
step scl_sidecar "$RSLEARN_PY" -I "$J" scl-sidecar --root "$P/roots/gate23" --scl "$OUT/scl_sidecar_gate23"
step scl_sidecar_items "$RSLEARN_PY" -I -c "import json,glob,sys; f=sorted(glob.glob(sys.argv[1]+'/windows/*/*/items.json')); d=json.load(open(f[0])); sys.exit(0 if len(f)==3 and d[0]['layer_name']=='sentinel2_scl' and len(d[0]['serialized_item_groups'])==12 else 1)" "$OUT/scl_sidecar_gate23"
step scl_check "$RSLEARN_PY" -I "$J" scl-check --root "$P/roots/gate23" --scl "$P/scl/gate23" --record "$R/e111_gate23_scl_check.json"
step item_identity "$RSLEARN_PY" -I "$J" item-identity --root "$P/roots/gate23" --units "$P/units_gate23.json" \
  --inv "$OUT/deploy_inventory.json" --run "$OUT/deploy_run" --flags-out "$OUT/identity_flags.json" --record "$R/e111_identity_gate23.json"
step item_identity_counts "$RSLEARN_PY" -I -c "import json,sys; r=json.load(open(sys.argv[1])); sys.exit(0 if (r['identical'], r['differ']) == (2, 1) else 1)" "$R/e111_identity_gate23.json"
step aprime_config "$RSLEARN_PY" -I "$J" aprime-config --units "$P/units_gate23.json" --inv "$OUT/deploy_inventory.json" \
  --run "$OUT/deploy_run" --out-dir "$OUT/aprime" --record "$R/e111_aprime_config.json"
step aprime_config_parses "$RSLEARN_PY" -I "$J" parse-model --config "$OUT/aprime/model_aprime_r0.yaml" --model-ids OLMOEARTH_V1_BASE
step deploy_time_range "$RSLEARN_PY" -I "$J" deploy-time-range --inv "$OUT/deploy_inventory.json" --run "$OUT/deploy_run" --year 2023 \
  --record "$R/e111_deploy_time_range23.json"
step prepare_summary "$RSLEARN_PY" -I "$J" prepare-summary --log "$OUT/prepare.log" --max-failed-share 0.02 --record "$R/e111_gate23_prepare1.json"
step materialize_summary "$RSLEARN_PY" -I "$J" prepare-summary --log "$OUT/materialize.log" --max-failed-share 0 --record "$R/e111_gate23_materialize1.json"
step check_groups "$RSLEARN_PY" -I "$J" check-groups --root "$OUT/smoke2" --record "$R/e111_smoke2_groups.json"
step prepare_summary_refuses "$RSLEARN_PY" -I -c "import subprocess,sys; sys.exit(0 if subprocess.run([sys.executable,'-I',sys.argv[1],'prepare-summary','--log',sys.argv[2],'--max-failed-share','0.001']).returncode==1 else 1)" "$J" "$OUT/prepare.log"
# the failure allowance after pass 3: prepare's and materialize's failures summed against MAX_FAILED_SHARE
step pass3_prepare_within "$RSLEARN_PY" -I "$J" prepare-summary --log "$OUT/prepare3_within.log" --max-failed-share 0.005 --record "$R/e111_gate23_prepare3.json"
step pass3_summed_within_allowance "$RSLEARN_PY" -I "$J" prepare-summary --log "$OUT/materialize3_within.log" --max-failed-share 0.005 \
  --plus "$R/e111_gate23_prepare3.json" --record "$R/e111_gate23_materialize3.json"
step pass3_summed_is_5 "$RSLEARN_PY" -I -c "import json,sys; r=json.load(open(sys.argv[1])); sys.exit(0 if r['summary']['failed'] + r['failed_added_from_earlier_record'] == 5 else 1)" "$R/e111_gate23_materialize3.json"
step pass3_refused_at_allowance_0 refused '' "$RSLEARN_PY" -I "$J" prepare-summary --log "$OUT/materialize3_within.log" --max-failed-share 0 \
  --plus "$R/e111_gate23_prepare3.json"
"$RSLEARN_PY" -I "$J" prepare-summary --log "$OUT/prepare3_above.log" --max-failed-share 0.005 --record "$OUT/prepare3_above.json" > /dev/null
step pass3_summed_above_allowance_refused refused '' "$RSLEARN_PY" -I "$J" prepare-summary --log "$OUT/materialize3_above.log" \
  --max-failed-share 0.005 --plus "$OUT/prepare3_above.json"
A=$OUT/allowance
step precheck_partial_refused_at_0 refused "above the failure allowance" "$RSLEARN_PY" -I "$J" precheck --root "$A/allow1" --units "$A/units_allow1.json" \
  --workers 2 --quarantine "$A/quarantine"
step precheck_two_partial_refused_at_cap refused "above the failure allowance" "$RSLEARN_PY" -I "$J" precheck --root "$A/allow2" \
  --units "$A/units_allow2.json" --workers 2 --max-drop-share 0.005 --quarantine "$A/quarantine"
step precheck_refusal_moves_nothing bash -c '[ ! -e "$1/quarantine" ] && [ -e "$1/allow1/windows/e111/b0000/layers/sentinel2/completed" ] && [ -e "$1/allow2/windows/e111/b0001/layers/sentinel2/completed" ]' _ "$A"
step precheck_partial_quarantined_within_cap "$RSLEARN_PY" -I "$J" precheck --root "$A/allow1" --units "$A/units_allow1.json" \
  --workers 2 --max-drop-share 0.005 --quarantine "$A/quarantine" --record "$OUT/allow1_precheck.json"
step precheck_allowance_above_cap_refused "$RSLEARN_PY" -I -c "import subprocess,sys; sys.exit(0 if subprocess.run([sys.executable,'-I',sys.argv[1],'precheck','--root',sys.argv[2],'--units',sys.argv[3],'--max-drop-share','0.01','--quarantine',sys.argv[4]],capture_output=True).returncode==2 else 1)" "$J" "$A/allow2" "$A/units_allow2.json" "$A/quarantine"
step inventory_counts_quarantined_by_cell bash -c '"$1" -I "$2" inventory --root "$3/allow1" --units "$3/units_allow1.json" --quarantine "$3/quarantine" --record "$3/allow1_inventory.json" > /dev/null && "$1" -I -c "import json,sys; r=json.load(open(sys.argv[1])); t=r[\"total\"]; c=r[\"by_stratum\"][\"tanzania:O\"]; sys.exit(0 if (t[\"quarantined\"], t[\"none\"], t[\"with_12_groups\"], c[\"quarantined\"]) == (1, 1, 200, 1) else 1)" "$3/allow1_inventory.json"' _ "$RSLEARN_PY" "$J" "$A"
echo "== e111.sh's prepare_root itself, in its order (rslearn stood in for): allowance, SCL failures, quarantine, reruns =="
# A stand-in for rslearn (prepare, ingest, materialize of the root and of its SCL sidecar), driven by a scenario file:
# which windows a materialize leaves partial or empty, at which windows the SCL materialize fails. It writes rslearn's
# summary lines to the log e111.sh reads, the completion markers and small rasters (reflectance and SCL on one grid).
SEQ=$OUT/seq
rm -rf "$SEQ"
mkdir -p "$SEQ"
cat > "$SEQ/fake_rslearn.py" <<'PYEOF'
import json, os, sys
import numpy as np, rasterio
from rasterio.transform import from_origin
G = 12
def gdir(p, layer, k):
    return os.path.join(p, "layers", layer if k == 0 else f"{layer}.{k}")
def done(p, layer):
    return sum(os.path.exists(os.path.join(gdir(p, layer, k), "completed")) for k in range(G))
def raster(path, dtype, value):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", height=8, width=8, count=1, dtype=dtype, crs="EPSG:32737",
                       transform=from_origin(0, 0, 10, 10)) as d:
        d.write(np.full((1, 8, 8), value, dtype))
def windows(root):
    wd = os.path.join(root, "windows")
    return [(g, n, os.path.join(wd, g, n)) for g in sorted(os.listdir(wd)) for n in sorted(os.listdir(os.path.join(wd, g)))]
act = sys.argv[1]
if act == "init":                          # init <root> <units json> <n>: windows prepared with twelve item groups
    root, units, n = sys.argv[2], sys.argv[3], int(sys.argv[4])
    os.makedirs(root)
    us = []
    for k in range(n):
        name = f"b{k:04d}"
        wd = os.path.join(root, "windows", "e111", name)
        os.makedirs(wd)
        x0 = 1024 * (k % 20)
        json.dump({"projection": {"crs": "EPSG:32737", "x_resolution": 10.0, "y_resolution": -10.0},
                   "bounds": [x0, 900000, x0 + 124, 900124],
                   "time_range": ["2017-01-01T00:00:00+00:00", "2017-12-31T00:00:00+00:00"], "options": {}},
                  open(os.path.join(wd, "metadata.json"), "w"))
        json.dump([{"layer_name": "sentinel2", "serialized_item_groups": [[{"name": f"S2X_{k}_{g}"}] for g in range(G)],
                    "materialized": False}], open(os.path.join(wd, "items.json"), "w"))
        us.append({"name": name, "country": "kenya" if k % 2 else "tanzania", "stratum": "OWX"[k % 3]})
    json.dump({"root": os.path.basename(root), "year": 2017, "group": "e111", "units": us}, open(units, "w"))
    sys.exit(0)
tag, root, log, scen = sys.argv[2], sys.argv[3], sys.argv[4], json.load(open(sys.argv[5]))
ws = windows(root) if os.path.isdir(os.path.join(root, "windows")) else []
head = f"INFO Total windows requested: {len(ws)}, duration: 1.0s\n"
if "-prepare" in tag:
    line = f"INFO   Layer sentinel2: prepared=0, skipped={len(ws)}, rejected=0, failed=0\n"
elif "-ingest" in tag:
    line = ""
elif "-materialize" in tag:                # rslearn skips a window whose item group 0 is completed
    made = failed = 0
    for g, n, p in ws:
        if os.path.exists(os.path.join(gdir(p, "sentinel2", 0), "completed")):
            continue
        want = scen.get("input", {}).get(n, G)
        for k in range(want):
            raster(os.path.join(gdir(p, "sentinel2", k), "B01_B02", "geotiff.tif"), "uint16", 1000)
            open(os.path.join(gdir(p, "sentinel2", k), "completed"), "w").close()
        made += want == G
        failed += want < G
    line = f"INFO   Layer sentinel2: materialized={made}, failed={failed}\n"
elif tag.endswith("-scl"):
    made = failed = 0
    for g, n, p in ws:
        if os.path.exists(os.path.join(gdir(p, "sentinel2_scl", 0), "completed")):
            continue
        if n in scen.get("scl_fail", []):
            failed += 1
            continue
        for k in range(G):
            raster(os.path.join(gdir(p, "sentinel2_scl", k), "SCL", "geotiff.tif"), "uint8", 4)
            open(os.path.join(gdir(p, "sentinel2_scl", k), "completed"), "w").close()
        made += 1
    line = f"INFO   Layer sentinel2_scl: materialized={made}, failed={failed}\n"
else:
    sys.exit(f"unknown tag {tag}")
os.makedirs(os.path.dirname(log), exist_ok=True)
open(log, "w").write(head + line)
PYEOF
seq_case() {   # seq_case <case dir> <root name> <MAX_FAILED_SHARE> <job> <scenario json>: e111.sh's prepare_root on it
  bash -c '
    set -euo pipefail
    [ -s "$1" ] || { echo "seq_case: no e111.sh (E111_SH unset?)"; exit 2; }
    eval "$(sed -n "/^masked() /,/^}\$/p; /^set_tmp() /,/^}\$/p; /^prepare_root() /,/^}\$/p" "$1")"
    declare -F masked set_tmp prepare_root > /dev/null
    case_dir=$2 name=$3 MAX_FAILED_SHARE=$4 JOB=$5 TMPN=0
    TSHORT=$(mktemp -d /tmp/e111s.XXXXXX)   # short, as on the cluster: set_tmp refuses a TMPDIR a socket cannot sit under
    PRIV=$case_dir/priv REC=$case_dir/rec DEPLOY=$case_dir/none VENV=$case_dir/no_venv NCPU=2
    CONFIG=$7/jobs/e102_config/model.yaml
    JPY=("$8" -I "$7/exp111_jobs.py")
    echo "$6" > "$case_dir/scenario-$JOB.json"
    run_logged() {   # the stand-in: rslearn replaced by fake_rslearn.py, which writes the log prepare_root reads
      local tag=$1 root
      case "$tag" in *-scl) root=$PRIV/scl/$name ;; *) root=$PRIV/roots/$name ;; esac
      "$RSLEARN_PY" -I "$SEQ/fake_rslearn.py" run "$tag" "$root" "$PRIV/logs/$tag-$JOB.log" "$case_dir/scenario-$JOB.json"
    }
    mkdir -p "$REC" "$PRIV/logs"
    prepare_root "$name"
    echo "prepare_root returned"
  ' _ "$E111_SH" "$@" "$EXP" "$RSLEARN_PY"
}
seq_init() {   # seq_init <case dir> <root name> <windows>: a prepared root (items.json, twelve groups listed, none materialized)
  rm -rf "$1"; mkdir -p "$1/priv/roots"
  "$RSLEARN_PY" -I "$SEQ/fake_rslearn.py" init "$1/priv/roots/$2" "$1/priv/units_$2.json" "$3"
  cp "$EXP/jobs/e111_config/dataset.json" "$1/priv/roots/$2/config.json"
}
seq_rec() {   # seq_rec <python expression over r(record name)>: true or the step fails
  "$RSLEARN_PY" -I -c "import json,os,sys; R=sys.argv[1]; r=lambda n: json.load(open(os.path.join(R, n))); sys.exit(0 if ($2) else 1)" "$1" "$2"
}
export RSLEARN_PY SEQ EXP E111_SH P
export -f seq_case seq_init seq_rec
# (a) the review's case: a window a materialize leaves partial (5 of 12 groups), the allowance 0.005: it gets no
# sidecar, the precheck quarantines it, the SCL check compares the 200 windows with input, the inventory counts it.
# (rslearn skips a window whose group 0 is completed, so passes 2 and 3 do not count it: the precheck does, from the
# files.)
seq_init "$SEQ/mat" seqmat 201
step prepare_root_partial_within_allowance bash -c 'o=$(seq_case "$1/mat" seqmat 0.005 2001 "{\"input\": {\"b0000\": 5}}" 2>&1) && grep -q "^prepare_root returned$" <<< "$o" && ! grep -q Traceback <<< "$o"' _ "$SEQ"
step prepare_root_partial_records seq_rec "$SEQ/mat/rec" 'r("e111_seqmat_precheck.json")["ok"] and r("e111_seqmat_precheck.json")["filesystem"]["quarantined"] == 1 and r("e111_seqmat_scl_check.json")["ok"] and r("e111_seqmat_scl_check.json")["windows"] == 200 and r("e111_seqmat_scl_materialize.json")["failed_total"] == 0 and r("e111_seqmat_scl_materialize.json")["denominator"] == 201 and r("e111_seqmat_inventory.json")["total"]["quarantined"] == 1 and r("e111_seqmat_inventory.json")["total"]["with_12_groups"] == 200'
# the review's reproduction itself: the old sidecar rule (every prepared window) beside the partial window is what the
# new scl-check refuses as a count, never a KeyError
step scl_check_refuses_a_mismatch_without_traceback bash -c '
  set -euo pipefail
  T=$1/mismatch; rm -rf "$T"; mkdir -p "$T"; cp -R "$1/mat/priv/roots/seqmat" "$T/root"; cp -R "$1/mat/priv/scl/seqmat" "$T/scl"
  cp -R "$1/mat/priv/quarantine/seqmat/2001/b0000/layers/." "$T/root/windows/e111/b0000/layers/"   # its 5 of 12 groups back
  cp -R "$T/scl/windows/e111/b0001" "$T/scl/windows/e111/b0000"                                       # 12 SCL groups (the old rule)
  o=$("$2" -I "$3" scl-check --root "$T/root" --scl "$T/scl" 2>&1) && exit 1
  ! grep -q Traceback <<< "$o" && grep -q "\"sidecar_without_input\": 1" <<< "$o"
' _ "$SEQ" "$RSLEARN_PY" "$J"
# (a2) the usual failure: a window a materialize leaves with no group at all (rslearn marks the twelve together), every
# pass: counted in pass 3 (1 of 201) and in the SCL step's sum, no sidecar, nothing to quarantine, "materialize_failed"
seq_init "$SEQ/mat0" seqmat0 201
step prepare_root_failed_window_within_allowance bash -c 'o=$(seq_case "$1/mat0" seqmat0 0.005 2007 "{\"input\": {\"b0000\": 0}}" 2>&1) && grep -q "^prepare_root returned$" <<< "$o"' _ "$SEQ"
step prepare_root_failed_window_records seq_rec "$SEQ/mat0/rec" 'r("e111_seqmat0_materialize3.json")["failed_total"] == 1 and r("e111_seqmat0_scl_materialize.json")["failed_total"] == 1 and r("e111_seqmat0_precheck.json")["ok"] and r("e111_seqmat0_scl_check.json")["windows"] == 200 and r("e111_seqmat0_scl_check.json")["ok"] and r("e111_seqmat0_inventory.json")["total"]["materialize_failed"] == 1 and r("e111_seqmat0_inventory.json")["total"]["quarantined"] == 0'
# (b) the SCL materialize fails at one window with full input: its failure is added to pass 3's, the precheck
# quarantines its input and sidecar, the SCL check and the inventory see 200 windows with input
seq_init "$SEQ/scl" seqscl 201
step prepare_root_scl_failure_within_allowance bash -c 'o=$(seq_case "$1/scl" seqscl 0.005 2002 "{\"scl_fail\": [\"b0001\"]}" 2>&1) && grep -q "^prepare_root returned$" <<< "$o"' _ "$SEQ"
step prepare_root_scl_failure_records seq_rec "$SEQ/scl/rec" 'r("e111_seqscl_scl_materialize.json")["failed_total"] == 1 and r("e111_seqscl_precheck.json")["filesystem"]["scl_incomplete"] == 1 and r("e111_seqscl_precheck.json")["ok"] and r("e111_seqscl_scl_check.json")["ok"] and r("e111_seqscl_scl_check.json")["windows"] == 200 and r("e111_seqscl_inventory.json")["total"]["quarantined"] == 1 and os.path.isdir(os.path.join(os.path.dirname(R), "priv", "quarantine", "seqscl", "2002", "b0001", "scl_sidecar"))'
# (c) both at once: 2 of 201 windows, above the allowance: refused at the SCL step, with prepare_root's message
seq_init "$SEQ/both" seqboth 201
step prepare_root_summed_above_allowance_refused bash -c 'o=$(seq_case "$1/both" seqboth 0.005 2003 "{\"input\": {\"b0000\": 0}, \"scl_fail\": [\"b0001\"]}" 2>&1) && exit 1; grep -q "the SCL materialize failed at windows that, with pass 3.s failures, exceed the allowance" <<< "$o" && ! grep -q Traceback <<< "$o"' _ "$SEQ"
step prepare_root_failed_window_has_no_sidecar bash -c '[ ! -e "$1/both/priv/scl/seqboth/windows/e111/b0000" ] && [ -e "$1/both/priv/scl/seqboth/windows/e111/b0002/items.json" ]' _ "$SEQ"
# (d) the allowance at its proposed 0: a failed window stops prepare at pass 3, a partial one at the precheck; both are
# refusals with prepare_root's or precheck's message, never a traceback, and nothing is quarantined
seq_init "$SEQ/zero" seqzero 201
step prepare_root_allowance_0_refused bash -c 'o=$(seq_case "$1/zero" seqzero 0 2004 "{\"input\": {\"b0000\": 0}}" 2>&1) && exit 1; grep -q "seqzero: materialize failed at more than 0 of the windows in pass 3" <<< "$o" && ! grep -q Traceback <<< "$o"' _ "$SEQ"
seq_init "$SEQ/zero5" seqzero5 201
step prepare_root_allowance_0_partial_refused bash -c 'o=$(seq_case "$1/zero5" seqzero5 0 2008 "{\"input\": {\"b0000\": 5}}" 2>&1) && exit 1; grep -q "precheck failed: windows with 1 to 11 item groups" <<< "$o" && ! grep -q Traceback <<< "$o" && [ ! -e "$1/zero5/priv/quarantine" ]' _ "$SEQ"
# (e) reruns (--from prepare): the window quarantined by job 2001 fails again (job 2005: a second quarantine, its own
# directory, nothing nested), then is fetched whole (job 2006: no longer counted as quarantined, its sidecar made again)
step prepare_root_rerun_quarantines_apart bash -c 'o=$(seq_case "$1/mat" seqmat 0.005 2005 "{\"input\": {\"b0000\": 5}}" 2>&1) && grep -q "^prepare_root returned$" <<< "$o"' _ "$SEQ"
step prepare_root_rerun_no_nesting bash -c 'Q=$1/mat/priv/quarantine/seqmat; [ "$(ls "$Q")" = "$(printf "2001\n2005")" ] && [ -e "$Q/2001/b0000/layers/sentinel2/completed" ] && [ -e "$Q/2005/b0000/layers/sentinel2/completed" ] && [ "$(find "$Q" -name completed | wc -l | tr -d " ")" = 10 ] && [ "$(ls "$Q/2001/b0000/layers/sentinel2")" = "$(printf "B01_B02\ncompleted")" ]' _ "$SEQ"
step prepare_root_rerun_fetched_whole bash -c 'o=$(seq_case "$1/mat" seqmat 0.005 2006 "{}" 2>&1) && grep -q "^prepare_root returned$" <<< "$o"' _ "$SEQ"
step prepare_root_rerun_inventory seq_rec "$SEQ/mat/rec" 'r("e111_seqmat_inventory.json")["total"]["quarantined"] == 0 and r("e111_seqmat_inventory.json")["total"]["quarantined_earlier_now_with_input"] == 1 and r("e111_seqmat_inventory.json")["total"]["with_12_groups"] == 201 and r("e111_seqmat_inventory.json")["quarantine_runs"] == 2 and r("e111_seqmat_scl_check.json")["windows"] == 201 and r("e111_seqmat_scl_check.json")["ok"]'
echo "== the gate job's records: removed when a gate starts, the device record only after its four checks, one job's =="
gate_case() {   # gate_case <case dir> <job> <commit> <predict call that fails, 0 for none>: e111.sh's gate mode itself, the GPU,
  # rslearn's predict and the layer guard stood in for; check-output is the real one, on the smoke's roots
  bash -c '
    set -euo pipefail
    [ -s "$1" ] && [ -d "$6" ] || { echo "gate_case: no e111.sh or no smoke roots (E111_SH, P unset?)"; exit 2; }
    eval "$(sed -n "/^rec_ok() /,/^}\$/p; /^masked() /,/^}\$/p; /^remove_aprime() /,/^}\$/p" "$1")"
    body=$(awk "/^  gate\\)\$/{f=1; next} /^  gatecheck\\)\$/{f=0} f" "$1" | sed "s/ ;;\$//")
    [ -n "$body" ]
    case_dir=$2 JOB=$3 E111_SHA=$4 FAIL_AT=$5
    PRIV=$6 REC=$case_dir/rec GATE=$case_dir/e111_gate.json DEPLOY=$case_dir/none VENV=$case_dir/no_venv
    INV23=$case_dir/none RUN23=$case_dir/none CONFIG=$7/jobs/e102_config/model.yaml MODE=gate E111_APRIME=0 REDO=gate
    JPY=("$8" -I "$7/exp111_jobs.py")
    export SLURM_JOB_ID=$JOB
    mkdir -p "$REC" "$PRIV/tmp"
    for r in gate23 gate17; do
      for k in precheck scl_check; do echo "{\"ok\": true}" > "$REC/e111_${r}_$k.json"; done
    done
    n=0
    gpu_prep() { "${JPY[@]}" device --name "NVIDIA B200" --commit "$E111_SHA" --record "$1" > /dev/null; }
    fresh_layer() { :; }
    run_predict() { n=$((n + 1)); [ "$n" != "$FAIL_AT" ] || { echo "predict $n failed (stand-in)"; exit 1; }; }
    eval "$body"
  ' _ "$E111_SH" "$@" "$P" "$EXP" "$RSLEARN_PY"
}
export -f gate_case
GC=$OUT/gatecase
rm -rf "$GC"; mkdir -p "$GC"
A40=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
B40=bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
step gate_job_writes_one_jobs_records bash -c 'gate_case "$1" 3001 "$2" 0 > /dev/null && "$3" -I "$4" gate-records --rec "$1/rec"' _ "$GC" "$A40" "$RSLEARN_PY" "$J"
step gate_device_record_names_job_and_commit seq_rec "$GC/rec" 'r("e111_device_gate.json")["job"] == "3001" and r("e111_device_gate.json")["commit"] == "a" * 40 and all(r(f"e111_{g}_{m}_output.json")["job"] == "3001" for g in ("gate23", "gate17") for m in ("v1", "v12"))'
# the review's sequence: a later gate (commit B) dies at its second predict; the earlier gate's records must not let a
# --from gatecheck pass B's device record beside A's output checks
step gate_rerun_dying_leaves_no_device_record bash -c 'gate_case "$1" 3002 "$2" 2 > /dev/null 2>&1 && exit 1; [ ! -e "$1/rec/e111_device_gate.json" ] && [ -s "$1/rec/e111_gate23_v1_output.json" ] && [ ! -e "$1/rec/e111_gate17_v12_output.json" ]' _ "$GC" "$B40"
step gatecheck_refuses_a_dead_gate refused "no record of the gate's GPU" "$RSLEARN_PY" -I "$J" gate-records --rec "$GC/rec"
step gatecheck_refuses_mixed_jobs bash -c '
  set -euo pipefail
  M=$1/mixed; rm -rf "$M"; mkdir -p "$M"
  echo "{\"job\": \"3002\", \"commit\": \"$2\", \"name\": \"NVIDIA B200\"}" > "$M/e111_device_gate.json"
  for g in gate23 gate17; do for m in v1 v12; do echo "{\"ok\": true, \"job\": \"3001\"}" > "$M/e111_${g}_${m}_output.json"; done; done
  o=$("$3" -I "$4" gate-records --rec "$M" 2>&1) && exit 1
  grep -q "is another job.s than the device record.s" <<< "$o"
' _ "$OUT" "$B40" "$RSLEARN_PY" "$J"
step gatecheck_refuses_a_failed_output_check bash -c '
  set -euo pipefail
  F=$1/failed; rm -rf "$F"; mkdir -p "$F"
  echo "{\"job\": \"3004\", \"commit\": \"$2\", \"name\": \"NVIDIA B200\"}" > "$F/e111_device_gate.json"
  for g in gate23 gate17; do for m in v1 v12; do echo "{\"ok\": true, \"job\": \"3004\"}" > "$F/e111_${g}_${m}_output.json"; done; done
  echo "{\"ok\": false, \"job\": \"3004\"}" > "$F/e111_gate17_v12_output.json"
  o=$("$3" -I "$4" gate-records --rec "$F" 2>&1) && exit 1
  grep -q "the output check for gate17 v12 did not pass" <<< "$o"
' _ "$OUT" "$B40" "$RSLEARN_PY" "$J"
step gate_after_rerun_passes_again bash -c 'gate_case "$1" 3003 "$2" 0 > /dev/null && "$3" -I "$4" gate-records --rec "$1/rec"' _ "$GC" "$B40" "$RSLEARN_PY" "$J"
echo "== a partly purged deployment: the time-range check counts what it cannot read, item identity is a note =="
step deploy_partial_purge bash -c '
  set -euo pipefail
  D=$1/deploy_partial; rm -rf "$D"; cp -R "$1/deploy_run" "$D"
  w=$(dirname "$(find "$D" -name metadata.json | sort | head -n 1)"); cp -R "$w" "${w}_twin"   # two windows, one purged
  rm "$w/metadata.json"
  "$2" -I "$3" deploy-time-range --inv "$1/deploy_inventory.json" --run "$D" --year 2023 --record "$1/rec/e111_deploy_time_range_partial.json" > /dev/null
  "$2" -I -c "import json,sys; r=json.load(open(sys.argv[1])); sys.exit(0 if r[\"ok\"] and r[\"checked\"] and r[\"unreadable\"] == 1 else 1)" "$1/rec/e111_deploy_time_range_partial.json"
  o=$("$2" -I "$3" item-identity --root "$4/roots/gate23" --units "$4/units_gate23.json" --inv "$1/deploy_inventory.json" --run "$D" \
        --flags-out "$1/identity_partial_flags.json" --record "$1/rec/e111_identity_partial.json" 2>&1)
  grep -q "note: item identity skipped" <<< "$o" && ! grep -q Traceback <<< "$o" && [ ! -e "$1/identity_partial_flags.json" ]
' _ "$OUT" "$RSLEARN_PY" "$J" "$P"
step watchdog_once "$RSLEARN_PY" -I "$J" watchdog --dir "$P" --roots "$P/roots/gate23" --max-gb 25 --once --reason "$OUT/wd_reason.txt"
step watchdog_trips "$RSLEARN_PY" -I -c "import subprocess,sys; sys.exit(0 if subprocess.run([sys.executable,'-I',sys.argv[1],'watchdog','--dir',sys.argv[2],'--max-gb','0.000001','--interval','0.1','--reason',sys.argv[3]]).returncode==3 else 1)" "$J" "$P" "$OUT/wd_reason2.txt"
step remove_layer_refuses "$RSLEARN_PY" -I -c "import subprocess,sys; sys.exit(0 if subprocess.run([sys.executable,'-I',sys.argv[1],'remove-layer','--root',sys.argv[2],'--layer','output_v1']).returncode!=0 else 1)" "$J" "$P/roots/gate23"
step load_check_logic "$RSLEARN_PY" -I - "$EXP" <<'EOF'
import sys
sys.path.insert(0, sys.argv[1])
import torch, exp111_jobs as j
m = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 2))
sd = {k: v.clone() for k, v in m.state_dict().items()}
assert j.compare_state_dicts(m, sd) == ([], [], [])
bad = dict(sd); bad.pop("1.bias"); bad["extra.weight"] = torch.zeros(1); bad["0.weight"] = torch.zeros(5, 3)
mi, un, sh = j.compare_state_dicts(m, bad)
assert (mi, un, sh) == (["1.bias"], ["extra.weight"], ["0.weight"]), (mi, un, sh)
print("compare_state_dicts finds missing, unexpected and shape-mismatched keys")
# the verdict on a module carrying olmoearth's own legacy hook (CompositeEncodings._drop_pos_embed_hook), as v1's
# checkpoint meets it under rslearn 0.1.17 (env jobs 1355980, 1361497): the legacy table is accepted within LEGACY_MAX
# (0.01) of the on-the-fly encoding, and whether it equals it is recorded
import argparse, os, tempfile
from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.nn.flexi_vit import CompositeEncodings
from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.nn.encodings import get_1d_sincos_pos_encoding
class Enc(CompositeEncodings):          # the hook's own class, built without the real encoder's tables
    def __init__(self, d=8):
        torch.nn.Module.__init__(self)
        self.lin, self.embedding_dim_per_embedding_type = torch.nn.Linear(3, 4), d
        self._register_load_state_dict_pre_hook(CompositeEncodings._drop_pos_embed_hook)
class Plain(torch.nn.Module):           # a module of another class that drops the same key: not accepted
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(3, 4)
        self._register_load_state_dict_pre_hook(CompositeEncodings._drop_pos_embed_hook)
class M(torch.nn.Module):
    def __init__(self, enc=None):
        super().__init__()
        self.composite_encodings, self.head = enc if enc is not None else Enc(), torch.nn.Linear(4, 2)
K = "composite_encodings.pos_embed"
base = {k: v.clone() for k, v in M().state_dict().items()}
table = get_1d_sincos_pos_encoding(torch.arange(20), 8)
assert (j.LEGACY_MAX, j.LEGACY_EQUAL) == (0.01, 1e-5)
r = j.load_check_result(M(), base)
assert r["ok"] and r["accepted_unexpected"] == [], r
r = j.load_check_result(M(), {**base, K: table.clone()})
assert r["ok"] and r["strict_load"] and r["accepted_unexpected"] == [K] and r["legacy_table_max_abs_diff"][K] <= 1e-6, r
assert r["legacy_table_is_the_formula"][K] and len(r["legacy_table_row_max_abs_diff"][K]) == 20, r
# a drifted table as in v1's checkpoint (up to 0.0037): accepted, recorded as not the formula's
r = j.load_check_result(M(), {**base, K: table + 0.004})
assert r["ok"] and not r["legacy_table_is_the_formula"][K] and abs(r["legacy_table_max_abs_diff"][K] - 0.004) < 1e-6, r
r = j.load_check_result(M(), {**base, K: table[None].clone()})
assert r["ok"], r
for what, bad in (("beyond the guard", table + 0.05), ("just beyond the guard", table + 0.0101),
                  ("nan", table.masked_fill(table > 0.9, float("nan"))), ("another width", get_1d_sincos_pos_encoding(torch.arange(20), 4)),
                  ("empty", table[:0]), ("complex", torch.complex(table, torch.ones_like(table))),
                  ("integer", table.round().to(torch.int64))):
    r = j.load_check_result(M(), {**base, K: bad})
    assert not r["ok"], (what, r)
r = j.load_check_result(M(Plain()), {**{k: v for k, v in base.items()}, K: table.clone()})
assert r["strict_load"] and not r["ok"] and r["legacy_table_max_abs_diff"][K] is None, r
r = j.load_check_result(M(), {**base, "head.extra": torch.zeros(1)})
assert not r["strict_load"] and not r["ok"] and r["load_error"], r
r = j.load_check_result(M(), {**base, "head.junk": 5})
assert not r["ok"] and r["non_tensor"] == 1, r
r = j.load_check_result(M(), {k: v for k, v in base.items() if k != "head.bias"})
assert not r["ok"], r
print("load_check_result: only CompositeEncodings' legacy table at the encoder's width and within 0.01 of the formula is accepted (equality recorded); any other extra, non-tensor or missing key fails")
d = tempfile.mkdtemp()
torch.save({"state_dict": base, "hyper_parameters": {"lr": 0.1}}, os.path.join(d, "plain.ckpt"))
sd, how = j.load_checkpoint_safely(os.path.join(d, "plain.ckpt"))
assert how == "weights_only" and sorted(sd) == sorted(base), how
torch.save({"state_dict": base, "hyper_parameters": argparse.Namespace(a=1)}, os.path.join(d, "stub.ckpt"))
sd, how = j.load_checkpoint_safely(os.path.join(d, "stub.ckpt"))
assert "inert stub" in how and sorted(sd) == sorted(base), how
import operator
for i, x in enumerate((os.stat(d), operator.methodcaller("lower"))):
    torch.save({"state_dict": base, "x": x}, os.path.join(d, f"bad{i}.ckpt"))
    try:
        j.load_checkpoint_safely(os.path.join(d, f"bad{i}.ckpt"))
        raise SystemExit(f"a checkpoint naming {type(x)} was loaded")
    except RuntimeError as ex:
        assert "refused" in str(ex), ex
print("load_checkpoint_safely: weights_only, inert stubs for other globals, refuses os.* and operator.*")
EOF

# prepare 1364257 died on "AF_UNIX path too long": rslearn's forkserver socket sits 32 characters under TMPDIR and a
# socket path holds at most 107; every TMPDIR goes through set_tmp, and the cluster's path leaves room
step tmpdir_fits_a_socket bash -c '
  set -euo pipefail
  eval "$(sed -n "/^set_tmp() /,/^}\$/p" "$1")"
  [ "$(grep -c "TMPDIR=" "$1")" = 1 ] && grep -q "^  export TMPDIR=\$TSHORT/\$JOB.\$TMPN$" "$1"
  grep -q "^TSHORT=\$PRIV/t$" "$1" && grep -q "^PRIV=\$SCRATCH/e111_private$" "$1"
  PRIV=/scratch/qi_zim_neu/olmoearth_inferenceX/e111_private; TSHORT=$PRIV/t; JOB=99999999; TMPN=98
  mkdir() { :; }
  set_tmp
  [ "$TMPDIR" = /scratch/qi_zim_neu/olmoearth_inferenceX/e111_private/t/99999999.99 ] && [ $(( ${#TMPDIR} + 32 )) -le 100 ]
  o=$(TSHORT=/scratch/$(printf "%070d" 0); set_tmp 2>&1) && exit 1
  grep -q "too long for rslearn.s forkserver socket" <<< "$o"
' _ "$E111_SH"
# v1's own temporal table (exp111_v1table.py, the route after gate A failed with the formula's, job 1365118): the one
# call it replaces, the answers it gives, the refusal when it was never used, and e111.sh routing only v1 through it
step v1table_route "$RSLEARN_PY" -I - "$EXP" "$OUT" "$E111_SH" <<'EOF'
import ast, json, os, sys
EXP, OUT, SH = sys.argv[1:4]
sys.path.insert(0, EXP)
import torch, exp111_v1table as vt
from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.nn import flexi_vit
from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.nn.encodings import get_1d_sincos_pos_encoding as formula
tree = ast.parse(open(flexi_vit.__file__).read())
calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "get_1d_sincos_pos_encoding"]
assert len(calls) == 1, f"flexi_vit calls the temporal encoding {len(calls)} times"
owner = [c.name for c in ast.walk(tree) if isinstance(c, ast.ClassDef) and any(n is calls[0] for n in ast.walk(c))]
assert owner == ["CompositeEncodings"], owner
table = formula(torch.arange(12), 192) + 0.003
counts = vt.install_own_table(table)
assert torch.equal(flexi_vit.get_1d_sincos_pos_encoding(torch.arange(5), 192), table[:5])
assert torch.allclose(flexi_vit.get_1d_sincos_pos_encoding(torch.arange(5), 8), formula(torch.arange(5), 8))
try:
    flexi_vit.get_1d_sincos_pos_encoding(torch.arange(13), 192)
    raise SystemExit("13 timesteps were answered from a 12-row table")
except ValueError:
    pass
assert counts == {"own": 1, "formula": 1}, counts
# the script's main(), with rslearn's CLI stood in for (this Mac's venv lacks rslearn's extras; the cluster's has
# them): one stand-in runs the temporal call once and exits 0, one only exits 0 (the table never used: refused), one
# exits 2 (its code passes through)
import types
d = os.path.join(OUT, "v1table")
os.makedirs(d, exist_ok=True)
ck = os.path.join(d, "v1.ckpt")
torch.save({"state_dict": {vt.TABLE_KEY: table.clone(), "model.head.weight": torch.zeros(2, 2)}}, ck)
def run(entry, record, ckpt=ck):
    sys.modules["rslearn.main"] = types.SimpleNamespace(main=entry)
    sys.argv = ["exp111_v1table.py", "--ckpt", ckpt, "--record", record, "--", "model", "predict"]
    try:
        vt.main()
    except SystemExit as ex:
        return ex.code
def uses_table():
    assert sys.argv == ["rslearn", "model", "predict"], sys.argv
    assert torch.equal(flexi_vit.get_1d_sincos_pos_encoding(torch.arange(12), 192), table)
    raise SystemExit(0)
def exits(code):
    def f():
        raise SystemExit(code)
    return f
assert run(uses_table, os.path.join(d, "e111_v1table_used.json")) == 0
R = json.load(open(os.path.join(d, "e111_v1table_used.json")))
assert R["calls_own_table"] == 1 and R["rslearn_exit"] == 0, R
assert abs(R["table_max_abs_diff_from_formula"] - 0.003) < 1e-6 and abs(R["table_row0_max_abs_diff"] - 0.003) < 1e-6, R
assert run(exits(0), os.path.join(d, "e111_v1table_unused.json")) == 3
assert json.load(open(os.path.join(d, "e111_v1table_unused.json")))["calls_own_table"] == 0
assert run(exits(2), os.path.join(d, "e111_v1table_failed.json")) == 2
torch.save({"state_dict": {"model.head.weight": torch.zeros(2, 2)}}, os.path.join(d, "none.ckpt"))
code = run(uses_table, os.path.join(d, "never.json"), os.path.join(d, "none.ckpt"))
assert isinstance(code, str) and code.startswith("refused") and not os.path.exists(os.path.join(d, "never.json")), code
# a stand-in that raises: the record names the exception, never exit 0
def crashes():
    raise RuntimeError("stand-in crash")
try:
    run(crashes, os.path.join(d, "e111_v1table_crash.json"))
    raise SystemExit("a crash passed")
except RuntimeError:
    pass
assert json.load(open(os.path.join(d, "e111_v1table_crash.json")))["rslearn_exit"] == "exception: RuntimeError"
# the variants' other patches: the encodings in float32, and the wrapper without autocast (stand-ins for the originals)
CE = flexi_vit.CompositeEncodings
saved = CE._apply_encodings_per_modality
CE._apply_encodings_per_modality = lambda self, name, tokens, *a, **k: tokens.dtype
c2 = {"fp32buf": 0}
vt.install_fp32_encodings(c2)
assert CE._apply_encodings_per_modality(None, "sentinel2_l2a", torch.zeros(2, dtype=torch.bfloat16)) == torch.float32
assert c2 == {"fp32buf": 1}, c2
CE._apply_encodings_per_modality = saved
from rslearn.models.olmoearth_pretrain import model as oe_model
# the no-autocast patch leaves __init__ alone (jsonargparse refused a replaced one, job 1368915) and turns autocast off
# at the first forward: a stand-in forward sees autocast_dtype None
import inspect
init_before = oe_model.OlmoEarth.__init__
saved_fwd = oe_model.OlmoEarth.forward
oe_model.OlmoEarth.forward = lambda self, *a, **k: self.autocast_dtype
c3 = {"no_autocast": 0}
vt.install_no_autocast(c3)
assert oe_model.OlmoEarth.__init__ is init_before
w = torch.nn.Module.__new__(oe_model.OlmoEarth)
torch.nn.Module.__init__(w)
w.autocast_dtype = torch.bfloat16
assert oe_model.OlmoEarth.forward(w) is None and oe_model.OlmoEarth.forward(w) is None and c3 == {"no_autocast": 1}, c3
oe_model.OlmoEarth.forward = saved_fwd
# a variant whose second patch never applied is refused like an unused table
sys.argv = ["exp111_v1table.py", "--ckpt", ck, "--record", os.path.join(d, "e111_v1table_fp32unused.json"),
            "--variant", "table_fp32buf", "--", "model", "predict"]
sys.modules["rslearn.main"] = types.SimpleNamespace(main=uses_table)
try:
    vt.main()
except SystemExit as ex:
    assert ex.code == 3, ex.code
R = json.load(open(os.path.join(d, "e111_v1table_fp32unused.json")))
assert R["variant"] == "table_fp32buf" and R["calls_own_table"] == 1 and R["calls_fp32_encodings"] == 0, R
CE._apply_encodings_per_modality = saved
sh = open(SH).read()
assert "\nV1_TABLE=formula\n" in sh and 'if [ "$m" = v1 ] && [ "$V1_TABLE" != formula ]; then' in sh
assert 'if ( V1_TABLE=$v run_predict "$PRIV/roots/gate23" v1 "output_d_$v" "$CONFIG" "diag-$v" ); then OKV+=("$v")' in sh
assert '--v1diag-variants "${OKV[*]}"' in sh
print("exp111_v1table: replaces flexi_vit's one temporal call (in CompositeEncodings), answers from the stored table at "
      "its width only; float32 encodings and no-autocast variants; refuses an unused route and a checkpoint without the "
      "table; records a crash; e111.sh routes v1 only, the committed route formula, v1diag all four")
EOF
echo "== the job scripts: syntax, lint, usage, inputs record, redo, the A' budget, the gate's commits, the submit chain (dry run), rule consistency =="
step bash_syntax bash -c 'bash -n "$1" && bash -n "$2"' _ "$E111_SH" "$SUBMIT_SH"
if command -v shellcheck > /dev/null; then
  step shellcheck shellcheck -S warning "$E111_SH" "$SUBMIT_SH"
fi
usage_case() {   # usage_case <ok|unknown|files|gib> [note]: e111.sh's usage() with find, du and timeout stood in for
  bash -c '
    set -euo pipefail
    eval "$(sed -n "/^usage() /,/^}\$/p" "$1")"
    HOMEDIR=$2/usage/$3/home; SCRATCH=$2/usage/$3/scratch; PRIV=$2/usage/$3/priv; REC=$2/rec; JOB=smoke
    mkdir -p "$HOMEDIR" "$SCRATCH" "$PRIV/tmp" "$REC"
    lines() { awk -v n="$1" -v k="$2" "BEGIN { for (i = 0; i < n; i++) print k }"; }
    du() { printf "100\t%s\n" "$2"; }
    case "$3" in
      ok) timeout() { shift; "$@"; }; find() { lines 1000 4; } ;;
      unknown) timeout() { return 124; } ;;
      files) timeout() { shift; "$@"; }; find() { lines 460000 4; } ;;
      gib) timeout() { shift; "$@"; }; find() { lines 1 95000000; } ;;
    esac
    declare -F usage > /dev/null
    tag=$([ "$3" = ok ] && echo env || echo "case_$3_${4:-refuse}")
    usage "$tag" ${4:-}
    echo "returned"
    if [ "$3" = ok ]; then python3 -c "import json,sys; r=json.load(open(sys.argv[1])); sys.exit(0 if r[\"home_files\"] == 1000 and r[\"home_kib\"] == 4000 else 1)" "$REC/e111_usage_env.json"; fi
  ' _ "$E111_SH" "$OUT" "$@"
}
export E111_SH SUBMIT_SH OUT
export -f usage_case
step usage_known usage_case ok
step usage_unknown_refused refused "refused: /home's usage is unknown" usage_case unknown
step usage_files_refused refused "refused: /home holds 460000 files" usage_case files
step usage_gib_refused refused "refused: /home is above 90 GiB" usage_case gib
# the end of prepare: recorded and noted, never failing a job that has done its work
step usage_note_unknown_goes_on bash -c 'o=$("$@") && grep -q "note: /home.s usage is unknown" <<< "$o" && grep -q "^returned$" <<< "$o"' _ usage_case unknown note
step usage_note_gib_goes_on bash -c 'o=$("$@") && grep -q "note: /home is above 90 GiB" <<< "$o" && grep -q "^returned$" <<< "$o"' _ usage_case gib note
step usage_note_files_goes_on bash -c 'o=$("$@") && grep -q "note: /home holds 460000 files" <<< "$o" && grep -q "^returned$" <<< "$o"' _ usage_case files note
# env's inputs record (counts and flags), written by e111.sh's own function, through the pack check below
step inputs_record bash -c '
  set -euo pipefail
  eval "$(sed -n "/^record_inputs() /,/^}\$/p" "$1")"
  declare -F record_inputs > /dev/null
  T=$2/inputs_case; rm -rf "$T"; mkdir -p "$T/labels/spatial_split" "$T/breadth"
  for k in $(seq 1 1459); do mkdir "$T/labels/spatial_split/w$k"; done
  echo geometry > "$T/geom.json"
  BREADTH=$T/breadth AI2_GEOMETRY=$T/geom.json AI2_GEOMETRY_SHA256=$(shasum -a 256 "$T/geom.json" | cut -d" " -f1)
  LABELS=$T/labels INV23=$T/none INV17=$T/none RUN23=$T/none RUN17=$T/none REC=$2/rec
  record_inputs 1 > /dev/null
  python3 -c "import json,sys; r=json.load(open(sys.argv[1])); sys.exit(0 if r[\"ok\"] is True and r[\"label_windows\"] == 1459 else 1)" "$REC/e111_inputs.json"
' _ "$E111_SH" "$OUT"
step redo_scoped bash -c 'set -euo pipefail; eval "$(sed -n "/^redo_for() /p" "$1")"; declare -F redo_for > /dev/null; REDO=gate,gatecheck; redo_for gate && redo_for gatecheck && ! redo_for select && ! redo_for gat && REDO="" && ! redo_for gate' _ "$E111_SH"
step aprime_budget bash -c 'set -euo pipefail; eval "$(sed -n "/^budget_s() /,/^}\$/p" "$1")"; declare -F budget_s > /dev/null; T_START=$(date +%s); GATE_LIMIT_S=9000; unset SLURM_JOB_END_TIME; b1=$(budget_s 900); SLURM_JOB_END_TIME=$(( $(date +%s) + 3000 )); b2=$(budget_s 900); [ "$b1" -ge 8090 ] && [ "$b1" -le 8100 ] && [ "$b2" -ge 2090 ] && [ "$b2" -le 2100 ]' _ "$E111_SH"
# the gate's two commits: a throwaway git repository with commit A (the gate job's), B (a guarded file changed) and C
# (an unguarded file changed); e111.sh's own same_guarded, need_frozen and guard_phase2 on it
step gate_commits bash -c '
  set -euo pipefail
  eval "$(sed -n "/^GUARDED=(/,/)\$/p; /^same_guarded() /,/^}\$/p; /^need_frozen() /,/^}\$/p; /^guard_phase2() /,/^}\$/p" "$1")"
  declare -F same_guarded guard_phase2 need_frozen > /dev/null
  G=$2/gitcase; rm -rf "$G"; mkdir -p "$G/exp"; cd "$G"
  git init -q; git config user.email smoke@example.invalid; git config user.name smoke
  echo a > exp/exp111_jobs.py; echo a > README.md; git add -A; git commit -qm A; CA=$(git rev-parse HEAD)
  echo b > exp/exp111_jobs.py; git commit -qam B; CB=$(git rev-parse HEAD)
  echo c > README.md; git commit -qam C; CC=$(git rev-parse HEAD)
  same_guarded "$CA" "$CA"; same_guarded "$CB" "$CC"
  ! same_guarded "$CA" "$CC"; ! same_guarded "$CC" "$CA"; ! same_guarded "" "$CC"; ! same_guarded "abc" "$CC"
  FROZEN=1; PRIV=$G; GATE=$G/gate.json; E111_SHA=$CC; PY=(true)
  printf "{\"pass\": true, \"predict_commit\": \"%s\", \"commit\": \"%s\"}" "$CB" "$CC" > "$GATE"
  (guard_phase2) > /dev/null
  printf "{\"pass\": true, \"predict_commit\": \"%s\", \"commit\": \"%s\"}" "$CA" "$CC" > "$GATE"
  o=$( (guard_phase2) 2>&1 ) && exit 1; grep -q "refused: the gate predicted or passed at" <<< "$o"
  printf "{\"pass\": false, \"predict_commit\": \"%s\", \"commit\": \"%s\"}" "$CB" "$CC" > "$GATE"
  o=$( (guard_phase2) 2>&1 ) && exit 1; grep -q "refused: the known-answer gate did not pass" <<< "$o"
  FROZEN=0; o=$( (guard_phase2) 2>&1 ) && exit 1; grep -q "is not frozen" <<< "$o"
  echo "same_guarded and guard_phase2 hold on both commits"
' _ "$E111_SH" "$OUT"
submit_dry() { (cd "$OUT" && E111_DRY_RUN=1 bash "$SUBMIT_SH" "$@" 2>&1); }
export -f submit_dry
step submit_phase1 bash -c 'o=$("$@"); [ "$(grep -c "^DRY: " <<< "$o")" = 6 ] && grep -q "E111_MODE=gate .*-p b200-devel -q interactive" <<< "$o" && ! grep -q rtx <<< "$o" && grep -q "E111_REDO=gate,gatecheck" <<< "$o" && grep -q "E111_MODE=pack1 .*--mail-type=END,FAIL --dependency=afterany:1005" <<< "$o"' _ submit_dry phase1
step submit_from_gate bash -c 'o=$("$@"); [ "$(grep -c "^DRY: " <<< "$o")" = 3 ] && grep -q "^DRY: E111_MODE=gate .* -t 02:30:00 *$" <<< "$o" && grep -q "E111_MODE=gatecheck .*--dependency=afterok:1001 --kill-on-invalid-dep=yes" <<< "$o" && ! grep -q "E111_MODE=env\|E111_MODE=select\|E111_MODE=prepare" <<< "$o"' _ submit_dry phase1 --from gate
step submit_phase2 bash -c 'o=$("$@"); [ "$(grep -c "^DRY: " <<< "$o")" = 5 ] && grep -q "E111_MODE=predict .*-p b200-devel" <<< "$o" && grep -q "E111_MODE=grade" <<< "$o" && ! grep -q "E111_REDO" <<< "$o"' _ submit_dry phase2
step submit_bad_from_refused refused "from takes one of" submit_dry phase1 --from predict
step submit_sizes_refused refused "E111_SIZES is gone" env E111_SIZES=300,300,150 bash -c 'cd "$1" && E111_DRY_RUN=1 bash "$2" phase1' _ "$OUT" "$SUBMIT_SH"
step submit_drop_partial_refused refused "E111_DROP_PARTIAL is gone" env E111_DROP_PARTIAL=1 bash -c 'cd "$1" && E111_DRY_RUN=1 bash "$2" phase1' _ "$OUT" "$SUBMIT_SH"
step rules_consistent "$RECORD_PY" -I - "$E111_SH" "$SUBMIT_SH" "$B" "$GR" "$J" "$EXP" <<'PYEOF'
import ast, os, re, sys
sh, sub, b, gr, jb = (open(p).read() for p in sys.argv[1:6])
EXP = sys.argv[6]
lim = int(re.search(r"^GATE_LIMIT_S=(\d+)", sh, re.M).group(1))
t = re.search(r"sub gate .*-t (\d+):(\d+):(\d+)", sub).groups()
assert lim == int(t[0]) * 3600 + int(t[1]) * 60 + int(t[2]), "GATE_LIMIT_S is not the gate job's -t"
for mode in ("gate", "predict"):
    line = re.search(rf"sub {mode} .*", sub).group(0)
    assert "-p b200-devel " in line and "rtx" not in line, f"{mode} is not pinned to b200-devel"
guarded = re.search(r"^GUARDED=\((.*?)\)", sh, re.M | re.S).group(1).split()
need = ["exp/exp111_blind_transfer.py", "exp/exp111_jobs.py", "exp/exp111_grade.py", "exp/exp111_v1table.py",
        "exp/jobs/e111.sh",
        "exp/jobs/e111_submit.sh", "exp/jobs/e111_config", "exp/jobs/e102_config/model.yaml", "exp/jobs/e98_config",
        "pyproject.toml", "uv.lock"]
# the record modules: the import graph of the three exp111 modules. In an exp111 module every import counts, inside
# functions too; in a record module the imports that run when it is imported (module level, through if/try/with and
# class bodies, not function bodies: exp78's and exp89's function-level imports of exp21, exp51, exp54 and exp70 run
# only in their own experiments). Followed through every module that is a file in exp/.
EXP111 = ("exp111_blind_transfer", "exp111_jobs", "exp111_grade", "exp111_v1table")
def local_imports(path, everywhere):
    out = set()
    def visit(nodes):
        for node in nodes:
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else \
                [node.module] if isinstance(node, ast.ImportFrom) and node.module and node.level == 0 else []
            for n in names:
                top = n.split(".")[0]
                if os.path.isfile(os.path.join(EXP, top + ".py")):
                    out.add(top)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)) and not everywhere:
                continue
            visit(list(ast.iter_child_nodes(node)))
    visit(ast.parse(open(path).read()).body)
    return out
seen, todo = set(), list(EXP111)
while todo:
    m = todo.pop()
    if m in seen:
        continue
    seen.add(m)
    todo += sorted(local_imports(os.path.join(EXP, m + ".py"), m in EXP111) - seen)
need += [f"exp/{m}.py" for m in sorted(seen)]
missing = [n for n in need if n not in guarded]
assert not missing, f"GUARDED lacks {missing}"
assert "exp/exp98_burned.py" in need, "the import graph no longer reaches exp98_burned (check the walk)"
assert "PY=(uv run --frozen python" in sh and "GPY=(uv run --frozen python" in sh and "uv run python" not in sh
for mode in ("predict", "read", "units", "grade"):
    body = re.search(rf"^  {mode}\)\n(.*?);;", sh, re.M | re.S).group(1)
    assert "guard_phase2" in body, f"{mode} does not run guard_phase2"
assert '"torch==2.14.1" "lightning==2.6.6" "jsonargparse==4.52.0"' in sh
assert "--check-draw" in sh and "--device-file" in sh and "device --expect-gate" in sh
assert "flip" not in (b + gr).lower(), "R3's flip count is back in the code"
# the gate's two commits: the gate job records its commit, gatecheck and guard_phase2 check it
assert 'device --commit "$E111_SHA"' in sh, "gpu_prep does not record the gate job's commit"
gc = re.search(r"^  gatecheck\)\n(.*?);;", sh, re.M | re.S).group(1)
assert 'same_guarded "$P_COMMIT" "$E111_SHA"' in gc, "gatecheck does not check the gate job's commit"
g2 = re.search(r"^guard_phase2\(\) .*?^}", sh, re.M | re.S).group(0)
assert 'g["predict_commit"], g["commit"]' in g2 and "same_guarded" in g2, "guard_phase2 does not check both commits"
# one key check: pack runs exp111_jobs.py pack-check, and no key set of its own
pk = re.search(r"^  pack1\|pack2\)\n(.*?);;", sh, re.M | re.S).group(1)
assert "exp111_jobs.py\" pack-check --dir \"$T\"" in pk and '"lon"' not in pk, "pack does not use the shared key check"
sys.path.insert(0, EXP)
import exp98_awf_deployment as e98, exp111_jobs as j
import exp111_blind_transfer as b111, exp111_grade as g111
gfl, bsz = g111.FLOORS["with_input"], b111.SIZES
assert set(e98.FORBIDDEN_KEYS) <= j.RECORD_FORBIDDEN_KEYS and {"pid", "px", "py", "cell"} <= j.RECORD_FORBIDDEN_KEYS
assert "jobs.check_record_keys" in b and "jobs.check_record_keys" in gr, "a record writer skips the shared key check"
# /home: env refuses, prepare notes
env = re.search(r"^  env\)\n(.*?);;", sh, re.M | re.S).group(1)
prep = re.search(r"^  prepare\)\n(.*?);;", sh, re.M | re.S).group(1)
assert re.search(r"^    usage env$", env, re.M) and re.search(r"^    usage prepare note$", prep, re.M)
assert not re.search(r"^\s+usage (gate|gatecheck|predict|read|units|grade|select)\b", sh, re.M)
# the failure allowance: a constant here (part of the commit), at most the cap, and not an option of the submit script
mfs = float(re.search(r"^MAX_FAILED_SHARE=([0-9.]+)$", sh, re.M).group(1))
assert 0 <= mfs <= j.MAX_FAILED_SHARE_CAP == 0.005
assert "--max-drop-share \"$MAX_FAILED_SHARE\"" in sh and "maxf=$MAX_FAILED_SHARE" in sh
assert "E111_DROP_PARTIAL E111" not in sub and "--drop-partial" not in sh
# E111_REDO=select is refused after freezing, in the job and in the module
sel = re.search(r"^  select\)\n(.*?);;", sh, re.M | re.S).group(1)
assert '[ "$FROZEN" = 0 ] ||' in sel and "refuse_redo_if_frozen(redo" in b
assert "--deploy-inv \"$INV17\" --deploy-run \"$RUN17\"" in sel, "select does not give the windows exp99's deployment"
# prepare_root's order: the SCL sidecar and its materialize (errors counted against the allowance) before the precheck,
# which quarantines, and the SCL check only after it
pr = re.search(r"^prepare_root\(\) .*?^}", sh, re.M | re.S).group(0)
order = ["scl-sidecar --root", "dataset materialize --root \"$PRIV/scl/$name\"", "--layer sentinel2_scl",
         "precheck --root", "scl-check --root", "inventory --root"]
pos = [pr.find(t) for t in order]
assert -1 not in pos and pos == sorted(pos), f"prepare_root's order is not sidecar, SCL materialize, its summary, precheck, scl-check, inventory: {pos}"
scl_mat = pr[pos[1]:pr.find("\n", pr.find("--retry-backoff-seconds 60", pos[1]))]
assert "--ignore-errors" in scl_mat, "the SCL materialize runs without --ignore-errors"
assert '--plus "$REC/e111_${name}_materialize3.json"' in pr and '--scl "$PRIV/scl/$name"' in pr and '--run-tag "$JOB"' in pr
# the gate job: an earlier gate's records removed before it predicts, the device record moved in only after the four
# output checks; gatecheck checks the five are one job's
gt = re.search(r"^  gate\)\n(.*?);;", sh, re.M | re.S).group(1)
i_rm, i_prep = gt.find('rm -f "$REC/e111_device_gate.json"'), gt.find('gpu_prep "$DEV_PENDING"')
i_loop, i_mv = gt.find("check-output --root"), gt.find('mv "$DEV_PENDING" "$REC/e111_device_gate.json"')
assert -1 not in (i_rm, i_prep, i_loop, i_mv) and i_rm < i_prep < i_loop < i_mv, "the gate's device record is not written after its checks"
assert 'rm -f "$REC/e111_${r}_${m}_output.json"' in gt and "e111_${r}_scl_check.json" in gt
assert 'gate-records --rec "$REC"' in gc, "gatecheck does not check the gate job's records"
pr2 = re.search(r"^  predict\)\n(.*?);;", sh, re.M | re.S).group(1)
assert "e111_blind17_scl_check.json" in pr2 and 'rm -f "$REC/e111_blind17_${m}_output.json"' in pr2
# the draw's sizes: part of the commit, never an option
assert "--sizes" not in sh and 'E111_SIZES is gone' in sh and 'E111_SIZES is gone' in sub
assert re.search(r"for v in E111_BLIND_PREP", sub) and "E111_SIZES E111" not in sub
assert '"--sizes"' not in b, "the module still takes --sizes"
assert gfl <= round(0.8 * sum(bsz.values())), "the floor of plots with input is above 80% of SIZES' plots"
print(f"consistent: the gate's limit and partition, {len(guarded)} guarded paths over the import graph "
      f"({len(seen)} modules), uv run --frozen, the phase-2 guards, both gate commits, one key check, /home, the "
      "allowance, redo after freezing, the pins, prepare_root's order, the gate's records, the sizes")
PYEOF
if [ -n "$PAGE" ]; then
  step page_matches_code "$RECORD_PY" -I - "$PAGE" "$E111_SH" "$EXP" <<'PYEOF'
import re, sys
page, sh, EXP = open(sys.argv[1]).read(), open(sys.argv[2]).read(), sys.argv[3]
sys.path.insert(0, EXP)
import exp111_blind_transfer as b, exp111_grade as g, exp111_jobs as j
flat = re.sub(r"\s+", " ", page)
need = {
    "status is a known one": re.search(r"^\*\*Status: (DRAFT|decided, not frozen|frozen on )", page, re.M) is not None,
    "binding at the phase-1 commit": "binding at the phase-1 commit" in flat,
    "floor 720": f"at least {g.FLOORS['with_input']} of the 900" in flat,
    "floors 40/40": f"at least {g.FLOORS['errors_per_rule']} errors and {g.FLOORS['correct_per_rule']} correct" in flat,
    "floors 25/30": f"at least {g.FLOORS['confident_strict_errors']} confident STRICT errors and "
                    f"{g.FLOORS['strict_only_errors']} STRICT-only" in flat,
    "floor zone 30": f"at least {g.FLOORS['zone_plots']} plots in shrubland_savanna" in flat,
    "P1 bars": f"**{g.BARS['P1a_strict_below']:.2f}** and its 90% interval lies above **{g.BARS['P1a_lower_above']:.2f}**" in flat,
    "P2b bars": f"at most **{g.BARS['P2b_strict_only_max']:.2f}**" in flat and f"at least **{g.BARS['P2b_lenient_min']:.2f}**" in flat,
    "P2a bar": f"at least **{g.BARS['P2a_pair_share_min']:.2f}**" in flat,
    "P3 bar": f"P(note) at least **{g.BARS['P3_p_note_min']}**" in flat,
    "bootstrap": f"{g.BOOT['R']:,} resamples, seed {g.BOOT['seed']}" in flat and f"{int(g.BOOT['level'] * 100)}% percentile" in flat,
    "std counts": "O {O}, W {W}, X {X}".format(**g.STD_COUNTS) in flat,
    "gate 98%": f"{int(b.GATE_BARS['same_class_min_share'] * 100)}% or more" in flat,
    "gate floors": f"at least {b.GATE_BARS['floor_with_input_both']['gate23']} of the 259" in flat
                   and f"at least {b.GATE_BARS['floor_with_input_both']['gate17']} of the 271" in flat,
    "gate spearman": f"at least {b.GATE_BARS['spearman_suspicion_min']}" in flat,
    "sizes": "{O}/{W}/{X}".format(**b.SIZES) in flat,
    "allowance": f"MAX_FAILED_SHARE` = {re.search(r'^MAX_FAILED_SHARE=([0-9.]+)$', sh, re.M).group(1)}" in flat
                 and f"at most {j.MAX_FAILED_SHARE_CAP}" in flat,
    "confident cut": all(f"{c}" in flat for c in b.CONFIDENT_CUT.values()),
}
bad = [k for k, v in need.items() if not v]
assert not bad, f"the page and the code differ: {bad}"
print(f"the page states the code's numbers: {len(need)} checks")
PYEOF
fi
# the pack jobs' key check (the same function every record writer uses) over every record this smoke wrote, under the
# names the cluster gives them; it must also refuse the dataset-check record as it was before the rename
KINDS=(e111_select.json e111_windows.json e111_gate23_dataset_check.json e111_gate17_dataset_check.json
  e111_blind17_dataset_check.json e111_gate23_precheck.json e111_blind17_precheck.json e111_gate23_inventory.json
  e111_blind17_inventory.json e111_gate23_v1_output.json e111_blind17_v12_output.json e111_gate23_scl_check.json
  e111_identity_gate23.json e111_aprime_config.json e111_deploy_time_range23.json e111_gate23_prepare1.json
  e111_gate23_materialize1.json e111_gate23_prepare3.json e111_gate23_materialize3.json e111_smoke2_groups.json
  e111_versions_env.json e111_configs.json e111_parse_model.json e111_device_gate.json e111_read_gate23_v1.json
  e111_read_gate17_v12.json e111_read_blind17_v1.json e111_gate.json e111_usage_env.json e111_inputs.json
  exp111_units_record.json exp111_summary.json)
step records_cover_the_cluster_kinds bash -c 'm=0; for k in "${@:2}"; do [ -s "$1/$k" ] || { echo "missing $k"; m=1; }; done; exit $m' _ "$R" "${KINDS[@]}"
step pack_check_every_record "$RECORD_PY" -I "$J" pack-check --dir "$R" --expect-files "${#KINDS[@]}"
# prepare_root's records of one root as the cluster names them (the SCL step's own among them), and the gate job's;
# then the same key check over every record the prepare_root and gate cases wrote
ROOT_KINDS=(prepare1 prepare2 prepare3 materialize1 materialize2 materialize3 scl_materialize precheck scl_check inventory)
step prepare_root_writes_every_record_kind bash -c 'm=0; for k in "${@:2}"; do [ -s "$1/e111_seqmat_$k.json" ] || { echo "missing $k"; m=1; }; done; exit $m' _ "$SEQ/mat/rec" "${ROOT_KINDS[@]}"
SEQ_DIRS=()
for d in "$SEQ"/*/rec "$GC/rec"; do SEQ_DIRS+=(--dir "$d"); done
step pack_check_prepare_root_and_gate_records "$RECORD_PY" -I "$J" pack-check "${SEQ_DIRS[@]}" --expect-files 60
# the pack job's fallback Python may lack numpy: the same check with numpy unimportable
step pack_check_without_numpy python3 -I -c "import runpy, sys; sys.modules['numpy'] = None; sys.argv = [sys.argv[1], 'pack-check', '--dir', sys.argv[2]]; runpy.run_path(sys.argv[0], run_name='__main__')" "$J" "$R"
mkdir -p "$OUT/pack_bad"
echo '{"root": "gate23", "bad": {"bounds": 0}, "ok": true}' > "$OUT/pack_bad/e111_gate23_dataset_check.json"
step pack_check_refuses_old_key refused "refused to pack" "$RECORD_PY" -I "$J" pack-check --dir "$OUT/pack_bad"
step record_writer_refuses_old_key refused "refusing to write" "$RECORD_PY" -I -c "import sys; sys.path.insert(0, sys.argv[1]); import exp111_jobs as j; j.write_json({'bad': {'bounds': 0}}, sys.argv[2])" "$EXP" "$OUT/pack_bad/never.json"
echo "== $pass passed, $fail failed =="
[ "$fail" = 0 ]
