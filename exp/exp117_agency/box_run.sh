#!/bin/bash
#SBATCH -J oeix-agency-box
#SBATCH -A p2026_0089_neu
#SBATCH -p cpu
#SBATCH -c 32
#SBATCH --mem=16G
#SBATCH -t 02:00:00
#SBATCH -o /home/qi_zim_neu/olmoearth_inferenceX/slurm/%x-%j.out
set -u
P=/scratch/qi_zim_neu/olmoearth_inferenceX/agency/1525456
SRC=$P/src
D=/scratch/qi_zim_neu/olmoearth_inferenceX/agency/$SLURM_JOB_ID
mkdir -p "$D" || exit 1
for f in judge_agency.py betting_widths.py agency_bounds.py strata_size.csv; do [ -f "$SRC/$f" ] || { echo "missing $SRC/$f"; exit 1; }; done
[ -d "$P/rec" ] && [ -d "$P/pkg/.venv" ] || { echo "missing the earlier job's clones"; exit 1; }
cat > "$D/ratio_box.py" <<'OEIX_FILE_END'
"""User's and producer's accuracy by a box of two linear betting intervals, against the ratio scan (aicr only).

R = theta_y / (theta_y + theta_w), with y the points mapped as the class and right, w the points in the denominator
and wrong (w = z - y, both 0/1). R rises with theta_y and falls with theta_w. Each of theta_y and theta_w gets the
stratified betting interval at alpha / 2 (each end at alpha / 4), so with probability at least 1 - alpha both hold, and
then R lies in [L_y / (L_y + U_w), U_y / (U_y + L_w)]: a union bound, valid whatever the betting interval's own
validity is (shown by the judge's enumerations). Each linear interval takes seconds, where the scan took 60 to 2,300 s.

usage: python ratio_box.py REC AREAS widths|mc R   (env BETTING_PY, BOUNDS_PY, JUDGE_PY as for judge_agency.py)
JSON on stdout, progress on stderr."""
import importlib.util
import json
import math
import multiprocessing
import os
import sys
import time
import traceback

import numpy as np

_spec = importlib.util.spec_from_file_location("judge_agency", os.environ["JUDGE_PY"])
ja = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ja)
ALPHA = ja.ALPHA
PROCS = max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))


def say(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def lin(D, x, alpha):
    return ja.bet_strata(ja.lin_strata(D, x), alpha=alpha)


def box_ratio(D, y_lo, z_lo, y_hi=None, z_hi=None, alpha=ALPHA):
    """The box interval for R = sum W mean(y) / sum W mean(z), y <= z in {0, 1}; with y_hi, z_hi, unknown values
    bounded (the lower end from y_lo, z_lo, the upper from y_hi, z_hi)."""
    y_hi = y_lo if y_hi is None else y_hi
    z_hi = z_lo if z_hi is None else z_hi
    ly, _ = lin(D, y_lo, alpha / 2)
    _, uw = lin(D, z_lo - y_lo, alpha / 2)
    _, uy = lin(D, y_hi, alpha / 2)
    lw, _ = lin(D, z_hi - y_hi, alpha / 2)
    lo = ly / (ly + uw) if ly + uw > 0 else 0.0
    hi = uy / (uy + lw) if uy + lw > 0 else 1.0
    return float(max(0.0, lo)), float(min(1.0, hi))


def q(D, y_lo, z_lo, y_hi=None, z_hi=None):
    out = {"n_numerator": int(y_lo.sum()), "n_denominator": int(z_lo.sum())}
    R, v = ja.normal_ratio(D, y_lo, z_lo)
    if R is None:
        out["note"] = "no labelled point in the denominator"
        return out
    out["estimate"] = R
    out["normal"] = [R - ja.Z * math.sqrt(v), R + ja.Z * math.sqrt(v)]
    t0 = time.time()
    out["box"] = list(box_ratio(D, y_lo, z_lo, y_hi, z_hi))
    out["seconds_box"] = time.time() - t0
    out["widths_points"] = {m: 100 * (out[m][1] - out[m][0]) for m in ("normal", "box")}
    out["box_contains_estimate"] = bool(out["box"][0] - 1e-9 <= R <= out["box"][1] + 1e-9)
    return out


def widths(rec, areas):
    res = {}
    f = lambda m: m.astype(float)
    G = ja.load_gfc(rec, areas)
    ref = G["ref"]
    for mname in ("gfc_v2", "worldcover"):
        mv = G["gfc"] if mname == "gfc_v2" else G["wc"]
        miss = (mv < 0) if mname == "worldcover" else np.zeros(mv.size, bool)
        b = bool(miss.any())
        for cls, cname in ((1, "forest"), (0, "nonforest")):
            known = mv == cls
            y_lo, y_hi = f(known & (ref == cls)), f((known | miss) & (ref == cls))
            uz_lo, uz_hi = f(known | (miss & (ref != cls))), f(known | (miss & (ref == cls)))
            pz = f(ref == cls)
            say(f"gfc {mname} {cname}")
            res[f"gfc2020/user_{cname}_{mname}"] = q(G, y_lo, uz_lo, y_hi if b else None, uz_hi if b else None)
            res[f"gfc2020/producer_{cname}_{mname}"] = q(G, y_lo, pz, y_hi if b else None, pz if b else None)
    N = ja.load_nlcd(rec, 2016)
    m, r, p = N["map"], N["rstar"], N["ref"]
    for c in sorted((set(m.tolist()) | set(r.tolist())) - {0}):
        mc, rc, pc = m == c, r == c, p == c
        say(f"nlcd class {c}")
        res[f"nlcd2016_B_2016/user/{c}"] = q(N, f(mc & rc), f(mc))
        res[f"nlcd2016_B_2016/producer/{c}"] = q(N, f(mc & rc), f(rc))
        res[f"nlcd2016_B_2016/user_primary/{c}"] = q(N, f(mc & pc), f(mc))
    return res


# ------------------------------------------------------------------ coverage by simulation
_CASES, _JOBS = {}, []


def case_ratio_finite(D, c, what, f):
    H, n, W = D["H"], D["n"], D["W"]
    m, r = D["map"], D["rstar"]
    A = (m == c) & (r == c)
    B = (m == c) & (r != c) if what == "user" else (m != c) & (r == c)
    Np = f * n
    pops = []
    for h in range(H):
        mk = D["masks"][h]
        Ah, Bh = int(round(A[mk].mean() * Np[h])), int(round(B[mk].mean() * Np[h]))
        pops.append((Ah, Bh, int(Np[h]) - Ah - Bh))
    theta = (sum(W[h] * pops[h][0] / Np[h] for h in range(H))
             / sum(W[h] * (pops[h][0] + pops[h][1]) / Np[h] for h in range(H)))

    def draw(rng):
        cts = [rng.multivariate_hypergeometric(np.array(pops[h], dtype=np.int64), int(n[h])) for h in range(H)]
        sy = [ja._st(W[h], n[h], float(Np[h]), None, [0, 1], [int(n[h] - cts[h][0]), int(cts[h][0])]) for h in range(H)]
        sw = [ja._st(W[h], n[h], float(Np[h]), None, [0, 1], [int(n[h] - cts[h][1]), int(cts[h][1])]) for h in range(H)]
        ly, uy = ja.bet_strata(sy, alpha=ALPHA / 2)
        lw, uw = ja.bet_strata(sw, alpha=ALPHA / 2)
        lo = ly / (ly + uw) if ly + uw > 0 else 0.0
        hi = uy / (uy + lw) if uy + lw > 0 else 1.0
        nlo, nhi = ja.normal_ratio_counts(list(W), list(n), [float(v) for v in Np], [list(map(int, ct)) for ct in cts])
        return {"box": (lo <= theta <= hi, hi - lo), "normal": (nlo <= theta <= nhi, nhi - nlo)}
    return {"theta": theta, "draw": draw,
            "model": f"each stratum a population of {f} x n_h units with the observed shares, drawn without replacement"}


def _job(i):
    name, rep = _JOBS[i]
    c = _CASES[name]
    rng = np.random.default_rng([118, c["index"], rep])
    try:
        return name, c["draw"](rng)
    except Exception:
        return name, {"error": traceback.format_exc()}


def mc(rec, areas, R):
    t0 = time.time()
    G, N = ja.load_gfc(rec, areas), ja.load_nlcd(rec, 2016)
    okg = (G["gfc"] == G["ref"]).astype(float)
    okw = (G["wc"] == G["ref"]).astype(float)
    ap = (N["map"] == N["ref"]).astype(float)
    aa = ((N["map"] == N["ref"]) | (N["map"] == N["alt"])).astype(float)
    cases = {"gfc_oa_point_frame": ja.mc_linear_iid(G, okg),
             "gfc_minus_worldcover_point_frame": ja.mc_diff_iid(G, okg, okw),
             "nlcd_oa_primary_or_alternate_point_frame": ja.mc_linear_iid(N, aa),
             "nlcd_oa_primary_population_10n": ja.mc_linear_finite(N, ap, 10),
             "nlcd_user_52_box_population_10n": case_ratio_finite(N, 52, "user", 10),
             "nlcd_user_23_box_population_10n": case_ratio_finite(N, 23, "user", 10),
             "nlcd_producer_52_box_population_10n": case_ratio_finite(N, 52, "producer", 10),
             "nlcd_user_24_box_population_10n": case_ratio_finite(N, 24, "user", 10)}
    for i, (k, v) in enumerate(cases.items()):
        v["index"] = i
        _CASES[k] = v
    _JOBS[:] = [(k, rep) for k in cases for rep in range(R)]
    raw = {k: [] for k in cases}
    say(f"mc: {len(_JOBS)} draws on {PROCS} workers")
    with multiprocessing.get_context("fork").Pool(PROCS) as pool:
        for name, res in pool.imap_unordered(_job, range(len(_JOBS)), chunksize=4):
            raw[name].append(res)
    se = math.sqrt(0.95 * 0.05 / R)
    out = {"draws": R, "alpha": ALPHA, "gate": 0.95 - 3 * se,
           "gate_note": "coverage at least 0.95 - 3 Monte Carlo standard errors at 0.95"}
    for k, rows in raw.items():
        errs = [x["error"] for x in rows if "error" in x]
        good = [x for x in rows if "error" not in x]
        blk = {"theta": cases[k]["theta"], "model": cases[k]["model"], "errors": len(errs),
               "first_error": errs[0] if errs else None}
        for m in ("betting", "box", "normal"):
            v = [x[m] for x in good if m in x]
            if v:
                blk[m] = {"coverage": float(np.mean([bool(t[0]) for t in v])),
                          "mean_width_points": float(100 * np.mean([t[1] for t in v])), "draws": len(v)}
        blk["holds"] = (not errs) and all(blk[m]["coverage"] >= out["gate"] for m in ("betting", "box") if m in blk)
        out[k] = blk
    out["holds"] = all(v["holds"] for v in out.values() if isinstance(v, dict) and "holds" in v)
    out["seconds"] = time.time() - t0
    return out


if __name__ == "__main__":
    rec, areas, mode = sys.argv[1], sys.argv[2], sys.argv[3]
    out = widths(rec, areas) if mode == "widths" else mc(rec, areas, int(sys.argv[4]) if len(sys.argv) > 4 else 300)
    json.dump(out, sys.stdout, indent=1, default=float)
    sys.stdout.write("\n")
OEIX_FILE_END
export UV_CACHE_DIR=/scratch/qi_zim_neu/olmoearth_inferenceX/uv-cache
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export BETTING_PY="$SRC/betting_widths.py" BOUNDS_PY="$SRC/agency_bounds.py" JUDGE_PY="$SRC/judge_agency.py"
cd "$P/pkg" || exit 1
PY="$HOME/.local/bin/uv run --frozen --no-sync python -I"
timeout 3000 $PY "$D/ratio_box.py" "$P/rec" "$SRC/strata_size.csv" widths > "$D/box_widths.json" 2> "$D/box_widths.err"
echo "box widths exit $?"
timeout 3600 $PY "$D/ratio_box.py" "$P/rec" "$SRC/strata_size.csv" mc 300 > "$D/box_mc.json" 2> "$D/box_mc.err"
echo "box mc exit $?"
tail -3 "$D/box_widths.err" "$D/box_mc.err"
