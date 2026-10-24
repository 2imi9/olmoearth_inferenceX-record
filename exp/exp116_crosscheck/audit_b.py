# Written on 24 October 2026 by an independent agent (workflow wf_8dbe7de7-5db) from the frozen plan
# (docs/plan/target_class_certificate.md), the report and the package's Usage alone, without the run's script or the
# package code; paths made relative to the record when it was committed, nothing else changed.
"""Independent second route for exp116 arm B (class design vs SRS).

Built from the frozen plan and docs/Usage.md 'Sampling by map class' only.
Does not import oe_inferencex.
"""
import json, os, sys, zlib, time
import numpy as np
from scipy.stats import hypergeom

ROOT = os.environ.get('EXP79_UNITS', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'out', 'exp79_units'))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'out', 'exp116_ab.json')
BUDGET, DRAWS, ALPHA = 400, 400, 0.05
STRICT = True   # tail probability must exceed the tail level (">"); False uses ">="


def allocation(sizes, budget=BUDGET):
    """sizes: dict class -> N_c (classes in increasing id). floor = min(50, budget // (2K)).
    Each class gets min(floor, N_c); the rest in proportion to N_c, largest remainder, capped at N_c."""
    cls = sorted(sizes)
    K = len(cls)
    floor = min(50, budget // (2 * K))
    a = {c: min(floor, sizes[c]) for c in cls}
    rest = budget - sum(a.values())
    active = [c for c in cls if sizes[c] > a[c]]
    fixed_extra = {c: 0 for c in cls}
    # remove classes whose proportional extra would exceed their room, iteratively
    while True:
        tot = sum(sizes[c] for c in active)
        if tot == 0 or rest <= 0:
            q = {c: 0.0 for c in active}
            break
        q = {c: rest * sizes[c] / tot for c in active}
        over = [c for c in active if q[c] > sizes[c] - a[c]]
        if not over:
            break
        for c in over:
            fixed_extra[c] = sizes[c] - a[c]
            rest -= sizes[c] - a[c]
            active.remove(c)
    flo = {c: int(np.floor(q[c])) for c in active}
    left = rest - sum(flo.values())
    order = sorted(active, key=lambda c: (-(q[c] - flo[c]), c))
    for c in order[:left]:
        flo[c] += 1
    for c in active:
        fixed_extra[c] = flo[c]
    alloc = {c: a[c] + fixed_extra[c] for c in cls}
    assert sum(alloc.values()) == min(budget, sum(sizes.values())), (alloc, sizes)
    assert all(alloc[c] <= sizes[c] for c in cls)
    return floor, alloc


def hyper_bounds(N, n, k, tail):
    """Exact two-sided hypergeometric bounds on M (successes in a population of N), from k successes
    in a simple random sample of n, each tail at level `tail`. Vectorized over arrays."""
    N = np.asarray(N, dtype=np.int64); n = np.asarray(n, dtype=np.int64)
    k = np.asarray(k, dtype=np.int64); tail = np.asarray(tail, dtype=float)
    N, n, k, tail = np.broadcast_arrays(N, n, k, tail)
    N = N.ravel().copy(); n = n.ravel().copy(); k = k.ravel().copy(); tail = tail.ravel().copy()
    cmp = (lambda p, t: p > t) if STRICT else (lambda p, t: p >= t)
    # lower: smallest M in [k, N-n+k] with P(X >= k | M) passing
    lo = k.copy(); hi = N - n + k
    act = lo < hi
    while act.any():
        i = np.nonzero(act)[0]
        mid = (lo[i] + hi[i]) // 2
        p = hypergeom.sf(k[i] - 1, N[i], mid, n[i])
        ok = cmp(p, tail[i])
        hi[i] = np.where(ok, mid, hi[i]); lo[i] = np.where(ok, lo[i], mid + 1)
        act = lo < hi
    lower = lo
    lo = k.copy(); hi = N - n + k
    act = lo < hi
    while act.any():
        i = np.nonzero(act)[0]
        mid = (lo[i] + hi[i] + 1) // 2
        p = hypergeom.cdf(k[i], N[i], mid, n[i])
        ok = cmp(p, tail[i])
        lo[i] = np.where(ok, mid, lo[i]); hi[i] = np.where(ok, hi[i], mid - 1)
        act = lo < hi
    upper = lo
    # n == 0: no information
    z = n == 0
    lower[z] = 0; upper[z] = N[z]
    return lower, upper


class Bounds:
    """Cache of hyper_bounds keyed by (N, n, k, tail)."""
    def __init__(self):
        self.c = {}
    def get(self, keys):
        miss = [q for q in set(keys) if q not in self.c]
        if miss:
            a = np.array(miss, dtype=object)
            lo, hi = hyper_bounds(a[:, 0].astype(np.int64), a[:, 1].astype(np.int64),
                                  a[:, 2].astype(np.int64), a[:, 3].astype(float))
            for q, l, h in zip(miss, lo, hi):
                self.c[q] = (int(l), int(h))
        return [self.c[q] for q in keys]


def run_map(name, seed_name=None):
    z = np.load(os.path.join(ROOT, name))
    dec = z['dec'].astype(np.int64); y = z['y'].astype(np.int64)
    nc = int(z['n_classes'][0])
    N = len(dec)
    cls = sorted(int(c) for c in np.unique(dec))
    K = len(cls)
    idx = {c: np.nonzero(dec == c)[0] for c in cls}          # increasing order
    sizes = {c: len(idx[c]) for c in cls}
    floor, alloc = allocation(sizes)
    crc = zlib.crc32((seed_name or name).encode())
    # truth
    Lr = max(nc, int(y.max()) + 1)
    Mtrue = np.zeros((K, Lr), dtype=np.int64)               # stratum x reference class
    for h, c in enumerate(cls):
        Mtrue[h] = np.bincount(y[idx[c]], minlength=Lr)
    ua_true = {c: Mtrue[h, c] / sizes[c] for h, c in enumerate(cls)}
    share_true = Mtrue.sum(0) / N
    pa_true = np.full(Lr, np.nan)
    for j in range(Lr):
        tot = Mtrue[:, j].sum()
        if tot > 0:
            pa_true[j] = (Mtrue[cls.index(j), j] if j in cls else 0) / tot
    err_true = float((dec != y).mean())
    S = sum(1 for c in cls if alloc[c] < sizes[c])
    B = Bounds()
    Lvariants = {'L=n_classes': nc, 'L=K': K,
                 'L=classes in map or reference': len(set(cls) | set(int(v) for v in np.unique(y)))}
    # class design draws
    rng = np.random.default_rng(np.random.SeedSequence([116, crc, 1]))
    ua_cov = {c: 0 for c in cls}; ua_w = {c: 0.0 for c in cls}
    joint_cov = {k: 0 for k in Lvariants}; counts_all_hold = {k: 0 for k in Lvariants}
    overall_cov = 0
    for d in range(DRAWS):
        kcount = np.zeros((K, Lr), dtype=np.int64)
        for h, c in enumerate(cls):
            s = rng.choice(idx[c], alloc[c], replace=False)
            kcount[h] = np.bincount(y[s], minlength=Lr)
        # UA per class at 95%
        keys = [(sizes[c], alloc[c], int(kcount[h, c]), ALPHA / 2) for h, c in enumerate(cls)]
        bb = B.get(keys)
        for h, c in enumerate(cls):
            l, u = bb[h]
            ua_cov[c] += (l <= Mtrue[h, c] <= u)
            ua_w[c] += (u - l) / sizes[c]
        # overall: union interval over strata of error counts at 1 - 0.05/S
        if S > 0:
            keys = [(sizes[c], alloc[c], int(alloc[c] - kcount[h, c]), ALPHA / S / 2) for h, c in enumerate(cls)]
            bb = B.get(keys)
            elo = sum(b[0] for b in bb) / N; ehi = sum(b[1] for b in bb) / N
        else:
            elo = ehi = err_true
        overall_cov += (elo - 1e-12 <= err_true <= ehi + 1e-12)
        # joint: per stratum and reference class count intervals at 1 - 0.05/(L*S)
        for lv, L in Lvariants.items():
            if S == 0:
                joint_cov[lv] += 1; counts_all_hold[lv] += 1; continue
            t = ALPHA / (L * S) / 2
            keys = [(sizes[c], alloc[c], int(kcount[h, j]), t) for h, c in enumerate(cls) for j in range(Lr)]
            bb = np.array(B.get(keys)).reshape(K, Lr, 2)
            lo = bb[..., 0]; hi = bb[..., 1]
            counts_all_hold[lv] += bool(np.all((lo <= Mtrue) & (Mtrue <= hi)))
            ok = True
            sl = lo.sum(0) / N; sh = hi.sum(0) / N
            if np.any(share_true < sl - 1e-12) or np.any(share_true > sh + 1e-12):
                ok = False
            for j in range(Lr):
                if np.isnan(pa_true[j]):
                    continue
                if j in cls:
                    hj = cls.index(j)
                    a_lo, a_hi = lo[hj, j], hi[hj, j]
                    o_lo = lo[:, j].sum() - a_lo; o_hi = hi[:, j].sum() - a_hi
                else:
                    a_lo = a_hi = 0; o_lo = lo[:, j].sum(); o_hi = hi[:, j].sum()
                p_lo = a_lo / (a_lo + o_hi) if a_lo + o_hi > 0 else 0.0
                p_hi = a_hi / (a_hi + o_lo) if a_hi + o_lo > 0 else 1.0
                if not (p_lo - 1e-12 <= pa_true[j] <= p_hi + 1e-12):
                    ok = False
            joint_cov[lv] += ok
    # SRS draws
    rng2 = np.random.default_rng(np.random.SeedSequence([116, crc, 2]))
    srs_w = {c: 0.0 for c in cls}; srs_cov = {c: 0 for c in cls}; srs_none = {c: 0 for c in cls}
    for d in range(DRAWS):
        s = rng2.choice(N, BUDGET, replace=False)
        ds = dec[s]; ys = y[s]
        keys = []; who = []
        for h, c in enumerate(cls):
            m = ds == c; n_c = int(m.sum())
            if n_c == 0:
                srs_w[c] += 1.0; srs_none[c] += 1; srs_cov[c] += 1
                continue
            keys.append((sizes[c], n_c, int((ys[m] == c).sum()), ALPHA / 2)); who.append((h, c))
        for (h, c), (l, u) in zip(who, B.get(keys)):
            srs_w[c] += (u - l) / sizes[c]
            srs_cov[c] += (l <= Mtrue[h, c] <= u)
    res = {'map': name, 'n_units': N, 'n_map_classes': K, 'n_classes': nc, 'floor': floor, 'S': S,
           'overall_coverage': overall_cov / DRAWS,
           'joint_coverage': {k: v / DRAWS for k, v in joint_cov.items()},
           'all_count_intervals_hold': {k: v / DRAWS for k, v in counts_all_hold.items()},
           'classes': [{'class': c, 'share_of_map': sizes[c] / N, 'n_units': sizes[c], 'allocated': alloc[c],
                        'ua_true': ua_true[c],
                        'ua_coverage': ua_cov[c] / DRAWS, 'ua_width_class': ua_w[c] / DRAWS,
                        'ua_width_srs': srs_w[c] / DRAWS, 'srs_ua_coverage': srs_cov[c] / DRAWS,
                        'srs_draws_none_labelled': srs_none[c]} for c in cls]}
    return res


if __name__ == '__main__':
    names = sys.argv[1:]
    out = []
    for nm in names:
        t0 = time.time()
        r = run_map(nm)
        r['seconds'] = round(time.time() - t0, 1)
        print(nm, r['seconds'], 's', flush=True)
        out.append(r)
    json.dump(out, open(os.environ.get('AUDIT_OUT', 'audit_b_result.json'), 'w'), indent=1)
