"""The claim ledger: load docs/claims.yaml, load the artifacts each claim rests on, evaluate the checks, find the
markers in the docs.

    python scripts/claims.py list [--status S] [--exp expNN]
    python scripts/claims.py stale
    python scripts/claims.py show <id>

Standard library plus PyYAML (a core dependency of the package). tests/test_claims.py imports this module.
"""
import argparse
import csv
import glob
import json
import os
import re
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REGISTRY = os.path.join(ROOT, "docs", "claims.yaml")
STATUSES = ("supported", "mixed", "rejected", "measured", "superseded")
DOC_GLOBS = ("docs/**/*.md", "README.md")
MARKER = re.compile(r"<!--\s*claim:([A-Za-z0-9_.-]+)\s*-->")
# The technical report tags a number with \claim{<id>}; tests/test_claims.py requires each tag to name a live claim.
REPORT_GLOBS = ("report/*.tex",)
REPORT_TAG = re.compile(r"\\claim\{([A-Za-z0-9_.-]+)\}")
REQUIRED = ("id", "statement", "status", "experiments", "artifacts", "check", "cited_in")

# The names a check may use besides A. Checks are expressions written by hand into the registry, not user input;
# the restricted namespace only keeps them honest about what they touch.
SAFE = {n: __builtins__[n] if isinstance(__builtins__, dict) else getattr(__builtins__, n)
        for n in ("abs", "all", "any", "float", "int", "len", "max", "min", "round", "sorted", "sum", "set",
                  "str", "list", "dict", "tuple", "isinstance", "enumerate", "zip", "range")}
SAFE["log"] = __import__("math").log        # kept for checks written before oracle_aurc was available here
# The perfect ranker's AURC, the package's own exact finite-n function rather than its n -> infinity limit. A check
# that needs the ranking headroom can now RECOMPUTE it from an experiment's per-task records instead of reading a
# pre-baked field out of the artifact it is meant to be testing. The asymptotic form e + (1 - e) log(1 - e) differs
# by O(1/n) and moved a published figure once (2026-09-21), so the two must not be mixed inside one fraction.
SAFE["oracle_aurc"] = __import__("oe_inferencex.metrics", fromlist=["metrics"]).oracle_aurc
SAFE["median"] = lambda v: __import__("statistics").median(list(v))
SAFE["attainable_ceiling"] = __import__("oe_inferencex.metrics", fromlist=["metrics"]).attainable_ceiling
SAFE["augrc_from_auroc"] = __import__("oe_inferencex.metrics", fromlist=["metrics"]).augrc_from_auroc


def load_registry(path=REGISTRY):
    """The list of claim entries, each validated for shape (not for truth; that is `evaluate`)."""
    with open(path, encoding="utf-8") as f:
        claims = yaml.safe_load(f)
    if not isinstance(claims, list):
        raise ValueError(f"{path}: expected a list of claims")
    for c in claims:
        missing = [k for k in REQUIRED if k not in c]
        if missing:
            raise ValueError(f"claim {c.get('id')!r} lacks {missing}")
        if c["status"] not in STATUSES:
            raise ValueError(f"claim {c['id']!r}: status {c['status']!r} not in {STATUSES}")
        if c["status"] == "superseded" and not c.get("superseded_by"):
            raise ValueError(f"claim {c['id']!r}: superseded without superseded_by")
        if c.get("superseded_by") and c["status"] != "superseded":
            raise ValueError(f"claim {c['id']!r}: superseded_by set but status is {c['status']!r}")
    return claims


def load_artifact(rel):
    """A JSON file as parsed, a CSV as a list of row dicts (strings); paths are relative to the repository root."""
    path = os.path.join(ROOT, rel)
    if rel.endswith(".json"):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    if rel.endswith(".csv"):
        with open(path, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    raise ValueError(f"{rel}: only .json and .csv artifacts can be checked")


def load_artifacts(claims):
    """Every artifact any claim references, loaded once: {relative path: data}."""
    A = {}
    for c in claims:
        for rel in c["artifacts"]:
            if rel not in A:
                A[rel] = load_artifact(rel)
    return A


def evaluate(claim, A):
    """The claim's check evaluated over the loaded artifacts; True when the artifacts still say what the claim says."""
    # names go in the globals: a lambda inside a check resolves its free names there, not in the locals
    return eval(claim["check"], {"__builtins__": {}, **SAFE, "A": A})


def doc_files(root=ROOT):
    """The Markdown files that may carry markers, as paths relative to the root."""
    out = []
    for pattern in DOC_GLOBS:
        out.extend(os.path.relpath(p, root) for p in glob.glob(os.path.join(root, pattern), recursive=True))
    return sorted(set(out))


def markers(root=ROOT):
    """{relative file: [(1-based line number, claim id)]} over every marker in the docs."""
    found = {}
    for rel in doc_files(root):
        with open(os.path.join(root, rel), encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                for m in MARKER.finditer(line):
                    found.setdefault(rel, []).append((i, m.group(1)))
    return found


def report_tags(root=ROOT):
    """{relative file: [(1-based line number, claim id)]} over every \\claim{} tag in the technical report."""
    found = {}
    for pattern in REPORT_GLOBS:
        for path in sorted(glob.glob(os.path.join(root, pattern))):
            rel = os.path.relpath(path, root)
            with open(path, encoding="utf-8") as f:
                for i, line in enumerate(f, 1):
                    if line.lstrip().startswith("%"):
                        continue
                    for m in REPORT_TAG.finditer(line):
                        found.setdefault(rel, []).append((i, m.group(1)))
    return found


def paragraph(lines, i):
    """The paragraph (blank-line delimited; a table row is its own line) around 0-based line i, as one string."""
    if lines[i].lstrip().startswith("|"):
        return lines[i]
    lo, hi = i, i
    while lo > 0 and lines[lo - 1].strip():
        lo -= 1
    while hi + 1 < len(lines) and lines[hi + 1].strip():
        hi += 1
    return "\n".join(lines[lo:hi + 1])


def stale(claims, A, root=ROOT):
    """The claims whose check fails or whose status is superseded, with the locations that cite them."""
    out = []
    for c in claims:
        try:
            ok = bool(evaluate(c, A))
            err = None
        except Exception as e:  # a missing key is a failure, not a crash: the artifact changed shape
            ok, err = False, f"{type(e).__name__}: {e}"
        if ok and c["status"] != "superseded":
            continue
        reason = "check fails" if not ok else f"superseded by {c['superseded_by']}"
        if err:
            reason += f" ({err})"
        out.append((c, reason))
    return out


def _cite_lines(claim, found, tags=None):
    locs = []
    for cite in claim["cited_in"]:
        lines = [ln for ln, cid in found.get(cite["file"], []) if cid == claim["id"]]
        locs.append(f"{cite['file']}:{','.join(map(str, lines)) or '?'}")
    for rel, v in (tags or {}).items():
        lines = [ln for ln, cid in v if cid == claim["id"]]
        if lines:
            locs.append(f"{rel}:{','.join(map(str, lines))}")
    return locs


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    ls = sub.add_parser("list", help="id, status, experiments and citing files per claim")
    ls.add_argument("--status", choices=STATUSES)
    ls.add_argument("--exp", help="only claims resting on this experiment, e.g. exp49")
    sub.add_parser("stale", help="re-evaluate every check; print failing or superseded claims with their citations")
    sh = sub.add_parser("show", help="print the full entry")
    sh.add_argument("id")
    a = p.parse_args(argv)

    claims = load_registry()
    if a.cmd == "list":
        found = markers()
        for c in claims:
            if a.status and c["status"] != a.status:
                continue
            if a.exp and a.exp not in c["experiments"]:
                continue
            files = sorted({x["file"] for x in c["cited_in"]})
            print(f"{c['id']:<44} {c['status']:<10} {','.join(c['experiments']):<28} {' '.join(files)}")
        return 0
    if a.cmd == "show":
        for c in claims:
            if c["id"] == a.id:
                print(yaml.safe_dump(c, sort_keys=False, allow_unicode=True, width=110))
                return 0
        print(f"no claim {a.id!r}", file=sys.stderr)
        return 1
    if a.cmd == "stale":
        A = load_artifacts(claims)
        found = markers()
        tags = report_tags()
        bad = stale(claims, A)
        if not bad:
            print(f"all {len(claims)} claims hold against the committed artifacts")
            return 0
        print("sentences to rewrite:")
        for c, reason in bad:
            print(f"- {c['id']} [{c['status']}]: {reason}")
            print(f"    {c['statement']}")
            for loc in _cite_lines(c, found, tags):
                print(f"    cited in {loc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
