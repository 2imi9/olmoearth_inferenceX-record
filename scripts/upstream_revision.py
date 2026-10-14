"""Which upstream revision the record was measured against, and whether upstream has moved since.

    python scripts/upstream_revision.py scan              # "repo_type repo_id sha" lines from a local HF cache
    python scripts/upstream_revision.py scan --cache DIR
    python scripts/upstream_revision.py check             # network: the pinned revisions against main today

Every experiment here calls `hf_hub_download` or `snapshot_download` with no `revision=`, so a run took whatever
`main` pointed at that day and no artifact recorded which bytes those were. `exp/out/upstream_revisions.json`
closes that: one commit sha per upstream repository the record rests on, read out of the Hugging Face caches the
runs actually used.

That file is a measurement of someone else's system at a point in time, not a statistic, so unlike
`headroom_by_encoder.json` it has no generator and cannot have one: it can only be re-measured. `scan` is how it
was measured and how to measure it again, and the record names the host and the command behind every reading.

`check` is the half that earns its keep later. It asks the Hub what `main` points at now and names every
repository that has moved, which is the question "is a fresh run comparable to the recorded one?" — the question
that had no answer at all until 21 September 2026. It also names any repository the code fetches that the record
does not pin, so a new upstream dependency cannot be added silently; `tests/test_upstream_revisions.py` asserts
the same thing offline.

Standard library plus huggingface_hub (a core dependency of the package) for `check` only; `scan` needs neither.
"""
import argparse
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RECORD = os.path.join(ROOT, "exp", "out", "upstream_revisions.json")

# Finding every upstream repository the code fetches. Two things have to be true of a line, and both are needed:
# it reaches a model hub, and the literal on it is shaped like a repo id rather than a date, a media type or a
# path. Written narrow on purpose — this is the forcing function for `check`, so a false positive is noise a
# reader will learn to ignore and a false negative is the defect it exists to catch.
SCANNED = ("exp/*.py", "oe_inferencex/*.py")
FETCHERS = ("hf_hub_download", "snapshot_download", "load_model_from_repo_id", "torch.hub.load")
CONSTANT = re.compile(r"^[A-Z][A-Z0-9_]*(\s*,\s*[A-Z][A-Z0-9_]*)*\s*=")   # HUB = "...", REPO, TREE = "...", ...
LITERAL = re.compile(r'"([^"\s]{2,}/[^"\s]{2,})"')
EXTENSION = re.compile(r"\.[A-Za-z]{2,5}$")


def _is_repo_id(s):
    """A `owner/name` a hub would accept, as opposed to `2024-06-01/2024-09-30` or `eval_settings/x.json`."""
    owner, _, name = s.partition("/")
    return ("/" not in name and any(c.isalpha() for c in owner)
            and not EXTENSION.search(name) and not EXTENSION.search(owner))


def load(path=RECORD):
    """The pinned revisions: {repo_id: entry}."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def cached_revision(cache_dir, repo_id, repo_type="dataset"):
    """The commit sha a local Hugging Face cache holds for one repo, or None if it does not hold it.

    Reads `<cache>/[hub/]<type>s--owner--name/refs/main`, the pointer the Hub client itself writes. No network.
    """
    folder = f"{repo_type}s--" + repo_id.replace("/", "--")
    for base in (cache_dir, os.path.join(cache_dir, "hub")):
        ref = os.path.join(base, folder, "refs", "main")
        if os.path.exists(ref):
            with open(ref, encoding="utf-8") as f:
                return f.read().strip()
    return None


def scan(cache_dir):
    """[(repo_type, repo_id, sha)] for everything a local Hugging Face cache holds, sorted."""
    out = []
    for base, _, _ in os.walk(cache_dir):
        if os.path.basename(base) != "refs":
            continue
        folder = os.path.basename(os.path.dirname(base))
        if "--" not in folder:
            continue
        kind, _, rest = folder.partition("--")
        if kind not in ("models", "datasets"):
            continue
        main = os.path.join(base, "main")
        if os.path.exists(main):
            with open(main, encoding="utf-8") as f:
                out.append((kind[:-1], rest.replace("--", "/"), f.read().strip()))
    return sorted(set(out))


def repos_named_in_the_code(root=ROOT):
    """{repo_id: [file:line]} for every upstream repository the code fetches by name.

    A repo id reaches a hub one of two ways here: inline in the call, or as a module constant the call is given
    a few lines later. Both are literals, so one pass over fetcher lines and top-level constant assignments
    finds both. A literal used as a dict key is a name, not an argument a fetcher is given: exp92's class table
    (`CLASSES = {"Grass/Shrub": 3, ...}`) is shaped like repo ids and reaches no hub.
    """
    import glob
    found = {}
    for pattern in SCANNED:
        for path in sorted(glob.glob(os.path.join(root, pattern))):
            with open(path, encoding="utf-8") as f:
                for i, line in enumerate(f, 1):
                    if not (any(fn in line for fn in FETCHERS) or CONSTANT.match(line)):
                        continue
                    for m in LITERAL.finditer(line):
                        if _is_repo_id(m.group(1)) and not line[m.end():].lstrip().startswith(":"):
                            found.setdefault(m.group(1), []).append(f"{os.path.relpath(path, root)}:{i}")
    return found


def cmd_scan(args):
    cache = args.cache or os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    rows = scan(cache)
    if not rows:
        print(f"no Hugging Face cache under {cache}", file=sys.stderr)
        return 1
    print(f"# {cache}")
    for kind, rid, sha in rows:
        print(f"{kind} {rid} {sha}")
    return 0


def cmd_check(args):
    from huggingface_hub import HfApi

    rec = load()
    api, moved, unpinned = HfApi(), [], []
    print(f"{'repository':42s} {'pinned':10s} {'main today':10s}")
    for rid, e in sorted(rec["repos"].items()):
        pinned = e["revision"]
        if pinned is None:
            unpinned.append(rid)
            print(f"{rid:42s} {'-':10s} {'-':10s}  not pinned")
            continue
        if e["host"] != "huggingface":                            # torch.hub reaches GitHub, not the Hub
            print(f"{rid:42s} {pinned[:8]:10s} {'-':10s}  on {e['host']}, not checked here")
            continue
        try:
            now = api.repo_info(rid, repo_type=e["repo_type"]).sha
        except Exception as exc:                                  # gated, renamed, withdrawn, or offline
            print(f"{rid:42s} {pinned[:8]:10s} {'?':10s}  {type(exc).__name__}: {str(exc)[:60]}")
            continue
        if now != pinned:
            moved.append(rid)
        print(f"{rid:42s} {pinned[:8]:10s} {now[:8]:10s}  {'MOVED' if now != pinned else ''}")

    named = repos_named_in_the_code()
    missing = sorted(set(named) - set(rec["repos"]))
    if missing:
        print("\nfetched by the code and not in the record:")
        for rid in missing:
            print(f"  {rid}  ({', '.join(named[rid])})")
    if unpinned:
        print(f"\nin the record with no revision: {', '.join(unpinned)} (each says why)")
    if moved:
        print(f"\n{len(moved)} moved since the record was measured: {', '.join(moved)}")
        print("A fresh run against these is not comparable to the recorded numbers until it is re-measured and")
        print("the difference is attributed. Re-measure with `scan` on the cache the run used.")
    else:
        print(f"\nnothing has moved; {len(rec['repos'])} repositories are where the record left them")
    return 1 if (moved or missing) else 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan", help="read revisions out of a local Hugging Face cache; no network")
    s.add_argument("--cache", default=None, help="cache directory (default: $HF_HOME, else ~/.cache/huggingface)")
    sub.add_parser("check", help="compare the record against main on the Hub today; exits 1 if anything moved")
    args = p.parse_args(argv)
    return cmd_scan(args) if args.cmd == "scan" else cmd_check(args)


if __name__ == "__main__":
    sys.exit(main())
