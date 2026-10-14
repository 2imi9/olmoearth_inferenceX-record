"""Merge a partial exp79 export into an encoder's export. When a devel-partition job hits its 4-hour limit the
export holds the tasks it finished (the JSON is rewritten after every task), and a follow-up job with
`--only <task>` writes the missing task to a separate file. This puts the two together, refusing to overwrite a
task both carry unless told to, so that no task's ten seeds come from two different jobs by accident.

    python scripts/merge_exp79_partial.py exp/out/exp79_seeds/anysat.json exp/out/exp79_seeds/anysat_m_sa_crop_type.json
"""
import json
import sys


def merge(base_path, part_path, overwrite=False):
    base, part = json.load(open(base_path)), json.load(open(part_path))
    if base["model"] != part["model"] or base["n_seeds"] != part["n_seeds"]:
        raise SystemExit(f"{base_path} is {base['model']} x {base['n_seeds']} seeds, {part_path} is {part['model']} x {part['n_seeds']}")
    if base.get("probe_lrs") != part.get("probe_lrs"):
        raise SystemExit("the two exports used different probe learning rates; they are not one run")
    added, kept = [], []
    for t, v in part["tasks"].items():
        if t in base["tasks"] and not overwrite:
            kept.append(t)
            continue
        base["tasks"][t] = v
        added.append(t)
    absent = {a["task"]: a for a in base.get("absent", [])}
    for a in part.get("absent", []):
        absent.setdefault(a["task"], a)
    base["absent"] = [a for t, a in absent.items() if t not in base["tasks"]]
    base.setdefault("merged_from", []).append({"file": part_path, "tasks": added})
    with open(base_path, "w") as f:
        json.dump(base, f, indent=1, default=float)
    print(f"{base_path}: added {added}, kept {kept}, now {len(base['tasks'])} tasks, absent {[a['task'] for a in base['absent']]}")


if __name__ == "__main__":
    merge(sys.argv[1], sys.argv[2], overwrite="--overwrite" in sys.argv[3:])
