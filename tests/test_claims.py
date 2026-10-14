"""The claim ledger (docs/claims.yaml) against the committed artifacts and the docs that cite it.

Every claim's check must hold on the artifacts under exp/out/, every citation must have its marker in the named
file, every marker in the docs must name a registered claim, a superseded claim's citing paragraph must name the
run that superseded it, and every bold-verdict row of the technique ledger must carry a marker."""
import importlib.util
import os
import re

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location("claims_cli", os.path.join(ROOT, "scripts", "claims.py"))
claims_cli = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(claims_cli)

CLAIMS = claims_cli.load_registry()
ARTIFACTS = claims_cli.load_artifacts(CLAIMS)
MARKERS = claims_cli.markers()
BY_ID = {c["id"]: c for c in CLAIMS}


def _read(rel):
    with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
        return f.read().split("\n")


def test_ids_are_unique_and_statuses_allowed():
    ids = [c["id"] for c in CLAIMS]
    assert len(ids) == len(set(ids)), sorted(i for i in ids if ids.count(i) > 1)
    assert all(re.fullmatch(r"[a-z0-9][a-z0-9-]*", i) for i in ids), [i for i in ids if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", i)]
    assert all(c["status"] in claims_cli.STATUSES for c in CLAIMS)
    assert all((c["status"] == "superseded") == bool(c.get("superseded_by")) for c in CLAIMS)
    # two digits up to exp99, three from exp100 on
    assert all(re.fullmatch(r"exp(\d\d|[1-9]\d\d)", e) for c in CLAIMS for e in c["experiments"]), "experiments are expNN ids"
    assert all(c["artifacts"] or c["check"].strip() == "True" for c in CLAIMS), "a claim without artifacts has a trivial check"
    assert all(c.get("note") for c in CLAIMS if not c["artifacts"]), "a claim without artifacts says so in its note"


def test_every_artifact_is_committed_under_exp_out():
    for c in CLAIMS:
        for rel in c["artifacts"]:
            assert rel.startswith("exp/out/") and os.path.exists(os.path.join(ROOT, rel)), (c["id"], rel)
            assert rel in ARTIFACTS


@pytest.mark.parametrize("claim", CLAIMS, ids=[c["id"] for c in CLAIMS])
def test_check_holds_on_the_artifacts(claim):
    """The artifacts still say what the statement says; on failure the claim id names the sentences to rewrite."""
    try:
        result = claims_cli.evaluate(claim, ARTIFACTS)
    except Exception as e:  # noqa: BLE001 - a changed artifact shape is a stale claim, reported by id
        pytest.fail(f"claim {claim['id']!r}: check raised {type(e).__name__}: {e}")
    assert result is True, f"claim {claim['id']!r} no longer holds: {claim['statement']}"


def test_every_citation_has_its_marker():
    missing = [(c["id"], x["file"]) for c in CLAIMS for x in c["cited_in"]
               if x["marker"] != c["id"] or c["id"] not in {cid for _, cid in MARKERS.get(x["file"], [])}]
    assert not missing, missing


def test_every_marker_names_a_registered_claim():
    unknown = [(f, ln, cid) for f, v in MARKERS.items() for ln, cid in v if cid not in BY_ID]
    assert not unknown, unknown


def test_every_marker_is_a_declared_citation():
    """A marker in a file the registry does not list for that claim is a citation the ledger does not know about."""
    undeclared = [(f, ln, cid) for f, v in MARKERS.items() for ln, cid in v
                  if cid in BY_ID and f not in {x["file"] for x in BY_ID[cid]["cited_in"]}]
    assert not undeclared, undeclared


def test_every_report_tag_names_a_live_claim():
    """The technical report (report/*.tex) tags numbers with \\claim{id}: each tag must name a registered claim that is
    not superseded, so a report sentence cannot rest on a withdrawn or unknown result."""
    tags = claims_cli.report_tags()
    assert tags, "the report carries no claim tags"
    unknown = [(f, ln, cid) for f, v in tags.items() for ln, cid in v if cid not in BY_ID]
    superseded = [(f, ln, cid) for f, v in tags.items() for ln, cid in v
                  if cid in BY_ID and BY_ID[cid]["status"] == "superseded"]
    assert not unknown, unknown
    assert not superseded, superseded


def test_every_claim_is_cited_somewhere():
    assert all(c["cited_in"] for c in CLAIMS), [c["id"] for c in CLAIMS if not c["cited_in"]]


def test_superseded_claims_name_the_superseding_run_next_to_the_marker():
    """Where a superseded claim is cited, the same paragraph or table row mentions the experiment that overturned it."""
    bad = []
    for c in CLAIMS:
        if c["status"] != "superseded":
            continue
        for x in c["cited_in"]:
            lines = _read(x["file"])
            for ln, cid in MARKERS.get(x["file"], []):
                if cid == c["id"] and c["superseded_by"] not in claims_cli.paragraph(lines, ln - 1):
                    bad.append((c["id"], x["file"], ln))
    assert not bad, bad


def test_bold_verdict_rows_of_the_technique_ledger_carry_a_marker():
    bare = []
    for ln, line in enumerate(_read("docs/TECHNIQUES.md"), 1):
        if not line.startswith("|") or re.fullmatch(r"\|(\s*-+\s*\|)+\s*", line):
            continue
        cells = [x.strip() for x in claims_cli.MARKER.sub("", line).strip().strip("|").split("|")]
        if any(re.fullmatch(r"\*\*.+\*\*", cell) for cell in cells) and not claims_cli.MARKER.search(line):
            bare.append((ln, cells[0][:60]))
    assert not bare, bare


def test_markers_sit_at_the_end_of_their_line():
    """The marker convention: comments after the sentence or after the row's final pipe, nothing after them."""
    bad = []
    for rel, found in MARKERS.items():
        lines = _read(rel)
        for ln, _ in found:
            tail = claims_cli.MARKER.split(lines[ln - 1])[-1]
            if tail.strip():
                bad.append((rel, ln, tail.strip()[:40]))
    assert not bad, bad


def test_stale_lists_only_superseded_claims_today():
    """The CLI's stale view (what to rewrite after a run) is empty except for the claims already marked superseded."""
    out = claims_cli.stale(CLAIMS, ARTIFACTS)
    assert {c["id"] for c, _ in out} == {c["id"] for c in CLAIMS if c["status"] == "superseded"}


def test_claims_from_exp79_on_name_an_independent_crosscheck():
    """A claim's check reads its number back from the artifact that produced it, which catches drift and cannot
    catch a formula that was wrong from the start. Three recorded results were corrected on 21-22 September for
    that reason. From exp79 on, a claim must also name a test that reaches the same number by another route - a
    known answer, an enumeration, or a second artifact - as `crosscheck: tests/<file>.py::<test>`."""
    for c in CLAIMS:
        x = c.get("crosscheck")
        if any(int(e[3:]) >= 79 for e in c["experiments"]):
            assert x, f"claim {c['id']!r} (from {c['experiments']}) names no independent crosscheck"
        if not x:
            continue                                   # older claims may carry one; any that is named must exist
        path, _, name = x.partition("::")
        with open(os.path.join(ROOT, path), encoding="utf-8") as f:
            assert f"def {name}(" in f.read(), f"claim {c['id']!r}: crosscheck {x} does not exist"
