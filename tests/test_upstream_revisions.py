"""exp/out/upstream_revisions.json: shaped right, honest about what it could not pin, and complete.

Offline, so it runs in CI. The check that earns its place is the last one: every upstream repository the code
fetches must appear in the record, so a new dependency cannot be added without pinning it — which is exactly how
the whole suite came to rest on an unrecorded revision of allenai/olmoearth-paper-embeddings."""
import importlib.util
import json
import os
import re

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location("upstream", os.path.join(ROOT, "scripts", "upstream_revision.py"))
upstream = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(upstream)

RECORD = upstream.load()
REPOS = RECORD["repos"]
SHA = re.compile(r"[0-9a-f]{40}")


@pytest.mark.parametrize("rid", sorted(REPOS))
def test_every_entry_is_shaped_right(rid):
    e = REPOS[rid]
    assert e["host"] in ("huggingface", "github"), e["host"]
    assert e["repo_type"] in ("model", "dataset", "torch.hub", "git"), e["repo_type"]   # git: a code or config repository
    assert e["carries"] and e["carries"][0].isalpha(), "an entry says what the repository carries"
    assert isinstance(e["named_at"], list) and isinstance(e["read_from"], list)


@pytest.mark.parametrize("rid", sorted(REPOS))
def test_a_revision_is_a_commit_sha_or_says_why_it_is_missing(rid):
    """The failure this forbids is a repository quietly dropped from the record because it was awkward."""
    e = REPOS[rid]
    if e["revision"] is None:
        assert e.get("why"), f"{rid} has no revision and no reason"
        assert not e["read_from"], f"{rid} claims a source but recorded no sha from it"
    else:
        assert SHA.fullmatch(e["revision"]), e["revision"]
        assert e["read_from"], f"{rid} has a sha and does not say which cache it came from"


def _resolve(path):
    """A package file is read from the package the tests import, so run against a built wheel this checks the file
    that ships; anything else from the checkout."""
    if path.startswith("oe_inferencex/"):
        import oe_inferencex
        return os.path.join(os.path.dirname(oe_inferencex.__file__), path[len("oe_inferencex/"):])
    return os.path.join(ROOT, path)


def test_the_line_numbers_it_cites_still_name_that_repository():
    """A cited file:line that has drifted is a citation to nothing, the defect the claim ledger exists to stop."""
    for rid, e in REPOS.items():
        for where in e["named_at"]:
            path, _, line = where.rpartition(":")
            with open(_resolve(path), encoding="utf-8") as f:
                text = f.readlines()[int(line) - 1]
            assert f'"{rid}"' in text, f"{where} no longer names {rid}: {text.strip()[:90]}"


def test_every_repository_the_code_fetches_is_in_the_record():
    named = upstream.repos_named_in_the_code()
    missing = sorted(set(named) - set(REPOS) - set(RECORD.get("not_repositories", {})))
    assert not missing, "fetched and unpinned: " + "; ".join(f"{r} ({', '.join(named[r])})" for r in missing)


def test_a_declared_non_repository_says_why_and_is_still_found_where_it_says():
    """An exception to the pin rule is a reasoned entry the scan still finds: a stale one (its literal gone from the
    code) or one shadowing a pinned repository fails."""
    named = upstream.repos_named_in_the_code()
    for rid, why in RECORD.get("not_repositories", {}).items():
        assert isinstance(why, str) and len(why) > 20, rid
        assert rid in named, f"{rid} is declared not a repository but the scan no longer finds it"
        assert rid not in REPOS


def test_the_package_side_is_scanned_from_the_installed_release():
    """The record has no copy of the package, so the scan reads the installed one; without that the check above
    would pass while scanning only exp/. The release names allenai/olmoearth_lcc in taskcard.py."""
    named = upstream.repos_named_in_the_code()
    assert any(w.startswith("oe_inferencex/taskcard.py:") for w in named.get("allenai/olmoearth_lcc", [])), named


def test_a_dict_key_is_not_a_repository(tmp_path):
    """exp92's class names are shaped like repo ids and sit in a module constant as dict keys; they are not
    repositories. A repo id as a constant's value, or as a dict value, is still found, and a package file is cited
    under oe_inferencex/ wherever the package is installed."""
    (tmp_path / "exp").mkdir()
    (tmp_path / "exp" / "a.py").write_text('CLASSES = {"Grass/Shrub": 3, "Snow/Ice" : 7}\n'
                                           'HUB = "owner/model"\nMODELS = {"base": "owner/other"}\n')
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "b.py").write_text('x = 1\nhf_hub_download("owner/data", "README.md")\n')
    assert upstream.repos_named_in_the_code(str(tmp_path), package=str(tmp_path / "pkg")) == {
        "owner/model": ["exp/a.py:2"], "owner/other": ["exp/a.py:3"], "owner/data": ["oe_inferencex/b.py:2"]}


def test_the_scan_reads_a_cache_the_way_the_record_was_built(tmp_path):
    """The reading path itself, on a cache laid out by hand: the record is only as good as this function."""
    ref = tmp_path / "hub" / "datasets--allenai--olmoearth-paper-embeddings" / "refs"
    ref.mkdir(parents=True)
    (ref / "main").write_text("6ea2c7973d2d6b79cd2221370bea6319f0eada68\n")
    assert upstream.scan(str(tmp_path)) == [
        ("dataset", "allenai/olmoearth-paper-embeddings", "6ea2c7973d2d6b79cd2221370bea6319f0eada68")]
    assert upstream.cached_revision(str(tmp_path), "allenai/olmoearth-paper-embeddings") == \
        REPOS["allenai/olmoearth-paper-embeddings"]["revision"]
    assert upstream.cached_revision(str(tmp_path), "allenai/OlmoEarth-v1-Base", "model") is None


@pytest.mark.parametrize("literal,is_repo", [
    ("allenai/olmoearth-paper-embeddings", True), ("125oii/dfc2020", True), ("gastruc/anysat", True),
    ("2024-06-01/2024-09-30", False), ("eval_settings/tiny_settings.enriched.json", False),
    ("exp/out/exp04_ckpt.npz", False), ("application/json", True)])
def test_the_repo_id_filter_separates_ids_from_dates_and_paths(literal, is_repo):
    """application/json is shaped exactly like a repo id, so only the fetcher-line gate keeps it out; the filter
    is not asked to do that job and the test says so rather than pretending otherwise."""
    assert upstream._is_repo_id(literal) is is_repo
