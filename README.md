olmoearth_inferenceX-record
===========================

The research record behind the olmoearth-inferencex package: the experiments, their committed outputs, the claim
ledger that ties each stated result to an output, the pages that report the results, and the technical report.

The package, its documentation and its own tests are in https://github.com/2imi9/olmoearth_inferenceX
(documentation: https://olmoearth-inferencex.readthedocs.io; releases: https://pypi.org/project/olmoearth-inferencex/).

A five-minute summary of what was done and found, for the OlmoEarth team: [CATCH_UP.md](CATCH_UP.md).

Source
------

Split on 14 October 2026 from 2imi9/olmoearth_inferenceX at commit 0d1161a09927cfd92091c7db945672a5bedead32
(release 1.12.1). At the split the files were as they were at that commit, with these changes: the thirteen split test files below;
the links that now cross between the two repositories; the claim citations that named the package's pages, with the
two demo claims now cited in `docs/Findings.md`; the report's title block and availability section, with the PDF
rebuilt; the upstream-revision check (`scripts/upstream_revision.py` and its test), which now reads the package's
files from the installed release; and `.gitignore`. Their earlier history is in that repository. Experiments archived
before the split are in its v1.9.0 tree. The record continues here: exp110 to exp115 were preregistered, run and
recorded in this repository after the split, and their history is this repository's. The package's releases from 1.12.2
point their PyPI page's links and their docstrings here; 1.13.0 (22 October 2026) adds `from-embeddings` and `boundary`
and draws 400 labels for the certified zone, as exp112 found; 1.17.0 (24 October 2026) adds the map certificate and the
class design, which exp116 grades with the target rule of 1.16.0 on fully labelled real maps. The checks below still
run against 1.12.1, the release the record's experiments ran on; exp116's checks read only its outputs.

| Path | Contents |
|---|---|
| `exp/` | the numbered experiment scripts; `exp/out/` their committed outputs; `exp/jobs/` cluster jobs; `exp/NOTES.md` the lab log |
| `scripts/` | named entry points: the claim ledger, the suite-regression runner, the upstream-revision check, scoring an area with a public fine-tuned model, the demo sample builder |
| `docs/claims.yaml` | the claim ledger: each claim, the artifacts it rests on, its check and where it is cited |
| `docs/` | Summary, Findings, the technique ledger, results, method, preregistrations (`plan/`), test sets, related work |
| `report/` | the technical report (`main.tex`, `main.pdf`, `references.bib`) |
| `tests/` | every claim against its artifact, the experiments' code, and the package's outputs against the recorded numbers |

Sixteen test files share their name with a test file of the package. Thirteen hold only the tests of that file that
use the record; the others stayed with the package. Three are whole copies (`test_model_assisted.py`,
`test_per_class.py`, `test_trust_zone.py`), kept because claims in the ledger name tests in them as crosschecks, and
`test_estimate.py` also keeps one package test that the ledger names as a crosscheck.

Checks
------

The checks run against the released package from PyPI, olmoearth-inferencex 1.12.1, never against a working tree.

```bash
uv sync                                   # olmoearth-inferencex[geo]==1.12.1 and pytest
uv run pytest                             # the claims, the experiments' code and the recorded numbers
uv run python scripts/claims.py stale     # lists the superseded claims; any other line is a claim to rewrite
uv sync --extra encoder                   # the experiment environment (torch; Linux resolves the cu128 build)
```

`scripts/claims.py stale` exits 1 while it lists any claim, and two claims are superseded, so it exits 1 today. The
CI accepts that and fails if any line says a check fails.

Some per-unit exports are too large to commit (`exp/out/exp79_units/`; their checksums are in `exp/out/exp79_seeds/`).
The tests that need them skip when they are absent, as do the equality tests against the encoder stack when the
`encoder` extra is not installed.

A few scripts written before the split read the package's source tree: `scripts/make_demo_sample.py`,
`exp/exp78_shipped_tiles.py`, `exp/jobs/e98_read.sh` and `exp/jobs/e99_read.sh`. No test runs them. To rerun one, run
it from a checkout of the code repository at the matching release.

The cluster jobs (`exp/jobs/` and `scripts/score_area.sh`) were also written before the split. They run in the
cluster's checkout of the code repository, and several use its `geo` extra, which this project does not define. A job
given a full commit checks that commit out, so it still runs at any commit from before the split. The others run
whatever the checkout holds, and `scripts/score_area.sh` and several others first reset it to `origin/main`, which no
longer holds the experiments. To rerun one of those, check out v1.12.1 in place of the reset. No test runs them
either.

Licence and citation
--------------------

Apache License 2.0 (LICENSE). Cite the technical report with CITATION.cff.
