olmoearth_inferenceX-record
===========================

The research record behind the olmoearth-inferencex package: the experiments, their committed outputs, the claim
ledger that ties each stated result to an output, the pages that report the results, and the technical report.

The package, its documentation and its own tests are in https://github.com/2imi9/olmoearth_inferenceX
(documentation: https://olmoearth-inferencex.readthedocs.io; releases: https://pypi.org/project/olmoearth-inferencex/).

Source
------

Split on 14 October 2026 from 2imi9/olmoearth_inferenceX at commit 0d1161a09927cfd92091c7db945672a5bedead32
(release 1.12.1). The files are as they were at that commit, with these changes: the thirteen split test files below;
the links that now cross between the two repositories; the claim citations that named the package's pages, with the
two demo claims now cited in `docs/Findings.md`; the report's title block and availability section, with the PDF
rebuilt; and `.gitignore`. Their history is in that repository. Experiments archived before the split are in its
v1.9.0 tree.

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
`test_per_class.py`, `test_trust_zone.py`), kept because claims in the ledger name tests in them as crosschecks.

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

Licence and citation
--------------------

Apache License 2.0 (LICENSE). Cite the technical report with CITATION.cff.
