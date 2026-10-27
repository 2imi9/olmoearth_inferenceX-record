# ADR-002: Feature folders for the `oe_inferencex` package (1.19.0)

**Status:** Accepted (2026-10-27). The owner decided three things on 26 Oct 2026: the layout is by role (the feature
folders of §2); the old import paths are kept by **one alias finder** in `oe_inferencex/__init__.py`, which the owner
put in place of the 18 shim files this plan first proposed, before the migration started; and the release is 1.19.0.
Once 1.19.0 is released it supersedes **only the package-layout clause** of ADR-001 ("the package, flat"). ADR-001's
decisions on the record, `exp/jobs/` and artifacts stand. Its body is not edited: an accepted ADR stays as it was, and
only its status line gets a pointer to this one.
**Date:** 2026-10-26 (proposed); 2026-10-27 (accepted, as built)
**Deciders:** Ziming (2imi9)
**Applies to:** `github.com/2imi9/olmoearth_inferenceX` at `987dc7ac7bc1a5548b48ebe92f0b8c52bcadd1ed` (1.18.1, `main`),
moving to 1.19.0. Built on the branch `feature-folders`, ending at `443281c45f572030472ee8a3323bae00b4ce4110`.
**Filed at:** `olmoearth_inferenceX-record/docs/plan/adr-002-feature-folders.md`, filed with the 1.19.0 release,
before it goes out. The migration itself does not touch the record (§7).

## 0. How a migration agent uses this document

This section was the migration's operating instructions. The migration is done (§8); the section stays as it was
followed.

- **Nothing runs on the Mac.** That means no `python`, `pytest`, `uv run`, `uv build`, `mkdocs` or script of any
  kind. Locally you may only read and edit files and use `git`, `gh` and `ssh`. Every check in §6 runs on aicr as an
  sbatch cpu job, account `p2026_0089_neu`. Each step is submitted with one `ssh` and read back with one `ssh`, with
  no polling loops. Before every submission, state its cost in one line: CPUs, memory, wall time, scratch space,
  connections.
- **Work through the steps of §5.2 in order: one step, one commit.** A step is done when its cluster job (§6.1) is
  green. Do not start the next step before then.
- **Function bodies move verbatim.** The only permitted text changes inside moved code are the import lines listed
  in §3.9 and the `est` handle bindings listed in §4.5. Anything else, such as renaming, reformatting or "fixing"
  something noticed on the way, is out of scope (§7.2) and will show up in the verbatim check (§6.2).
- **A comment directly above a block moves with that block.** So does a section banner (`# ----- IO`), which goes
  with the block after it. Module docstrings stay with the module (or package `__init__`) that inherits the old path.
- Where this document and the code disagree, the code at `987dc7a` wins: stop and report the difference rather than
  guessing.

## 1. Context

### 1.1 Why now

ADR-001 (2026-09-17) kept the package flat. It set two revisit triggers: "the package passes about 25 modules" and
"gains a second heavy dependency family". Both have fired:

- **Module count.** The package now has 27 modules plus `__init__.py`, 18,568 lines in all and 84 public names. At
  ADR-001 it had 16 modules, about 2,800 lines and 55 public names.
- **Dependency families.** Beyond torch (`encoder`) and `geo`, the `embeddings` extra (zarr, obstore, pyproj,
  scikit-learn) and the `mcp` extra (the MCP SDK) now exist.

| File | Lines | File | Lines | File | Lines | File | Lines |
|---|---:|---|---:|---|---:|---|---:|
| `cli.py` | 4,137 | `plan.py` | 844 | `stratified.py` | 396 | `scale.py` | 187 |
| `estimate.py` | 2,974 | `assess.py` | 802 | `metrics.py` | 386 | `lcc.py` | 181 |
| `mcp_server.py` | 1,810 | `decide.py` | 700 | `sequential.py` | 350 | `__init__.py` | 110 |
| `certificate.py` | 1,289 | `agency.py` | 532 | `calibrate.py` | 324 | `awf.py` | 99 |
| `olmoearth.py` | 887 | `compare.py` | 475 | `explain.py` | 275 | `data.py` | 87 |
| `taskcard.py` | 404 | `demo.py` | 254 | `stats.py` | 246 | `evidence.py` | 85 |
| `signals.py` | 242 | `reliability.py` | 215 | `embeddings.py` | 201 | `figstyle.py` | 76 |

### 1.2 What the flat layout costs now

- **`cli.py` holds everything the commands share or own.** It has 120 top-level blocks: the module docstring, 16
  commands (`cmd_*`), 85 helpers, 17 constants, `build_parser`, `main` and the `__main__` guard. They serve nine
  features, so a change to one command edits the file every command lives in.
- **`estimate.py` holds the logic of four features.** Its 130 top-level names cover:
  - the sampling designs and error-rate intervals (the estimate feature);
  - the zone tests behind `certify` (32 names);
  - the labelled which-map test;
  - the per-class audit design;
  - exact hypergeometric arithmetic, which `plan` and `agency` also use.
- **Lazy import cycles have accumulated.** Every one of these exists only because of where code happens to sit:
  - `estimate` ↔ `sequential`, through `sequential_order`;
  - `cli` ↔ `certificate`, through the wording helpers and raster I/O;
  - `cli` ↔ `decide`, through the wording helpers;
  - `cli` ↔ `demo`;
  - `cli` ↔ `mcp_server`.

### 1.3 The owner's decisions (given, not reopened here)

1. **Feature folders, by role.** Each parallel feature gets its own subpackage, holding its logic and its own CLI
   command module. Code shared by two or more features goes into a common subpackage. This is not ADR-001's
   Option B, which split by dependency. Dependency groups stay documented and enforced: the guarded modules sit in
   folders whose `__init__` imports nothing (R1, R6 in §5.1).
2. **The public API stays identical.** The same 84 names in `oe_inferencex.__all__` resolve to the same objects.
3. **Old import paths keep working.** `oe_inferencex.estimate`, `.cli`, `.compare` and every other path a user,
   test or record script imports today still imports. Each becomes either a package that keeps the old names, or an
   old path that one alias finder in `oe_inferencex/__init__.py` answers with the moved module. (This plan first
   proposed a thin shim module at each old path; the owner replaced the shims with the finder before the migration.)
4. **The console entry point keeps working:** `oe-inferencex = "oe_inferencex.cli:main"`, at `pyproject.toml:34`.
5. **Behaviour is unchanged, byte for byte.** Every stdout and stderr text, exit code, written file and JSON key
   order stays the same. The one exception is the `written_by.version` stamp, which becomes 1.19.0. (As built
   there is one more: the task-card page names its source file `oe_inferencex/readers/taskcard.py`, §8.)
6. **Released as 1.19.0.**

### 1.4 Forces

- **Byte-identity has to be shown, not argued.** The battery in §6.3 shows it.
- **Tests patch module globals.** Fifteen patch sites depend on which module object a function reads its globals
  from (§4.6).
- **Some state must exist exactly once:**
  - the `RULES_OF` ContextVar;
  - two `lru_cache`s (`_hypergeom_interval`, `plan._ends`);
  - two memo dicts (`_SIGN_CRIT`, `plan._TABLES`);
  - the `certificate._RUN` RLock;
  - the `mcp_server._CAPTURE` lock.
- **A plain install has numpy only.** `import oe_inferencex.cli` must not pull in torch, rasterio, zarr or the MCP SDK
  (`tests/test_package.py:21`, and CI's wheel job).
- **The research record is pinned to `olmoearth-inferencex==1.12.1`**, so 1.19.0 does not reach it until someone
  bumps the pin (§7.1, R15).

## 2. Decision

### 2.1 Decisions

| Id | Decision | Why |
|---|---|---|
| D1 | One folder per feature. Each holds its logic and a `command.py`, whose commands each have a registrar `add_<cmd>_parser(sub)`. `estimate/` has two command modules, `sample_command.py` and `estimate_command.py`. | The owner's decision. Registrars keep `build_parser`'s 16 blocks verbatim and in order. |
| D2 | The grade feature's folder is **`agency/`, not `grade/`**. | `oe_inferencex.grade` is a public function in `__all__`. Importing a subpackage `oe_inferencex.grade` would rebind that attribute to the module. |
| D3 | `mcp_server.py` stays **one module at the top level**. `demo.py` and `sample/` stay too. | MCP is an interface, not a feature: its CARDS, INSTRUCTIONS and HARD_RULES cross-reference every tool, one `_CAPTURE` lock guards every tool, and FastMCP resolves string annotations in module globals. The tests re-import exactly `oe_inferencex.mcp_server` (`test_mcp_server.py:296-308`) and read the file `oe_inferencex/mcp_server.py` (`test_agent_borrowings.py:37`). `demo.SAMPLE` is a path relative to `demo.py`; a wrong one silently falls back to the made-up map. `sample/` is package data. |
| D4 | `signals` goes to `common/`, not `review/`. *(Refines the sketch.)* | `stats` imports `signals.midrank_pct` at module top. With signals in `review/`, `common` would depend on a feature. `signals` imports nothing from the package. |
| D5 | `evidence`, `awf`, `data`, `figstyle` and `reliability` go to a new `research/` folder, not `readers/`. *(Refines the sketch.)* | No command imports any of them. They are the record's toolkit, and four need torch or matplotlib. `readers/` keeps the input routes that a command or user reads through. |
| D6 | The labelled which-map test (`compare_from_disagreement`, `sample_disagreement`, `sign_critical`, `which_map_rule` and their notes) goes to `estimate/which_map.py`. | It is reached through `sample --other` then `estimate`. `compare/` stays the label-free feature. |
| D7 | `_map_windows` and `_two_map_windows` go to **`review/windows.py`**, not `common/`. *(Refines the inventory.)* | They call `assess_classmap`, `assess_prediction` and `_pooled_argmax`. In `common/` they would make common depend on review. The commands that use them (estimate, certify, compare, sample, plan) depend on review anyway. |
| D8 | Old paths are kept in three ways (§4.1): the 18 moved modules through **one alias finder** in `oe_inferencex/__init__.py`, with no file left at their old paths; 5 **packages holding the old code in `__init__`** (`compare`, `scale`, `plan`, `decide`, `agency`); 2 **re-export packages** (`estimate`, `cli`). | An alias keeps module identity, so monkeypatches and private names keep working. The owner chose one finder over the 18 shim files this plan first proposed. Where a folder takes the old module's name, a file at the old path would be impossible anyway. |
| D9 | Moves are verbatim. The only edits are import lines (§3.9) and one `est` binding per module that reads `est.X` (§4.5). | Byte-identity, and a reviewable diff (`git diff --color-moved`). |
| D10 | Each piece of shared state is defined once and re-exported by reference (R8). | Two `RULES_OF` objects would break verify of 1.17.x certificates. A second lru_cache would make `cache_clear` miss. |
| D11 | No deletions, renames or merges. That includes the dead `_band_count` (`cli.py:705`), the unused `boundary_indicator` import (`cli.py:60`), and same-named duplicates in different modules: `_check_patch` in assess and cli, `PROB_TOLERANCE` in cli and olmoearth, `Z95` in estimate and stratified, `_code` in estimate and agency, `_pc` in decide and mcp_server. | Each would change what an old path accepts or how a module behaves. They are follow-ups (§7.3). |
| D12 | `common/` holds code that two or more features share. At module top it imports nothing from a feature; the few lazy exceptions are listed in §5.1. | Keeps the import graph layered (§5.1). |
| D13 | The version is 1.19.0, set in a separate last commit after the battery passes (§5.2 S14). | The battery then runs on code whose version string is still 1.18.1, so the comparison needs no normalisation. |

### 2.2 Target tree (every folder and file)

```
oe_inferencex/
├── __init__.py              public API: __all__ (84 names) unchanged; imports the new paths and binds the 1.18.1 module attributes from them; the alias finder for the 18 old paths (§4.2); docstring lists the folders; fallback version 1.19.0
├── demo.py                  unchanged module (`oe-inferencex demo`); 3 import lines repointed (§3.9)
├── mcp_server.py            unchanged module, the MCP server; 2 import lines repointed (§3.9)
├── sample/                  unchanged package data
│   ├── NOTICE.txt           the sample map's licence notice
│   └── dynamic_world_tile.npz   the real map `demo` audits
├── common/                  code two or more features share; imports no feature at module top (R3)
│   ├── __init__.py          docstring only
│   ├── metrics.py           = metrics.py: AURC, excess AURC, capture, calibration error, design-weighted forms
│   ├── signals.py           = signals.py: confidence readings, no-model pixel controls, midrank_pct
│   ├── stats.py             = stats.py: exact sign tests, cluster tests, bootstraps, spearman
│   ├── hypergeom.py         from estimate.py: exact hypergeometric arithmetic (14 names, incl. the lru-cached _hypergeom_interval)
│   ├── sampling.py          from estimate.py: _condition_index, _condition_guards, the review-set check (6 names) + sequential_order from sequential.py
│   ├── readings.py          from estimate.py: the --also-accept second reading (_code, _also_accept_pairs, also_accept_reading)
│   ├── raster.py            from cli.py: read_raster and its mask helpers, write_raster, affine_apply, window_coords, pool_valid, _same_grid (11)
│   ├── output.py            from cli.py: the written_by stamp, output-path and argument checks, _where, _pct (12)
│   ├── maps.py              from cli.py: PROB_TOLERANCE, _check_scores, decisions (3)
│   ├── product.py           from cli.py: the class-map-plus-confidence input and its two parser options (6)
│   ├── condition.py         from cli.py: the --condition layer and its names (3)
│   ├── labels.py            from cli.py: a labelled sample read back with its sidecar, --also-accept rows (12)
│   └── wording.py           from cli.py: the class design's wording, class_words and friends (7)
├── review/                  where to look first: assess, the cues, the fused ranker; commands assess, boundary
│   ├── __init__.py          docstring only
│   ├── assess.py            = assess.py
│   ├── explain.py           = explain.py
│   ├── calibrate.py         = calibrate.py
│   ├── windows.py           from cli.py: _map_windows, _two_map_windows (maps read into per-window readings for labelled routes)
│   └── command.py           from cli.py: cmd_assess, cmd_boundary, _budget_tag; add_assess_parser, add_boundary_parser
├── compare/                 two inferences of one scene, no labels; command compare
│   ├── __init__.py          = compare.py, verbatim but for 2 import lines
│   └── command.py           from cli.py: cmd_compare, _window_majority; add_compare_parser
├── scale/                   one map at several window sizes; command scale
│   ├── __init__.py          = scale.py, verbatim but for 1 import statement
│   └── command.py           from cli.py: cmd_scale, _window_size_text; add_scale_parser
├── estimate/                how wrong the map is, from labels; commands sample, estimate
│   ├── __init__.py          estimate.py's docstring, `import math`, `import numpy as np`, and the re-export of all 130 old names; no logic
│   ├── designs.py           sampling designs, allocations, sample_for_estimation, their notes (14)
│   ├── intervals.py         Wilson, t, stratified and cluster intervals, design effect, _union_interval (11)
│   ├── error_rate.py        estimate_error_rate, _estimate_core, estimate_from_indices and their notes (15)
│   ├── model_assisted.py    the model-assisted intervals (4)
│   ├── per_class.py         per-class accuracy, the HT ratios, the class design's exact intervals (14)
│   ├── which_map.py         the labelled which-map test (8)
│   ├── audit.py             the per-class audit design (9)
│   ├── command_text.py      from cli.py: _zone_words, shared by the sample and estimate audit routes
│   ├── sample_command.py    from cli.py: cmd_sample and its routes (16 names); add_sample_parser
│   └── estimate_command.py  from cli.py: cmd_estimate and its routes (8 names); add_estimate_parser
├── certify/                 which part of the map can be trusted; commands certify, certificate, verify
│   ├── __init__.py          docstring only
│   ├── zone.py              from estimate.py: zone tests, certify_zone, certify_by_condition, RULES_OF, ZONE_*, certify notes (32)
│   ├── sequential.py        = sequential.py, but sequential_order now lives in common/sampling and is re-bound here
│   ├── certificate.py       = certificate.py
│   └── command.py           from cli.py: cmd_certify, cmd_certificate, cmd_verify and 9 helpers/constants; 3 registrars
├── plan/                    how many labels each route needs; command plan
│   ├── __init__.py          = plan.py, verbatim but for 1 import line
│   └── command.py           from cli.py: cmd_plan and 6 helpers; add_plan_parser
├── decide/                  typed answers read from results; command decide
│   ├── __init__.py          = decide.py, verbatim but for 1 lazy import line
│   └── command.py           from cli.py: cmd_decide, _worded_print; add_decide_parser
├── agency/                  an agency's stratified sample graded; command grade (NOT grade/, D2)
│   ├── __init__.py          = agency.py, verbatim
│   ├── stratified.py        = stratified.py, verbatim but for 1 lazy import line
│   └── command.py           from cli.py: cmd_grade, _range_pct, _iv_pct; add_grade_parser
├── readers/                 input routes; commands from-olmoearth, from-embeddings
│   ├── __init__.py          docstring only; MUST NOT import its modules (extras, R6)
│   ├── olmoearth.py         = olmoearth.py: an OlmoEarth run's rslearn dataset -> scores and a condition layer
│   ├── embeddings.py        = embeddings.py: a geo-embeddings Zarr store + labelled points -> class probabilities
│   ├── lcc.py               = lcc.py: range reader for the served LCC rasters
│   ├── taskcard.py          = taskcard.py: what each OlmoEarth fine-tuned model is (keeps its __main__ block)
│   └── command.py           from cli.py: cmd_from_olmoearth, cmd_from_embeddings, _skipped_text; 2 registrars
├── research/                the research record's toolkit; no command imports it
│   ├── __init__.py          docstring only; MUST NOT import its modules (torch, matplotlib, R6)
│   ├── evidence.py          = evidence.py (torch)
│   ├── awf.py               = awf.py (torch, rasterio, olmoearth_pretrain)
│   ├── data.py              = data.py (STAC, rasterio, torch); never a folder named data/ (.gitignore)
│   ├── figstyle.py          = figstyle.py (matplotlib)
│   └── reliability.py       = reliability.py (numpy only)
└── cli/                     the command line: parser assembly and main
    ├── __init__.py          cli.py's docstring (the parser reads it), its import block, the re-export of every cli.py name, build_parser, main
    ├── __main__.py          `python -m oe_inferencex.cli`
    ├── demo_command.py      from cli.py: cmd_demo; add_demo_parser
    └── mcp_command.py       from cli.py: cmd_mcp; add_mcp_parser
```

The tree has 12 package folders plus `sample/`, and 66 `.py` files: 3 top-level modules (`__init__`, `demo`,
`mcp_server`) and 63 files in folders. No file is left at any of the 18 old paths of the moved modules (assess,
explain, calibrate, metrics, signals, stats, reliability, evidence, awf, data, figstyle, olmoearth, embeddings, lcc,
taskcard, stratified, sequential, certificate). The alias finder in `__init__.py` answers them (§4.2), taskcard's
under `python -m` too.
No folder name collides with the `.gitignore` patterns `data/`, `build/`, `dist/` and `site/`, with
`oe_inferencex/sample/`, with a public name or with the MCP SDK's top-level `mcp`.

## 3. Move map

### 3.0 Conventions

- "**=** X.py" means `git mv` of the whole file. Its body stays verbatim except for the import lines in §3.9.
- "From X.py" means the block is cut from X.py and pasted verbatim into the new module, with the comment lines
  directly above it.
- Each new module begins with a one-line docstring, then the imports its blocks use, then the blocks in their
  original order. The docstring is new text, and allowed.
- The imports a module needs are listed in §3.3.1 (estimate split) and §3.1.2 (cli split). The gate is the same in
  both cases: `pyflakes` reports no undefined name beyond the 1.18.1 baseline's (§6.1, check 1).

### 3.1 `cli.py` (4,137 lines): every top-level block

#### 3.1.1 Blocks in source order

The 119 blocks plus the `__main__` guard (lines 4136-4137). The module docstring (lines 1-44) goes verbatim to
`cli/__init__.py`: `build_parser`'s description is `__doc__.split("\n\n")[0]`, lines 1-24.

| Name | Lines | New module | Notes |
|---|---|---|---|
| `read_raster` | 64-113 | `oe_inferencex/common/raster.py` | rasterio lazy. Also read by certify/certificate (lazy, 3 sites) and, through cli, tests/test_bughunt_1_17_2.py:17 and tests/golden/condition_1_3_1/generate.py:212 |
| `_mask_bands` | 116-129 | `oe_inferencex/common/raster.py` | certify/certificate `_declared` (lazy, 489) |
| `_read_masked` | 132-138 | `oe_inferencex/common/raster.py` |  |
| `_gdal_masked` | 141-152 | `oe_inferencex/common/raster.py` |  |
| `_mask_hint` | 155-159 | `oe_inferencex/common/raster.py` | used by sample and by review/windows `_map_windows` |
| `affine_apply` | 162-168 | `oe_inferencex/common/raster.py` | certify/certificate `_verify`, `point` (lazy); research/data `fetch_s2_window` (lazy) |
| `write_raster` | 171-183 | `oe_inferencex/common/raster.py` | certify/certificate `_write_bundle` (lazy) |
| `window_coords` | 186-192 | `oe_inferencex/common/raster.py` |  |
| `pool_valid` | 195-197 | `oe_inferencex/common/raster.py` | not `assess._pool_valid`; both stay |
| `_stamp` | 200-208 | `oe_inferencex/common/output.py` | keeps `from oe_inferencex import __version__` INSIDE the function (tests patch the package's `__version__`) |
| `_pct` | 212-215 | `oe_inferencex/common/output.py` | tests/test_cli.py:244 through cli |
| `PROB_TOLERANCE` | 218-218 | `oe_inferencex/common/maps.py` | `olmoearth.PROB_TOLERANCE` is a separate copy; do not merge |
| `_check_scores` | 221-254 | `oe_inferencex/common/maps.py` | mcp_server matches its refusal text; text unchanged |
| `_condition_names` | 258-281 | `oe_inferencex/common/condition.py` | tests/test_bughunt_1_17_2.py:452 through cli |
| `_read_condition` | 284-297 | `oe_inferencex/common/condition.py` |  |
| `_condition_args` | 300-307 | `oe_inferencex/common/condition.py` | reads `args` with getattr (hand-built Namespaces in demo and tests) |
| `_unrecorded_note` | 310-319 | `oe_inferencex/estimate/sample_command.py` |  |
| `_certify_need_note` | 322-333 | `oe_inferencex/estimate/sample_command.py` | default `delta=est.ZONE_DELTA` is bound at def time: the `est` import line sits above it |
| `_check_patch` | 336-340 | `oe_inferencex/common/output.py` | not `assess._check_patch`, a different function of the same name; command modules import this one |
| `_check_out_dir` | 343-347 | `oe_inferencex/common/output.py` |  |
| `_check_out_file` | 350-358 | `oe_inferencex/common/output.py` |  |
| `_num_arg` | 361-365 | `oe_inferencex/common/output.py` | tests/test_bughunt_1_17_2.py:616 through cli |
| `_nodata_arg` | 368-373 | `oe_inferencex/common/output.py` | tests/test_bughunt_1_17_2.py:616 through cli |
| `_same_file` | 376-383 | `oe_inferencex/common/output.py` |  |
| `_refuse_overwriting_inputs` | 386-393 | `oe_inferencex/common/output.py` |  |
| `_sample_map_inputs` | 396-408 | `oe_inferencex/common/labels.py` |  |
| `_budget_tag` | 411-414 | `oe_inferencex/review/command.py` |  |
| `cmd_assess` | 417-578 | `oe_inferencex/review/command.py` | demo.run calls `cli.cmd_assess`; tests/test_audit_fixes.py:23, tests/test_cli.py:1175 call it with hand-built Namespaces |
| `cmd_from_embeddings` | 582-604 | `oe_inferencex/readers/command.py` | edit lazy import 585 (§3.9) |
| `cmd_boundary` | 608-702 | `oe_inferencex/review/command.py` | edit lazy import 611 (§3.9); prints a `sample` command line |
| `_band_count` | 705-712 | `oe_inferencex/common/raster.py` | dead since f43daee (no caller); moved as is, not deleted |
| `_window_size_text` | 716-730 | `oe_inferencex/scale/command.py` | tests/test_scale.py:303 through cli |
| `cmd_scale` | 733-827 | `oe_inferencex/scale/command.py` | edit lazy import 737 (§3.9) |
| `_range_pct` | 831-837 | `oe_inferencex/agency/command.py` |  |
| `_iv_pct` | 840-843 | `oe_inferencex/agency/command.py` |  |
| `cmd_grade` | 846-915 | `oe_inferencex/agency/command.py` | its local variable `est` (900) shadows nothing in agency/command (no `est` import there); keep it that way |
| `cmd_certificate` | 919-936 | `oe_inferencex/certify/command.py` | edit lazy import 923 (§3.9) |
| `cmd_verify` | 939-1003 | `oe_inferencex/certify/command.py` | edit lazy import 943 (§3.9); returns 1 when not verified |
| `decisions` | 1007-1055 | `oe_inferencex/common/maps.py` |  |
| `_same_grid` | 1058-1069 | `oe_inferencex/common/raster.py` |  |
| `PRODUCT_SIGNAL` | 1072-1072 | `oe_inferencex/common/product.py` |  |
| `MAX_CLASS_ID` | 1073-1073 | `oe_inferencex/common/product.py` |  |
| `NO_RANGE_NOTE` | 1074-1076 | `oe_inferencex/common/product.py` |  |
| `_read_product` | 1079-1127 | `oe_inferencex/common/product.py` |  |
| `_population_note` | 1130-1138 | `oe_inferencex/common/labels.py` |  |
| `_json_nodata` | 1141-1148 | `oe_inferencex/common/output.py` |  |
| `_product_args` | 1151-1159 | `oe_inferencex/common/product.py` |  |
| `_two_map_windows` | 1162-1254 | `oe_inferencex/review/windows.py` | refusals say "compare" whichever command calls it (unchanged) |
| `_window_majority` | 1257-1274 | `oe_inferencex/compare/command.py` |  |
| `cmd_compare` | 1277-1371 | `oe_inferencex/compare/command.py` | lazy imports 1311, 1338 are absolute `oe_inferencex.compare` and stay |
| `cmd_demo` | 1374-1376 | `oe_inferencex/cli/demo_command.py` | lazy `from oe_inferencex.demo import run` stays lazy |
| `_top1` | 1381-1390 | `oe_inferencex/estimate/sample_command.py` |  |
| `cmd_sample` | 1393-1707 | `oe_inferencex/estimate/sample_command.py` | edit lazy imports 1547, 1562, 1656 (§3.9); mutates args.seed; tests/test_cli.py:1183 calls it through cli |
| `_certify_plan` | 1710-1742 | `oe_inferencex/estimate/sample_command.py` | writes the plan certify/command `_keep_to_plan` enforces |
| `_sample_map` | 1745-1780 | `oe_inferencex/estimate/sample_command.py` |  |
| `_sidecar_map_fields` | 1783-1802 | `oe_inferencex/estimate/sample_command.py` |  |
| `_extend_command` | 1805-1824 | `oe_inferencex/estimate/sample_command.py` | prints a `sample` command line |
| `_kept_rows_mismatch` | 1827-1841 | `oe_inferencex/estimate/sample_command.py` |  |
| `_reading_text` | 1844-1856 | `oe_inferencex/estimate/sample_command.py` | prints part of a `sample` command line |
| `_sequential_to_extend` | 1859-1905 | `oe_inferencex/estimate/sample_command.py` |  |
| `_CLASS_WORDS` | 1911-1914 | `oe_inferencex/common/wording.py` | with its comment 1906-1910; certify/certificate `_class_words_1171` reads it (lazy) |
| `_PATH_KEYS` | 1915-1915 | `oe_inferencex/common/wording.py` | certify/certificate `_class_words_1171` reads it (lazy) |
| `class_words` | 1918-1970 | `oe_inferencex/common/wording.py` | edit lazy imports 1934, 1936 (§3.9); reads the one RULES_OF ContextVar; mcp_server and tests reach it as `cli.class_words` |
| `_RULE_WORDS` | 1975-1975 | `oe_inferencex/common/wording.py` |  |
| `class_names` | 1978-1987 | `oe_inferencex/common/wording.py` | mcp_server as `cli.class_names` |
| `class_wording` | 1990-1995 | `oe_inferencex/common/wording.py` | decide (lazy, 693) and mcp_server; decide's import is repointed to common/wording |
| `result_class_names` | 1998-2019 | `oe_inferencex/common/wording.py` | decide (lazy, 693) and mcp_server |
| `CLASS_RULE_TEXT` | 2022-2023 | `oe_inferencex/estimate/sample_command.py` |  |
| `OTHER_DATES_NOTE` | 2026-2029 | `oe_inferencex/estimate/sample_command.py` | mcp_server:1005; tests/test_agent_borrowings.py:167 through cli |
| `_sample_disagreement` | 2032-2084 | `oe_inferencex/estimate/sample_command.py` |  |
| `_zone_words` | 2087-2099 | `oe_inferencex/estimate/command_text.py` | shared by the sample and estimate audit routes, both in estimate/ |
| `_sample_class_audit` | 2102-2175 | `oe_inferencex/estimate/sample_command.py` |  |
| `_sidecar_design` | 2178-2185 | `oe_inferencex/common/labels.py` | mcp_server:1130 as `cli._sidecar_design` |
| `_estimate_disagreement` | 2188-2250 | `oe_inferencex/estimate/estimate_command.py` | lazy `oe_inferencex.decide` import (2231) stays |
| `_reference_column` | 2253-2282 | `oe_inferencex/common/labels.py` |  |
| `_also_accept` | 2285-2300 | `oe_inferencex/common/labels.py` |  |
| `_also_accept_rows` | 2303-2335 | `oe_inferencex/common/labels.py` |  |
| `_also_words` | 2338-2345 | `oe_inferencex/common/labels.py` |  |
| `_unused_note` | 2348-2350 | `oe_inferencex/common/labels.py` |  |
| `_verdict_words` | 2353-2355 | `oe_inferencex/estimate/estimate_command.py` |  |
| `_estimate_class_audit` | 2358-2446 | `oe_inferencex/estimate/estimate_command.py` |  |
| `FILL_INSTRUCTION` | 2451-2452 | `oe_inferencex/estimate/sample_command.py` | tests/golden/condition_1_3_1/changes.py:83 names ("oe_inferencex.cli", "FILL_INSTRUCTION") |
| `_check_sidecar` | 2455-2461 | `oe_inferencex/common/labels.py` |  |
| `_csv_rows` | 2464-2473 | `oe_inferencex/common/labels.py` | edit lazy import 2469 (§3.9) |
| `_labelled_sample` | 2476-2546 | `oe_inferencex/common/labels.py` | tests/test_cli.py:1082, tests/test_also_accept.py:353 through cli |
| `_rounding_tolerance` | 2549-2555 | `oe_inferencex/common/labels.py` | certify/certificate `_rounding_tolerance` wraps it (lazy); tests/test_cli.py:592 |
| `_where` | 2558-2564 | `oe_inferencex/common/output.py` | certify/certificate `cli_where` wraps it (lazy); tests/test_bughunt_1_17_2.py:495 |
| `_readable_path` | 2567-2570 | `oe_inferencex/common/output.py` |  |
| `_map_windows` | 2573-2669 | `oe_inferencex/review/windows.py` | PATCHED by tests/test_also_accept.py:685: retarget (§4.7, S10i); also called by cmd_certify |
| `_reference_classes` | 2672-2687 | `oe_inferencex/estimate/estimate_command.py` |  |
| `_rate` | 2690-2695 | `oe_inferencex/estimate/estimate_command.py` |  |
| `_outside_note` | 2698-2719 | `oe_inferencex/estimate/estimate_command.py` | mcp_server:1184,1229; tests/test_cli.py:1057 |
| `cmd_estimate` | 2722-2943 | `oe_inferencex/estimate/estimate_command.py` |  |
| `_keep_to_plan` | 2946-2977 | `oe_inferencex/certify/command.py` | mutates args.alpha, delta, rule, level_cut |
| `cmd_certify` | 2980-3088 | `oe_inferencex/certify/command.py` | certificate.verify re-runs it in process through `cli.main` |
| `ZONE_ALSO_KEYS` | 3095-3096 | `oe_inferencex/certify/command.py` | tests/test_also_accept.py:446 through cli |
| `_also_mask` | 3099-3101 | `oe_inferencex/certify/command.py` | the string "_also_accept.npy" is a literal, not a call of `_also_accept` |
| `_zone_reading_note` | 3104-3106 | `oe_inferencex/certify/command.py` | tests/test_also_accept.py:81-85,452 through cli |
| `_zone_also_accept` | 3109-3129 | `oe_inferencex/certify/command.py` |  |
| `_zone_also_text` | 3132-3161 | `oe_inferencex/certify/command.py` |  |
| `_certify_sequential` | 3164-3216 | `oe_inferencex/certify/command.py` | edit lazy import 3168 (§3.9) |
| `_certify_by_condition` | 3219-3302 | `oe_inferencex/certify/command.py` |  |
| `_worded_print` | 3305-3307 | `oe_inferencex/decide/command.py` |  |
| `cmd_decide` | 3310-3364 | `oe_inferencex/decide/command.py` | lazy `from oe_inferencex.decide import decide` stays |
| `_plan_windows` | 3367-3385 | `oe_inferencex/plan/command.py` |  |
| `_points` | 3388-3389 | `oe_inferencex/plan/command.py` |  |
| `_labels` | 3392-3393 | `oe_inferencex/plan/command.py` |  |
| `_ladder_text` | 3396-3404 | `oe_inferencex/plan/command.py` |  |
| `_sample_command` | 3407-3428 | `oe_inferencex/plan/command.py` | prints a `sample` command line; tests/test_bughunt_1_17_2.py:750,925 through cli |
| `_reach` | 3431-3445 | `oe_inferencex/plan/command.py` |  |
| `cmd_plan` | 3448-3666 | `oe_inferencex/plan/command.py` | lazy `oe_inferencex.plan` imports stay; mutates args.patch |
| `_skipped_text` | 3669-3676 | `oe_inferencex/readers/command.py` |  |
| `cmd_from_olmoearth` | 3679-3728 | `oe_inferencex/readers/command.py` | edit lazy import 3683 (§3.9) |
| `cmd_mcp` | 3731-3740 | `oe_inferencex/cli/mcp_command.py` | `from oe_inferencex import mcp_server` MUST stay lazy (mcp_server imports cli at module top) |
| `ALSO_ACCEPT_HELP` | 3743-3744 | `oe_inferencex/estimate/estimate_command.py` | tests/test_also_accept.py:827-831 through cli |
| `CERTIFY_ALSO_ACCEPT_HELP` | 3746-3747 | `oe_inferencex/certify/command.py` | tests/test_also_accept.py:827-831 through cli |
| `_product_parser_args` | 3750-3759 | `oe_inferencex/common/product.py` | called by the assess, sample and plan registrars; scale has its own copies of the two options |
| `build_parser` | 3762-4128 | `oe_inferencex/cli/__init__.py` | stays in cli/__init__.py next to the docstring it reads; its 16 blocks become registrars (§3.2) |
| `main` | 4131-4133 | `oe_inferencex/cli/__init__.py` | the console entry point `oe_inferencex.cli:main`; mcp_server calls it as `cli.main` |
| `if __name__ == "__main__"` | 4136-4137 | `oe_inferencex/cli/__main__.py` | becomes cli/__main__.py |

Counts by destination:

| Destination | Names |
|---|---:|
| `common/raster.py` | 11 |
| `common/output.py` | 12 |
| `common/maps.py` | 3 |
| `common/product.py` | 6 |
| `common/condition.py` | 3 |
| `common/labels.py` | 12 |
| `common/wording.py` | 7 |
| `review/windows.py` | 2 |
| `review/command.py` | 3 |
| `readers/command.py` | 3 |
| `scale/command.py` | 2 |
| `agency/command.py` | 3 |
| `certify/command.py` | 12 |
| `compare/command.py` | 2 |
| `estimate/sample_command.py` | 16 |
| `estimate/command_text.py` | 1 |
| `estimate/estimate_command.py` | 8 |
| `decide/command.py` | 2 |
| `plan/command.py` | 7 |
| `cli/demo_command.py` | 1 |
| `cli/mcp_command.py` | 1 |
| `cli/__init__.py` | 2 (`build_parser`, `main`) |
| `cli/__main__.py` | the guard |
| **Total** | **119 + guard** |

#### 3.1.2 `cli.py`'s import block (lines 45-60) and what each new module imports

`cli/__init__.py` keeps the block, with the package imports repointed:

```python
import argparse
import csv
from types import SimpleNamespace
import json
import math
import os
import sys

import numpy as np

from oe_inferencex.review.assess import (RULE_TEXT, _boundary_valid, _pool, _pool_valid, _pooled_argmax, assess_classmap,
                                         assess_prediction, pool_condition, summary)
from oe_inferencex import estimate as est
from oe_inferencex.compare import compare_inferences
from oe_inferencex.review.explain import explain_review_set
from oe_inferencex.common.signals import boundary_indicator
```

It then imports every name of the table above from its new module, one `from ... import (...)` per module,
including the private names and the registrars. This keeps every name of `dir(oe_inferencex.cli)` at 1.18.1, among
them the imported `np`, `est`, `summary` and the unused `boundary_indicator`. No `import *`.

Cross-module imports each new module needs. Computed from the bodies at 987dc7a; docstring mentions excluded:

| Module | Package imports at module top (lazy imports stay inside their functions, §3.9) |
|---|---|
| `common/raster.py` | none (numpy, os; rasterio lazy) |
| `common/output.py` | none (numpy, os, json as used); `_stamp` keeps its lazy `from oe_inferencex import __version__` |
| `common/maps.py` | `common.raster`: read_raster |
| `common/product.py` | `common.raster`: _same_grid, read_raster |
| `common/condition.py` | `common.raster`: _same_grid, read_raster |
| `common/labels.py` | `from oe_inferencex.common import readings as est` (§4.5); lazy `csv_rows` (§3.9) |
| `common/wording.py` | none at top; lazy `RULES_OF`, `_class_words_1171` (§3.9) |
| `review/windows.py` | `review.assess`: _pooled_argmax, assess_classmap, assess_prediction; `common.raster`: _mask_hint, _same_grid, pool_valid, read_raster; `common.maps`: _check_scores, decisions; `common.product`: PRODUCT_SIGNAL, _read_product; `common.labels`: _rounding_tolerance; `common.output`: _readable_path; `types.SimpleNamespace`, numpy, os |
| `review/command.py` | `review.assess`: assess_classmap, assess_prediction, pool_condition, summary; `review.explain`: explain_review_set; `common.condition`: _condition_args, _read_condition; `common.maps`: PROB_TOLERANCE, _check_scores, decisions; `common.output`: _check_out_dir, _check_out_file, _check_patch, _nodata_arg, _pct, _refuse_overwriting_inputs, _stamp, _where; `common.product`: PRODUCT_SIGNAL, _product_args, _product_parser_args, _read_product; `common.raster`: _same_grid, read_raster, window_coords, write_raster |
| `readers/command.py` | `common.output`: _check_out_dir |
| `scale/command.py` | `review.assess`: summary; `common.maps`: _check_scores; `common.output`: _check_out_file, _pct, _refuse_overwriting_inputs, _stamp, _where; `common.product`: _product_args, _read_product; `common.raster`: _same_grid, read_raster |
| `agency/command.py` | `common.output`: _check_out_file, _refuse_overwriting_inputs, _stamp, _where (no `est`, see `cmd_grade`'s note) |
| `certify/command.py` | `from oe_inferencex.certify import zone as est` (§4.5); `common.labels`: _also_accept_rows, _also_words, _labelled_sample, _population_note, _sample_map_inputs, _sidecar_design, _unused_note; `common.output`: _check_out_file, _refuse_overwriting_inputs, _same_file, _stamp; `common.wording`: class_wording; `review.windows`: _map_windows |
| `compare/command.py` | `oe_inferencex.compare`: compare_inferences; `review.assess`: _boundary_valid, summary; `review.windows`: _two_map_windows; `common.output`: _check_out_dir, _pct, _refuse_overwriting_inputs, _stamp, _where; `common.raster`: _same_grid, pool_valid, read_raster, window_coords, write_raster |
| `estimate/command_text.py` | `from oe_inferencex import estimate as est` |
| `estimate/sample_command.py` | `from oe_inferencex import estimate as est`; `estimate.command_text`: _zone_words; `review.assess`: RULE_TEXT, _pool, _pool_valid, assess_classmap, assess_prediction, pool_condition; `review.windows`: _two_map_windows; `common.condition`: _condition_args, _condition_names, _read_condition; `common.labels`: _check_sidecar, _csv_rows, _rounding_tolerance; `common.maps`: _check_scores; `common.output`: _check_patch, _json_nodata, _nodata_arg, _num_arg, _refuse_overwriting_inputs, _same_file, _stamp, _where; `common.product`: PRODUCT_SIGNAL, _product_args, _product_parser_args, _read_product; `common.raster`: _mask_hint, read_raster, window_coords; `common.wording`: class_wording |
| `estimate/estimate_command.py` | `from oe_inferencex import estimate as est`; `estimate.command_text`: _zone_words; `review.windows`: _map_windows; `common.labels`: _also_accept, _also_accept_rows, _also_words, _check_sidecar, _csv_rows, _labelled_sample, _population_note, _reference_column, _sample_map_inputs, _sidecar_design, _unused_note; `common.output`: _check_out_file, _refuse_overwriting_inputs, _stamp; `common.product`: MAX_CLASS_ID; `common.wording`: class_wording |
| `decide/command.py` | `common.output`: _check_out_file, _refuse_overwriting_inputs, _stamp; `common.wording`: class_wording, result_class_names |
| `plan/command.py` | `review.assess`: assess_classmap, assess_prediction; `review.windows`: _two_map_windows; `common.maps`: _check_scores; `common.output`: _check_out_file, _check_patch, _nodata_arg, _num_arg, _refuse_overwriting_inputs, _stamp; `common.product`: PRODUCT_SIGNAL, _product_args, _product_parser_args, _read_product; `common.raster`: read_raster |
| `cli/demo_command.py` | none (demo is imported lazily inside `cmd_demo`) |
| `cli/mcp_command.py` | none (mcp_server is imported lazily inside `cmd_mcp`; it MUST stay lazy) |

Standard-library and numpy imports (`csv`, `json`, `math`, `os`, `numpy as np`) go into each module that uses them.
`math` is used only by `cmd_estimate`, so it goes to `estimate/estimate_command.py`. `csv` goes to the review,
compare and sample command modules.

**Two different functions are both named `_check_patch`**, at `assess.py:121` and `cli.py:336`. The command modules
import **cli's**, from `common/output.py`, and never `review.assess._check_patch`.

### 3.2 `build_parser` (cli.py:3762-4128): blocks become registrars

`cli/__init__.py` keeps `build_parser`'s first two lines and its last line verbatim (3763, 3764 and 4128). In between
it calls the registrars in this fixed order: the order is visible in `oe-inferencex --help` and must not change.
Each registrar is

```python
def add_<cmd>_parser(sub):
    <the block's lines, verbatim, at the same 4-space indentation>
```

and returns nothing.

| # | Command | Lines | Registrar | In module | Reads |
|---:|---|---|---|---|---|
| 1 | assess | 3765-3785 | `add_assess_parser` | `review/command.py` | `_product_parser_args`, `cmd_assess` |
| 2 | demo | 3786-3790 | `add_demo_parser` | `cli/demo_command.py` | `cmd_demo` |
| 3 | from-embeddings | 3791-3804 | `add_from_embeddings_parser` | `readers/command.py` | `cmd_from_embeddings` |
| 4 | boundary | 3805-3817 | `add_boundary_parser` | `review/command.py` | `cmd_boundary` |
| 5 | scale | 3818-3835 | `add_scale_parser` | `scale/command.py` | `cmd_scale`; its own `--confidence`, `--confidence-range` |
| 6 | grade | 3836-3873 | `add_grade_parser` | `agency/command.py` | `cmd_grade`; the required group `--weight`/`--strata` |
| 7 | certificate | 3874-3882 | `add_certificate_parser` | `certify/command.py` | `cmd_certificate` |
| 8 | verify | 3883-3895 | `add_verify_parser` | `certify/command.py` | `cmd_verify` |
| 9 | compare | 3896-3911 | `add_compare_parser` | `compare/command.py` | `cmd_compare` |
| 10 | sample | 3912-3989 | `add_sample_parser` | `estimate/sample_command.py` | f-strings `est.CLASS_FLOOR` (3934), `est.ZONE_DELTA` (3958); `_product_parser_args`; `cmd_sample` |
| 11 | estimate | 3990-4020 | `add_estimate_parser` | `estimate/estimate_command.py` | `ALSO_ACCEPT_HELP`, `cmd_estimate` |
| 12 | certify | 4021-4049 | `add_certify_parser` | `certify/command.py` | f-string `est.ZONE_DELTA` (4030; `est` is certify.zone there); `CERTIFY_ALSO_ACCEPT_HELP`; `cmd_certify` |
| 13 | decide | 4050-4058 | `add_decide_parser` | `decide/command.py` | `cmd_decide` |
| 14 | plan | 4059-4103 | `add_plan_parser` | `plan/command.py` | `_product_parser_args`, `cmd_plan` |
| 15 | mcp | 4104-4106 | `add_mcp_parser` | `cli/mcp_command.py` | `cmd_mcp` |
| 16 | from-olmoearth | 4107-4127 | `add_from_olmoearth_parser` | `readers/command.py` | `cmd_from_olmoearth` |

The resulting `cli/__init__.py` ends with:

```python
def build_parser():
    p = argparse.ArgumentParser(prog="oe-inferencex", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    add_assess_parser(sub)
    add_demo_parser(sub)
    add_from_embeddings_parser(sub)
    add_boundary_parser(sub)
    add_scale_parser(sub)
    add_grade_parser(sub)
    add_certificate_parser(sub)
    add_verify_parser(sub)
    add_compare_parser(sub)
    add_sample_parser(sub)
    add_estimate_parser(sub)
    add_certify_parser(sub)
    add_decide_parser(sub)
    add_plan_parser(sub)
    add_mcp_parser(sub)
    add_from_olmoearth_parser(sub)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args)
```

`cli/__main__.py`:

```python
"""`python -m oe_inferencex.cli` (tests/test_mcp_server.py:924 starts the MCP server this way)."""
import sys

from oe_inferencex.cli import main

if __name__ == "__main__":
    sys.exit(main())
```

The old `if __name__ == "__main__":` block (lines 4136-4137) is not kept in `cli/__init__.py`. `import sys` stays
there, because `oe_inferencex.cli.sys` is an attribute today.

### 3.3 `estimate.py` (2,974 lines): every top-level name

- The module docstring (lines 1-39) goes verbatim to `estimate/__init__.py`.
- `import math` and `import numpy as np` (lines 40-42) stay in `estimate/__init__.py` for the old namespace, and go
  into each submodule that uses them. Per-class and certify.zone (`null_count`) use the module-level `math`; the
  other functions import `math` and `fractions` locally.
- Names created through `__import__` stay as they are:
  - `@__import__("functools").lru_cache` on `_hypergeom_interval`;
  - `__import__("contextvars").ContextVar` for `RULES_OF`.

| Name | Lines | New module | Notes |
|---|---|---|---|
| `Z95` | 44 | `oe_inferencex/estimate/intervals.py` | tests read `est.Z95`; agency/stratified keeps its own copy |
| `N_STRATA` | 45 | `oe_inferencex/estimate/designs.py` |  |
| `MIN_PER_STRATUM` | 46 | `oe_inferencex/estimate/designs.py` | default argument in designs; read by intervals, error_rate, model_assisted |
| `M_PER_TILE` | 47 | `oe_inferencex/estimate/designs.py` | default of design_effect (intervals) |
| `REVIEW_SET_SIGMAS` | 48 | `oe_inferencex/common/sampling.py` |  |
| `Q_FLOOR` | 49 | `oe_inferencex/estimate/designs.py` |  |
| `MIN_TILES` | 50 | `oe_inferencex/estimate/designs.py` |  |
| `TILES_WARNING` | 51-59 | `oe_inferencex/estimate/error_rate.py` | golden changes.py:74 names ("oe_inferencex.estimate", "TILES_WARNING") |
| `SCOPE_ESTIMATE` | 60-69 | `oe_inferencex/estimate/error_rate.py` | mcp_server:1232; the record imports it |
| `SCOPE_CERTIFY` | 70-73 | `oe_inferencex/certify/zone.py` | sequential:316, mcp_server:1415 |
| `CONDITION_NOTE` | 74-76 | `oe_inferencex/estimate/error_rate.py` |  |
| `CONDITION_WHOLE_MAP` | 77-90 | `oe_inferencex/estimate/error_rate.py` | cli (estimate command) |
| `CLASS_WHOLE_MAP` | 91-96 | `oe_inferencex/estimate/error_rate.py` |  |
| `PER_CLASS_EXACT_NOTE` | 97-102 | `oe_inferencex/estimate/per_class.py` | golden changes.py:147 |
| `PER_CLASS_NOT_GRADED` | 103-104 | `oe_inferencex/estimate/per_class.py` |  |
| `FAMILY_NOTE` | 105-108 | `oe_inferencex/certify/zone.py` |  |
| `FAMILY_NOTE_ONE` | 109-112 | `oe_inferencex/certify/zone.py` |  |
| `UNRECORDED` | 113 | `oe_inferencex/common/sampling.py` |  |
| `CONFIDENCE_REFUSAL` | 114-118 | `oe_inferencex/estimate/designs.py` | the record imports it |
| `wilson_interval` | 121-152 | `oe_inferencex/estimate/intervals.py` | PUBLIC (ox.__all__) |
| `_T975` | 155-156 | `oe_inferencex/estimate/intervals.py` |  |
| `t_quantile_975` | 159-172 | `oe_inferencex/estimate/intervals.py` |  |
| `exact_coverage_srs` | 175-191 | `oe_inferencex/estimate/intervals.py` | PUBLIC |
| `_hyper_tail` | 194-219 | `oe_inferencex/common/hypergeom.py` | PATCHED by tests/test_trust_zone.py:643 (retarget, S3); plan reads `est._hyper_tail` |
| `_TIE` | 223-235 | `oe_inferencex/common/hypergeom.py` | read at call time by _tie and _at_most |
| `EXACT_N_MAX` | 238 | `oe_inferencex/common/hypergeom.py` |  |
| `_tie` | 241-253 | `oe_inferencex/common/hypergeom.py` | sequential reads `est._tie` |
| `_level` | 256-259 | `oe_inferencex/common/hypergeom.py` | sequential, plan, tests and the record read it through est |
| `_float_below` | 262-267 | `oe_inferencex/common/hypergeom.py` |  |
| `_float_above` | 270-275 | `oe_inferencex/common/hypergeom.py` |  |
| `_hyper_tail_exact` | 278-284 | `oe_inferencex/common/hypergeom.py` |  |
| `_at_most` | 287-293 | `oe_inferencex/common/hypergeom.py` | plan and tests read it through est |
| `hypergeom_interval` | 295-322 | `oe_inferencex/common/hypergeom.py` | PUBLIC; REASSIGNED by tests/test_class_design.py:63-84 (retarget to estimate/per_class, S5); agency/stratified imports it lazily |
| `_hypergeom_interval` | 325-348 | `oe_inferencex/common/hypergeom.py` | STATE: one lru_cache object; tests call `est._hypergeom_interval.cache_clear()` |
| `confidence_strata` | 351-355 | `oe_inferencex/estimate/designs.py` | PUBLIC |
| `stratified_mean_and_variance` | 358-382 | `oe_inferencex/estimate/intervals.py` | PUBLIC |
| `stratified_interval` | 385-393 | `oe_inferencex/estimate/intervals.py` | PUBLIC |
| `stratified_interval_wilson` | 396-415 | `oe_inferencex/estimate/intervals.py` | PUBLIC |
| `cluster_interval` | 418-453 | `oe_inferencex/estimate/intervals.py` | PUBLIC |
| `design_effect` | 456-470 | `oe_inferencex/estimate/intervals.py` | PUBLIC |
| `neyman_allocation` | 473-500 | `oe_inferencex/estimate/designs.py` | PUBLIC |
| `equal_allocation` | 503-546 | `oe_inferencex/estimate/designs.py` | PUBLIC; tests compare its signature |
| `CLASS_FLOOR` | 549 | `oe_inferencex/estimate/designs.py` | sample registrar help text, mcp_server:913 |
| `class_allocation` | 552-585 | `oe_inferencex/estimate/designs.py` |  |
| `draw_stratified` | 588-594 | `oe_inferencex/estimate/designs.py` |  |
| `_condition_index` | 597-648 | `oe_inferencex/common/sampling.py` | review/assess imports it (S3 repoints assess to common/sampling) |
| `sample_for_estimation` | 651-815 | `oe_inferencex/estimate/designs.py` | PUBLIC; edit lazy imports 710, 718 (S3) |
| `_condition_guards` | 818-870 | `oe_inferencex/common/sampling.py` | used by error_rate, per_class and certify/zone; in common so certify never imports estimate |
| `_per_condition` | 873-892 | `oe_inferencex/estimate/error_rate.py` |  |
| `_union_interval` | 895-934 | `oe_inferencex/estimate/intervals.py` | tests and the record read it through est |
| `_hypergeom_fractions` | 937-945 | `oe_inferencex/common/hypergeom.py` |  |
| `UNJUDGED_NOTE` | 948-954 | `oe_inferencex/estimate/error_rate.py` |  |
| `UNJUDGED_STRATIFIED_NOTE` | 955-959 | `oe_inferencex/estimate/error_rate.py` |  |
| `REVIEWER_NOTE` | 960-965 | `oe_inferencex/estimate/error_rate.py` |  |
| `_stratum_union` | 968-980 | `oe_inferencex/estimate/error_rate.py` |  |
| `SEQUENTIAL_READ_NOTE` | 983-985 | `oe_inferencex/estimate/error_rate.py` | cli (estimate command) |
| `estimate_error_rate` | 988-1103 | `oe_inferencex/estimate/error_rate.py` | PUBLIC |
| `_estimate_core` | 1106-1258 | `oe_inferencex/estimate/error_rate.py` |  |
| `review_set_threshold` | 1261-1274 | `oe_inferencex/common/sampling.py` |  |
| `review_set_check` | 1277-1319 | `oe_inferencex/common/sampling.py` | PUBLIC; used by error_rate, certify/zone and sequential |
| `estimate_from_indices` | 1322-1345 | `oe_inferencex/estimate/error_rate.py` | PUBLIC |
| `MIN_FOR_TUNING` | 1348-1354 | `oe_inferencex/estimate/model_assisted.py` |  |
| `tuned_coefficient` | 1357-1363 | `oe_inferencex/estimate/model_assisted.py` |  |
| `model_assisted_interval` | 1366-1379 | `oe_inferencex/estimate/model_assisted.py` |  |
| `stratified_model_assisted_interval` | 1382-1408 | `oe_inferencex/estimate/model_assisted.py` |  |
| `MIN_PER_CLASS` | 1411-1418 | `oe_inferencex/estimate/per_class.py` |  |
| `RARE_ERRORS` | 1419-1420 | `oe_inferencex/estimate/per_class.py` |  |
| `_wald` | 1423-1425 | `oe_inferencex/estimate/per_class.py` |  |
| `_wilson_eff` | 1428-1441 | `oe_inferencex/estimate/per_class.py` |  |
| `_interval` | 1444-1449 | `oe_inferencex/estimate/per_class.py` |  |
| `_ht_total` | 1452-1466 | `oe_inferencex/estimate/per_class.py` |  |
| `_ht_ratio` | 1469-1479 | `oe_inferencex/estimate/per_class.py` |  |
| `CLASS_DESIGN_METHOD` | 1482-1487 | `oe_inferencex/estimate/per_class.py` |  |
| `CLASS_NORMAL_NOTE` | 1488-1490 | `oe_inferencex/estimate/per_class.py` |  |
| `CLASS_BLIND_NOTE` | 1491-1496 | `oe_inferencex/estimate/designs.py` | sample command, mcp_server:1035 |
| `CLASS_DESIGN_EXACT_NOTE` | 1497-1500 | `oe_inferencex/estimate/per_class.py` |  |
| `_class_design_exact` | 1503-1549 | `oe_inferencex/estimate/per_class.py` | reads `hypergeom_interval` from estimate/per_class globals (the patch target of tests/test_class_design.py) |
| `estimate_per_class` | 1552-1871 | `oe_inferencex/estimate/per_class.py` |  |
| `ZONE_GRID` | 1874-1880 | `oe_inferencex/certify/zone.py` | default argument bound when sequential, plan and certify define their functions |
| `ZONE_DELTA` | 1881 | `oe_inferencex/certify/zone.py` | default argument in estimate, sequential, plan; cli and mcp_server read it |
| `ZONE_RULES` | 1882 | `oe_inferencex/certify/zone.py` |  |
| `TARGET_WEIGHT` | 1883-1888 | `oe_inferencex/certify/zone.py` | plan, sample command |
| `TARGET_NOTE` | 1889-1893 | `oe_inferencex/certify/zone.py` |  |
| `TARGET_ALONE_NOTE` | 1894-1898 | `oe_inferencex/certify/zone.py` |  |
| `ZONE_CUTS` | 1899 | `oe_inferencex/certify/zone.py` |  |
| `ZONE_CUT` | 1900 | `oe_inferencex/certify/zone.py` | STATE read at call time (zone_cut, certify_zone, plan via `est.ZONE_CUT`); never from-import it into plan |
| `PREFIX_NOTE` | 1901-1910 | `oe_inferencex/certify/zone.py` | golden changes.py:29 |
| `_log_choose` | 1913-1915 | `oe_inferencex/common/hypergeom.py` | sequential reads `est._log_choose` |
| `hypergeom_cdf` | 1918-1934 | `oe_inferencex/common/hypergeom.py` |  |
| `zone_edge` | 1937-1952 | `oe_inferencex/certify/zone.py` |  |
| `_edge_of` | 1955-1961 | `oe_inferencex/certify/zone.py` | sequential reads `est._edge_of` |
| `RULES_OF` | 1964-1966 | `oe_inferencex/certify/zone.py` | STATE: the ONE ContextVar "oe_inferencex_rules_of"; set by certify/certificate `_rules_of`, read by null_count and common/wording `class_words` |
| `null_count` | 1969-1978 | `oe_inferencex/certify/zone.py` | PATCHED by tests/test_bughunt_1_17_2.py:814 (retarget, S4); sequential and plan read `est.null_count` |
| `zone_pvalue` | 1981-1997 | `oe_inferencex/certify/zone.py` | PATCHED by tests/test_trust_zone.py:629 (retarget, S4); must sit with certify_zone |
| `zone_pvalue_exact` | 2001-2011 | `oe_inferencex/certify/zone.py` |  |
| `zone_upper_bound` | 2014-2031 | `oe_inferencex/certify/zone.py` |  |
| `min_labels_to_certify` | 2034-2049 | `oe_inferencex/certify/zone.py` | sequential, plan, sample command, mcp_server |
| `zone_order` | 2052-2062 | `oe_inferencex/certify/zone.py` |  |
| `zone_cut` | 2065-2088 | `oe_inferencex/certify/zone.py` |  |
| `zone_levels` | 2091-2098 | `oe_inferencex/certify/zone.py` |  |
| `zone_counts` | 2101-2109 | `oe_inferencex/certify/zone.py` |  |
| `target_weights` | 2112-2129 | `oe_inferencex/certify/zone.py` |  |
| `apply_zone_rule` | 2132-2180 | `oe_inferencex/certify/zone.py` |  |
| `certify_zone` | 2183-2299 | `oe_inferencex/certify/zone.py` | the record's test_exp89 patches it on its own `est` (it calls `est.certify_zone`) |
| `_family_note` | 2302-2323 | `oe_inferencex/certify/zone.py` |  |
| `certify_by_condition` | 2326-2455 | `oe_inferencex/certify/zone.py` | PUBLIC |
| `WHICH_MAP_NOTE` | 2458-2466 | `oe_inferencex/estimate/which_map.py` |  |
| `SIGN_TEST_SHARE` | 2467 | `oe_inferencex/estimate/which_map.py` |  |
| `SIGN_TEST_NOTE` | 2468-2473 | `oe_inferencex/estimate/which_map.py` |  |
| `_SIGN_CRIT` | 2474 | `oe_inferencex/estimate/which_map.py` | STATE: module-level memo dict |
| `sign_critical` | 2477-2495 | `oe_inferencex/estimate/which_map.py` |  |
| `which_map_rule` | 2498-2507 | `oe_inferencex/estimate/which_map.py` |  |
| `sample_disagreement` | 2510-2528 | `oe_inferencex/estimate/which_map.py` |  |
| `compare_from_disagreement` | 2531-2612 | `oe_inferencex/estimate/which_map.py` | edit lazy import 2590 (S1) |
| `AUDIT_SHARE` | 2615-2625 | `oe_inferencex/estimate/audit.py` |  |
| `AUDIT_ZONE_TEXT` | 2626-2629 | `oe_inferencex/estimate/audit.py` |  |
| `AUDIT_METHOD` | 2630-2632 | `oe_inferencex/estimate/audit.py` |  |
| `AUDIT_NOTE` | 2633-2636 | `oe_inferencex/estimate/audit.py` |  |
| `AUDIT_READING_NOTE` | 2637-2647 | `oe_inferencex/estimate/audit.py` |  |
| `ESTIMATE_READING_NOTE` | 2648-2659 | `oe_inferencex/estimate/error_rate.py` |  |
| `ISSUED_NOTE` | 2660-2668 | `oe_inferencex/certify/zone.py` | golden changes.py:97; mcp_server:1408 |
| `ZONE_READING_NOTE` | 2669-2676 | `oe_inferencex/certify/zone.py` |  |
| `ZONE_READING_SEQUENTIAL` | 2677-2678 | `oe_inferencex/certify/zone.py` |  |
| `_audit_zone` | 2681-2714 | `oe_inferencex/estimate/audit.py` |  |
| `sample_class_audit` | 2717-2764 | `oe_inferencex/estimate/audit.py` |  |
| `interval_verdict` | 2767-2773 | `oe_inferencex/estimate/audit.py` | the record imports it |
| `_code` | 2776-2792 | `oe_inferencex/common/readings.py` | not agency._code (a different function); both stay |
| `_also_accept_pairs` | 2795-2816 | `oe_inferencex/common/readings.py` | common/labels `_also_accept` reads it as `est._also_accept_pairs` (est bound to common/readings there) |
| `class_audit_from_sample` | 2819-2920 | `oe_inferencex/estimate/audit.py` |  |
| `also_accept_reading` | 2923-2974 | `oe_inferencex/common/readings.py` | common/labels `_also_accept_rows` reads it (shared by estimate and certify commands) |

Counts: common/hypergeom 14, common/sampling 6, common/readings 3, estimate/designs 14, estimate/intervals 11,
estimate/error_rate 15, estimate/model_assisted 4, estimate/per_class 14, certify/zone 32, estimate/which_map 8,
estimate/audit 9. Total 130.

#### 3.3.1 Imports of the estimate split

The cross-module references in the code at 987dc7a. Docstring mentions are excluded; the local variable
`interval_verdict` in `compare_from_disagreement`, at line 2588, is not a reference to the function.

| Module | Imports at module top |
|---|---|
| `common/hypergeom.py` | numpy as used (functions import `math` and `fractions` locally, as today) |
| `common/sampling.py` | numpy |
| `common/readings.py` | numpy as used |
| `estimate/designs.py` | `common.sampling`: _condition_index. `sequential_order` stays a lazy import inside `sample_for_estimation`, repointed (§3.9) |
| `estimate/intervals.py` | `estimate.designs`: MIN_PER_STRATUM, M_PER_TILE; `common.hypergeom`: _hypergeom_fractions |
| `estimate/error_rate.py` | `estimate.intervals`: _union_interval, cluster_interval, stratified_interval_wilson, stratified_mean_and_variance, wilson_interval; `estimate.designs`: MIN_PER_STRATUM; `common.hypergeom`: _float_above, _float_below, _level, hypergeom_interval; `common.sampling`: _condition_guards, review_set_check |
| `estimate/model_assisted.py` | `estimate.intervals`: Z95; `estimate.designs`: MIN_PER_STRATUM |
| `estimate/per_class.py` | `import math`; `estimate.intervals`: Z95, _union_interval, stratified_interval_wilson, stratified_mean_and_variance, wilson_interval; `common.hypergeom`: hypergeom_interval; `common.sampling`: _condition_guards |
| `estimate/which_map.py` | `common.hypergeom`: _level, hypergeom_interval. The lazy `stats` import stays inside `compare_from_disagreement`, repointed (§3.9) |
| `estimate/audit.py` | `estimate.designs`: draw_stratified; `common.hypergeom`: _level, hypergeom_interval; `common.readings`: _also_accept_pairs |
| `certify/zone.py` | `import math`; `common.hypergeom`: _at_most, _hyper_tail_exact, _level, hypergeom_cdf; `common.sampling`: _condition_guards, review_set_check. **Nothing from `oe_inferencex.estimate`** (R3) |
| `estimate/__init__.py` | the docstring; `import math`; `import numpy as np`; then the re-export of all 130 names. Order: `common.hypergeom`, `common.sampling`, `common.readings`, `estimate.designs`, `estimate.intervals`, `estimate.error_rate`, `estimate.model_assisted`, `estimate.per_class`, `certify.zone`, `estimate.which_map`, `estimate.audit`. Private names are named explicitly. It imports no command module (R2) |

No edge points from `designs` to `intervals`, so `intervals` → `designs` is one-way. `certify.zone` imports only
`common`. That is why `estimate/__init__` can re-export `certify.zone` without a cycle (§5.1).

### 3.4 `sequential.py` (350 lines) → `certify/sequential.py`, its old path answered by the alias finder

| Names | New module |
|---|---|
| `_null_count` 53-56, `_log_tail` 59-83, `_log_upper_count` 86-98, `log_evalue` 101-115, `evalue_exact` 118-132, `reaches` 135-146, `min_labels_sequential` 149-176, `level_path` 179-208, `_finite` 211-212, `SEQUENTIAL_NOTE` 215-218, `ANCHOR_CAP` 231, `anchor_coverage` 234-247, `certify_zone_sequential` 250-350; module docstring; imports `math`, `Fraction`, `np` | `oe_inferencex/certify/sequential.py` |
| `sequential_order` 221-228 | `oe_inferencex/common/sampling.py`. `certify/sequential.py` re-binds it with `from oe_inferencex.common.sampling import sequential_order`, for `certify_zone_sequential` (294) and for `sq.sequential_order` in the tests |
| `est` (line 50, `from . import estimate as est`) | becomes `from oe_inferencex import estimate as est` (§3.9, §4.5) |

### 3.5 `certificate.py` (1,289 lines) → `certify/certificate.py`, its old path answered by the alias finder

Every top-level name moves unchanged:

- **Constants:** CERT_VERSION, BUNDLE, ZONE_KEYS, PER_CONDITION_KEYS, SIDE_OWN, ZONE_OWN, ZONE_ADDED, RULES_CHANGED,
  `_RUN` (STATE: the one RLock shared by `_redraw` and `_verify`), CSV_INTEGERS, RASTER_MAGIC, ISSUERS_WORD, OS_FILES,
  MAX_BYTES, MASK_TAGS, RFC3339, RELEASE, ALPHA_RANGE, DELTA_RANGE.
- **Classes:** Unreadable, `_Sizes`.
- **Functions:** `_load`, `_regular`, `_crs_text`, `_crs_spelled`, `_same_crs`, `_terms`, pixel_digest, file_sha256,
  `_nodata`, `_plain_raster`, `_closed_gdal`, raster_digest, map_digests, names_given, `_grid_level`, seed_settings,
  seed_from, draw_seed, `_side_settings`, cli_where, `_reads_alone`, `_map_paths`, `_rfc3339`, `_footprint`, `_name`,
  printable, `_statement`, `_item`, `_map_grid`, `_declared`, `_layer_on_grid`, `_bundle_file`, `_by_class`,
  `_cond_name`, build, `_write_bundle`, `_check`, `_redraw`, `_arg`, `_strip_added`, `_certified_sizes`, `_release`,
  `_rules_of`, `_class_words_1171`, `_without_added`, `_differ`, `_rounding_tolerance`, csv_rows, `_csv_differs`,
  `_tiff_pages`, `_read_mask`, `_short`, verify, `_verify`, `_issuer_fields_ok`, `_anchor_history`, point.
- **The module's imports**, among them `hashlib`, which a test reads as `cert.hashlib`.

The only edits are the 15 lazy import lines listed in §3.9.

Not recommended for 1.19.0: splitting `certificate.py` into issue and verify modules. `_write_bundle` calls `verify`,
which would force three modules (bundle ← verify ← issue). The split would also move the patch target `_rules_of`.

### 3.6 Modules whose folder takes their name: the old code becomes the package `__init__`

| Old file | New file | Every top-level name, all moving with the file | Edits |
|---|---|---|---|
| `compare.py` | `compare/__init__.py` | NAN, _decision, _indicator, _same_shape, _group_ids, _count_by_group, _key, _phi_counts, phi, disagreement, where, stability, over_groups, _crosstab_counts, crosstab, which_side, _utc_day, _iso_day, _period, _iso, dates_reading, compare_inferences, _relation, _graded_against, determinism_check; imports math, np, cue_enrichment, TIE_TOL, sign_test, wins_losses_ties | lines 38-39 (§3.9) |
| `scale.py` | `scale/__init__.py` | PATCHES, SURVIVAL_LINE, CENSUS_NOTE, UNSEEN_NOTE, ZOOM_NOTE, NO_WINDOW_NOTE, pixel_classes, _check_patches, _surviving, _reference_block, scale_profile; imports np and the 8 assess names | line 17 (§3.9) |
| `plan.py` | `plan/__init__.py` | POWER, TAIL, MAX_LABELS, SPLITS, DENSE_SPLITS, WORST_KEPT, ZONE_DRAWS, NUMPY_URN, DEFAULT_RATE, PLAN_NOTE, INTERVAL_NOTE, STEPS, ladder, runs, recommend, REFINE, WINDOW, WINDOW_MAX, refine, _check, _check_count, _hyper_pmf, `_TABLES` (STATE), _gallop, _ends_range, `_ends` (STATE: lru_cache), error_rate_probability, plan_error_rate, class_flag_probability, _class_counts, plan_class, which_map_probability, _parity, _worst_split, plan_which_map, _grid_level, zone_floor, entry_budgets, _zone_index, zone_level_probability, target_probability, _pass_probability, _cdf, bonferroni_probability, replacement_bound, zone_prefix_probability, plan_zone; imports functools, math, np, est | line 39 (§3.9). `est.ZONE_CUT` (552) stays an attribute read at call time |
| `decide.py` | `decide/__init__.py` | QUESTIONS, NO_LABELS_WHICH, PLUGIN_REFUSED, SEPARATE, DATES_APART, EACH_AT_ITS_DATE, NO_LABELS_WHICH_DATES, NO_LABELS_WHICH_ONE_DATE, _no_labels_which, _pc, interval_second_look, comparison_second_look, certificate_second_look, readings_joint, readings_second_look, kind, available, parse, _labels_clause, _warned, _rate_below, _answer_error_rate_below, _answer_class, _answer_class_audit, _as_sentence, _apart, _dates_clause, _level_pc, sign_test_head, _answer_more_accurate, _answer_trusted, _answer_share_differs, decide; `from __future__ import annotations` stays the first statement; imports json, os | line 693 (§3.9). **No submodule may be named `decide.py`** (it would rebind `oe_inferencex.decide.decide`) |
| `agency.py` | `agency/__init__.py` | UNKNOWN_CODES, WEIGHT_TOLERANCE, `_OTHER` (sentinel), INTERVAL_NOTE, UNKNOWN_NOTE, NOT_COMPUTED_NOTE, PURE_PARTLY_NOTE, UNSAMPLED_NOTE, POINT_READ_NOTE, SAMPLED_ONLY_NOTE, PURE_NOTE, PURE_SKIPPED_NOTE, ONE_LABEL_NOTE, ALTERNATE_RULE, PRIMARY_RULE, _code, _half_unit, _read_table, _sort_key, read_points, read_strata, _frame, design_from_weights, design_from_sizes, _alts, _right, _ref_class, _completions, _diff_bounds, _oa_bounds, _class_bounds, _natural, grade, _quantities, linear_estimate, _q, _pure_plan, _error_matrix; imports csv, math, re, np and the 7 stratified names | none: `from .stratified import ...` (28) resolves to `agency/stratified.py`. **No submodule may be named `grade.py`** |
| `stratified.py` | `agency/stratified.py` (old path answered by the alias finder) | Z95, KAPPA, K_MIX, SPAN, PURE_GAMMA, `MAX_WORK` (patched by tests/test_agency.py:300), METHOD_LINEAR, METHOD_BOX, METHOD_PURE, TooLarge, _logsumexp, _stratum_logavg, c_grid, Design, _strata, _log_parts, _root_low, betting_interval, normal_linear, linear, _z, normal_ratio, ratio, pure_stratum_ratio; imports math, np | line 364 (§3.9) |

### 3.7 Whole modules moved to a folder, each old path answered by the alias finder

Every top-level name moves with its module. Each module keeps its imports, its module-level state and its import
guards byte-identical. The guards' messages name the old paths ("oe_inferencex.evidence needs torch"), which still
work.

| Old file | New file | Top-level names | Edits |
|---|---|---|---|
| `assess.py` | `review/assess.py` | QUANTILES, MAX_CONDITIONS, RULE_TEXT, SCOPE_ASSESS, SCOPE_ASSESS_K, PROBABILITY_WARNING, LOGIT_REGRESSION, LOGIT_RANGE_WARNING, LOGIT_NONNEGATIVE_WARNING, LOGIT_NONPOSITIVE_WARNING, MARGIN_FORM_WARNING, CLASS_SHARE_TEXT, NO_VALID_WINDOW, _check_patch, _as_mask, _masked_as_nodata, _check_shape, _reference_index, _nonfinite_as_nodata, _pool, _pool_valid, `_pooled_argmax` (patched by tests/test_bughunt_1_17_2.py:241; `_assess` and `pool_condition` call it by bare name, keep it so), pool_condition, _boundary_valid, ORDERS, boundary_first_score, _sliding, boundary_band, assess_classmap, assess_prediction, _check_probabilities, _assess, _condition_block, _same_mean, _tied_with, _n_distinct, review_order, review_mask, summary; imports warnings, np, `_condition_index`, aurc_expected, boundary_indicator, midrank_pct | 51, 52, 53, 686 |
| `explain.py` | `review/explain.py` | Cue, BOLIVIA, EXP82_BOUNDARY, EXP82_LOW_CONFIDENCE, BOUNDARY_SATURATED, WC_DISAGREE, WC_DISAGREE_23, CUES, _valid, derive_cues, top_fraction, cooccurrence, cue_enrichment, explain_review_set, library_table, confusion_pairs | none |
| `calibrate.py` | `review/calibrate.py` | MIN_GROUP_ERRORS, _logistic, Fusion, _standardise, _folds, _crossfit, _prior_logit, _crossfit_prior_logit, _scoreable, fit_ranker, side_features, fit_side | 24 (23 stays: `oe_inferencex.compare` is a package) |
| `signals.py` | `common/signals.py` | S2_BANDS, confidence, boundary_indicator, aligned_tile_phase, crop_dependence, _crop, ndwi, ndwi_gradient, ndwi_level, s2_patch_variance, midrank_pct, combine_midrank, shift_averaged_probability, pool_to_windows | none |
| `metrics.py` | `common/metrics.py` | _NAN, _expected_cum_errors, aurc_expected, risk_coverage, oracle_aurc, excess_aurc, capture_at_budget, augrc, augrc_from_auroc, attainable_ceiling, selective_accuracy, expected_calibration_error, capture_at_budget_expected, _check_weights, weighted_mean, weighted_aurc, weighted_excess_aurc, weighted_capture_at_budget, weighted_auroc | none |
| `stats.py` | `common/stats.py` | TIE_TOL, sign_test, rank_sum_test, wins_losses_ties, paired_comparison, clustered_sign_test, block_bootstrap_indices, cluster_bootstrap_difference, spearman, paired_cluster_bootstrap; imports comb, np, aurc_expected, midrank_pct | 16, 17 |
| `reliability.py` | `research/reliability.py` | CHUNK, _chunks, _dist, kmeans, centroid_signals, input_extremity, _entropy, ensemble_uncertainty, bootstrap_tiles, standardize, poly2, dawid_skene | none |
| `evidence.py` | `research/evidence.py` | the torch guard; re-exports aurc_expected, risk_coverage, dawid_skene; train_logistic_head, predict_head, predict_logit, rasterize_polyline, pool_to_patches, train_softmax_head, predict_softmax_head | 9, 10 |
| `awf.py` | `research/awf.py` | the guard; ROOT (patched by the record's test_exp89 at a pin bump), CROP, N_MONTHS, FILL, GROUPS, list_windows, load_window_full, crop_stack, _normalizer, stacks_to_sample; the third-party Modality, Normalizer, Strategy, MaskedOlmoEarthSample, MaskValue | none |
| `data.py` | `research/data.py` | the guard; STAC, _catalog, fetch_s2_window, fetch_worldcover_window, s2_to_sample, embed; the third-party imports (the record reads `Modality` through it) | 44 |
| `figstyle.py` | `research/figstyle.py` | the guard; `matplotlib.use("Agg")` (an import side effect, unchanged); PATCH_M, AWF_CLASSES, setup, letter, map_panel, rc_panel | none |
| `olmoearth.py` | `readers/olmoearth.py` | SOURCE, PROB_TOLERANCE, SUM_TOLERANCE, `MAX_GRID_BYTES` (patched, tests/test_olmoearth.py:508), SCL_CLOUDY, COVER, CLOUD, SCORES_JSON, CONDITIONS_JSON, FIX_PROBABILITIES, _rasterio, bandset_dirname, _bands_of_dirname, assess_command, _windows, _metadata, _completed_outputs, _no_output, output_kind, _plan_grids, SPARSE_GRID, _check_size, _write_tif, _class_ids, _probabilities, read_output, read_window_probs, _input_layers, _bandset_header, _item_groups, _read_group, _code_name, _command_names, condition_rule, condition_from_inputs; `MAX_CONDITIONS`, its own binding copied from assess (patched, tests/test_olmoearth.py:410; keep the module-level copy) | 55 |
| `embeddings.py` | `readers/embeddings.py` | STORE, ENDPOINT, MAX_PIXELS, _need, open_store, dequantize, zone_name, _grid, read_area, read_points, fit_and_predict, write_scores | none |
| `lcc.py` | `readers/lcc.py` | BASE, TILE_PX, BANDS, LAND_COVER, _TYPE_SIZE, _TYPE_FMT, _resolve, `_range` (patched, tests/test_bugfix_periphery.py:159), LCCTile, tile_for, read_window, change_probability, nodata_mask | none |
| `taskcard.py` | `readers/taskcard.py` (keep the basename: argparse's prog comes from argv[0]) | RAW, GH_API, PROJECTS, S2_DEFAULT_GROUPS, `_get` (patched, tests/test_taskcard.py:14), TaskCard, encoder_card, _walk, _task_from_node, `_LISTING_UNREACHABLE` (STATE), _variants, project_cards, _timesteps, project_card, _audit_settings, lcc_card, to_markdown (its string at 347 naming `oe_inferencex/taskcard.py` was moved byte-identical; the review fix 796dcb1 then made it name `oe_inferencex/readers/taskcard.py`, §8), main, the `__main__` block | none in the move; the to_markdown line in review |

### 3.8 Modules that stay where they are

| File | Names | Edits |
|---|---|---|
| `__init__.py` | the imports at lines 55-75; `__version__`; `__all__`; `_version`; PackageNotFoundError | **Imports repointed to the new paths.** The finder is installed at the end of the module, so the package's own imports cannot go through old paths. `from .review.assess import (...)` and the rest import the same objects as before, and `from .common import metrics, signals, stats`, `from .review import assess, calibrate, explain` and `from .agency import stratified` bind the module attributes the 1.18.1 imports bound as a side effect, so `import oe_inferencex` still binds `oe_inferencex.metrics`, `.assess` and so on, as the moved modules. Added: `_MOVED` and the class `_MovedModules`, inserted once at `sys.meta_path[0]` (§4.2). Changed: the docstring, whose module list (16-37) becomes a folder list and whose "the layout stays flat (ADR-001)" (38-39) becomes "feature folders (docs/plan/adr-002-feature-folders.md in the research record)"; and the fallback `"1.18.1"` at line 80, which becomes `"1.19.0"` in S14. |
| `demo.py` | UNKNOWN, LAND, WATER, FLAG, WRONG, RANDOM, INK, DW_COLOURS, SAMPLE, FONT, sample_scene, write_png, _outline, _text, picture, _made_up, _real, run | Lines 23, 24 and 166 (§3.9). `cli.cmd_assess` (175) stays as it is: the cli package re-exports it. The stale comment at 165 is left alone (§7.2). |
| `mcp_server.py` | every name (NEEDS_EXTRA ... main, `Field`, `_CAPTURE`, `TOOLS`, the 9 tools) | Lines 59 and 1728 (§3.9). Lines 58 (`__version__, cli`), 60 (`est`), 61 (`_decide`) and 488 (`from .cli import class_words`) stay: they name real packages. **`cli.main` stays an attribute call (362)**, because tests/test_mcp_server.py:270 patches `mcp_server.cli.main`. |
| `sample/` | NOTICE.txt, dynamic_world_tile.npz | none |

### 3.9 Every import-line edit, by step

These are the only text changes allowed inside moved code. "Lazy" means the line sits inside a function and must
stay there.

| Step | File (new path) : old line | Old | New |
|---|---|---|---|
| S1 | common/stats.py:16 | `from oe_inferencex.metrics import aurc_expected` | `from oe_inferencex.common.metrics import aurc_expected` |
| S1 | common/stats.py:17 | `from oe_inferencex.signals import midrank_pct` | `from oe_inferencex.common.signals import midrank_pct` |
| S1 | assess.py:52 | `from oe_inferencex.metrics import aurc_expected` | `from oe_inferencex.common.metrics import aurc_expected` |
| S1 | assess.py:53 | `from oe_inferencex.signals import boundary_indicator, midrank_pct` | `from oe_inferencex.common.signals import boundary_indicator, midrank_pct` |
| S1 | calibrate.py:24 | `from oe_inferencex.metrics import ...` | `from oe_inferencex.common.metrics import ...` (same names) |
| S1 | compare.py:39 | `from oe_inferencex.stats import TIE_TOL, sign_test, wins_losses_ties` | `from oe_inferencex.common.stats import TIE_TOL, sign_test, wins_losses_ties` |
| S1 | evidence.py:9 | `from oe_inferencex.metrics import aurc_expected, risk_coverage  # noqa...` | `from oe_inferencex.common.metrics import aurc_expected, risk_coverage  # noqa...` |
| S1 | demo.py:24 | `from oe_inferencex.metrics import attainable_ceiling` | `from oe_inferencex.common.metrics import attainable_ceiling` |
| S1 | cli.py:60 | `from oe_inferencex.signals import boundary_indicator` | `from oe_inferencex.common.signals import boundary_indicator` |
| S1 | estimate.py:2590 (lazy, `compare_from_disagreement`) | `from . import stats` | `from oe_inferencex.common import stats` |
| S2 | review/assess.py:686 (lazy, `_assess`) | `from oe_inferencex.explain import confusion_pairs` | `from oe_inferencex.review.explain import confusion_pairs` |
| S2 | cli.py:55-56 | `from oe_inferencex.assess import (...)` | `from oe_inferencex.review.assess import (...)` (same names) |
| S2 | cli.py:59 | `from oe_inferencex.explain import explain_review_set` | `from oe_inferencex.review.explain import explain_review_set` |
| S2 | cli.py:611 (lazy, `cmd_boundary`) | `from .assess import boundary_band` | `from oe_inferencex.review.assess import boundary_band` |
| S2 | compare.py:38 | `from oe_inferencex.explain import cue_enrichment` | `from oe_inferencex.review.explain import cue_enrichment` |
| S2 | scale.py:17-18 | `from .assess import (NO_VALID_WINDOW, ...)` | `from oe_inferencex.review.assess import (NO_VALID_WINDOW, ...)` (same names) |
| S2 | olmoearth.py:55 | `from oe_inferencex.assess import MAX_CONDITIONS` | `from oe_inferencex.review.assess import MAX_CONDITIONS` |
| S2 | demo.py:23 | `from oe_inferencex.assess import assess_prediction` | `from oe_inferencex.review.assess import assess_prediction` |
| S2 | demo.py:166 (lazy, `run`) | `from oe_inferencex.assess import _pool, _pooled_argmax` | `from oe_inferencex.review.assess import _pool, _pooled_argmax` |
| S2 | mcp_server.py:59 | `from oe_inferencex import assess as _assess` | `from oe_inferencex.review import assess as _assess` |
| S3 | review/assess.py:51 | `from oe_inferencex.estimate import _condition_index` | `from oe_inferencex.common.sampling import _condition_index` |
| S3 | estimate.py:710, 718 (lazy, `sample_for_estimation`) | `from .sequential import sequential_order` | `from oe_inferencex.common.sampling import sequential_order` |
| S3 | sequential.py (new line after 50) | (none) | `from oe_inferencex.common.sampling import sequential_order` |
| S6 | certificate.py:230, 244, 330 (lazy) | `from .cli import read_raster` | `from oe_inferencex.common.raster import read_raster` |
| S6 | certificate.py:324 (lazy, `cli_where`) | `from .cli import _where` | `from oe_inferencex.common.output import _where` |
| S6 | certificate.py:489 (lazy, `_declared`) | `from .cli import _mask_bands` | `from oe_inferencex.common.raster import _mask_bands` |
| S6 | certificate.py:624 (lazy, `_write_bundle`) | `from .cli import write_raster` | `from oe_inferencex.common.raster import write_raster` |
| S6 | certificate.py:780 (lazy, `_class_words_1171`) | `from .cli import _CLASS_WORDS, _PATH_KEYS` | `from oe_inferencex.common.wording import _CLASS_WORDS, _PATH_KEYS` |
| S6 | certificate.py:833 (lazy, `_rounding_tolerance`) | `from .cli import _rounding_tolerance as tol` | `from oe_inferencex.common.labels import _rounding_tolerance as tol` |
| S6 | certificate.py:1206, 1278 (lazy, `_verify`, `point`) | `from .cli import affine_apply` | `from oe_inferencex.common.raster import affine_apply` |
| S6 | data.py:44 (lazy, `fetch_s2_window`) | `from .cli import affine_apply` | `from oe_inferencex.common.raster import affine_apply` |
| S6 | decide.py:693 (lazy, `decide`) | `from .cli import class_wording, result_class_names` | `from oe_inferencex.common.wording import class_wording, result_class_names` |
| S6 | common/wording.py (`class_words`, was cli.py:1934, lazy) | `from .estimate import RULES_OF` | `from oe_inferencex.certify.zone import RULES_OF` |
| S6 | common/wording.py (`class_words`, was cli.py:1936, lazy) | `from .certificate import _class_words_1171` | `from oe_inferencex.certificate import _class_words_1171` (S8 changes it again) |
| S6 | common/labels.py (`_csv_rows`, was cli.py:2469, lazy) | `from .certificate import csv_rows` | `from oe_inferencex.certificate import csv_rows` (S8 changes it again) |
| S7 | research/evidence.py:10 | `from oe_inferencex.reliability import dawid_skene  # noqa...` | `from oe_inferencex.research.reliability import dawid_skene  # noqa...` |
| S7 | cli.py:585 (lazy, `cmd_from_embeddings`) | `from . import embeddings as emb` | `from oe_inferencex.readers import embeddings as emb` |
| S7 | cli.py:3683 (lazy, `cmd_from_olmoearth`) | `from oe_inferencex import olmoearth as oe` | `from oe_inferencex.readers import olmoearth as oe` |
| S8 | certify/sequential.py:50 | `from . import estimate as est` | `from oe_inferencex import estimate as est` |
| S8 | certify/certificate.py:266 (lazy, `_grid_level`) | `from .estimate import ZONE_GRID` | `from oe_inferencex.certify.zone import ZONE_GRID` |
| S8 | certify/certificate.py:530 (lazy, `build`) | `from .compare import _period` | `from oe_inferencex.compare import _period` |
| S8 | certify/certificate.py:637 (lazy, `_write_bundle`) | `from . import __version__` | `from oe_inferencex import __version__` (stays inside the function: tests patch the package's version) |
| S8 | certify/certificate.py:675, 987 (lazy, `_redraw`, `_verify`) | `from . import cli` | `from oe_inferencex import cli` |
| S8 | certify/certificate.py:766 (lazy, `_rules_of`) | `from . import estimate as est` | `from oe_inferencex.certify import zone as est` |
| S8 | certify/certificate.py:1247 (lazy, `_anchor_history`) | `from .sequential import anchor_coverage` | unchanged: it now resolves to `certify/sequential.py` |
| S8 | common/wording.py (`class_words`, lazy) | `from oe_inferencex.certificate import _class_words_1171` | `from oe_inferencex.certify.certificate import _class_words_1171` |
| S8 | common/labels.py (`_csv_rows`, lazy) | `from oe_inferencex.certificate import csv_rows` | `from oe_inferencex.certify.certificate import csv_rows` |
| S8 | cli.py:923, 943 (lazy, `cmd_certificate`, `cmd_verify`) | `from . import certificate as cert` | `from oe_inferencex.certify import certificate as cert` |
| S8 | cli.py:1562 (lazy, `cmd_sample`) | `from .certificate import draw_seed` | `from oe_inferencex.certify.certificate import draw_seed` |
| S8 | cli.py:1656 (lazy, `cmd_sample`) | `from .certificate import names_given, raster_digest` | `from oe_inferencex.certify.certificate import names_given, raster_digest` |
| S8 | cli.py:1547, 3168 (lazy, `cmd_sample`, `_certify_sequential`) | `from oe_inferencex import sequential as sq` | `from oe_inferencex.certify import sequential as sq` |
| S8 | mcp_server.py:1728 (lazy, tool `verify`) | `from . import certificate as cert` | `from oe_inferencex.certify import certificate as cert` |
| S9 | plan/__init__.py:39 | `from . import estimate as est` | `from oe_inferencex import estimate as est` |
| S9 | agency/stratified.py:364 (lazy, `pure_stratum_ratio`) | `from .estimate import hypergeom_interval` | `from oe_inferencex.common.hypergeom import hypergeom_interval` |
| S10c | scale/command.py (`cmd_scale`, was cli.py:737, lazy) | `from .scale import scale_profile` | `from oe_inferencex.scale import scale_profile` |

**Unchanged on purpose.** These lazy lines already name an absolute path that still works:

- `cli.py:205` `_stamp`: `from oe_inferencex import __version__`
- `cli.py:850` `cmd_grade`: `from oe_inferencex import agency`
- `cli.py:1311` and `1338` `cmd_compare`: `oe_inferencex.compare`
- `cli.py:1375` `cmd_demo`: `oe_inferencex.demo`
- `cli.py:2231` `_estimate_disagreement`: `oe_inferencex.decide`
- `cli.py:3312` `cmd_decide`: `oe_inferencex.decide`
- `cli.py:3399` and `3452` `cmd_plan`: `oe_inferencex.plan`
- `cli.py:3734` `cmd_mcp`: `from oe_inferencex import mcp_server`
- `mcp_server.py:488`: `from .cli import class_words`

The cli.py lines above carry their old numbers. They move with their functions in S10.

**Blocks expected not to be byte-identical** after the move, because they contain an edited line. These are the
blocks the verbatim check (§6.2) may report. Anything else it reports is a defect.

- cli: cmd_from_embeddings, cmd_boundary, cmd_scale, cmd_certificate, cmd_verify, cmd_sample, class_words,
  _csv_rows, _certify_sequential, cmd_from_olmoearth, build_parser.
- estimate: sample_for_estimation, compare_from_disagreement.
- certificate: raster_digest, map_digests, _grid_level, cli_where, _reads_alone, _declared, build, _write_bundle,
  _redraw, _rules_of, _class_words_1171, _rounding_tolerance, _verify, point.
- assess: _assess.
- decide: decide.
- data: fetch_s2_window.
- stratified: pure_stratum_ratio.
- mcp_server: verify.
- demo: run.

That is 33 blocks. The plan said 32, a miscount; the step check at 88b4884 (aicr job 1543224) matched these 33.
The review fixes (796dcb1) later changed three more texts: the docstring of `cmd_scale` (already listed), the
docstring of `cmd_grade` and `to_markdown`'s line. So at 796dcb1 two blocks beyond the 33 differ from 1.18.1: `cmd_grade` and
`to_markdown` (§8).

## 4. Compatibility

### 4.1 What each old path becomes

| Old path | After 1.19.0 | The module object you get |
|---|---|---|
| `oe_inferencex.assess`, `.explain`, `.calibrate` | the alias finder (no file at the old path) | `oe_inferencex.review.assess`, `.review.explain`, `.review.calibrate` |
| `oe_inferencex.metrics`, `.signals`, `.stats` | the alias finder (no file at the old path) | `oe_inferencex.common.metrics`, `.common.signals`, `.common.stats` |
| `oe_inferencex.reliability`, `.evidence`, `.awf`, `.data`, `.figstyle` | the alias finder (no file at the old path) | `oe_inferencex.research.<same>` |
| `oe_inferencex.olmoearth`, `.embeddings`, `.lcc`, `.taskcard` | the alias finder (no file at the old path) | `oe_inferencex.readers.<same>` |
| `oe_inferencex.stratified` | the alias finder (no file at the old path) | `oe_inferencex.agency.stratified` |
| `oe_inferencex.sequential`, `.certificate` | the alias finder (no file at the old path) | `oe_inferencex.certify.sequential`, `.certify.certificate` |
| `oe_inferencex.compare`, `.scale`, `.plan`, `.decide`, `.agency` | package; the old module's code is its `__init__.py` | the package itself (same code, same globals) |
| `oe_inferencex.estimate` | package; `__init__.py` re-exports all 130 names, plus `math` and `np` | the package; each function object is the one its submodule defines |
| `oe_inferencex.cli` | package; `__init__.py` keeps the import block, re-exports all 117 moved names, and defines `build_parser` and `main` | the package; `main` is defined there (the console entry point) |
| `oe_inferencex`, `oe_inferencex.demo`, `oe_inferencex.mcp_server` | unchanged module | itself |

These 27 old paths are exactly the ones `eqv_surface.py` checks (§6.5).

### 4.2 The alias finder (as built)

This plan first proposed a shim file at each of the 18 old paths. The owner replaced them, before the migration, with
one import hook at the end of `oe_inferencex/__init__.py`. This is its text at ebd78e5, exactly (lines 130-189; the
module's top imports `importlib.abc`, `importlib.machinery`, `importlib.util`, `os` and `sys` for it):

```python
# The 1.18.1 module paths that 1.19.0 moved into feature folders (ADR-002), old name -> new path. Importing an old
# path gives the moved module itself, not a copy: `oe_inferencex.metrics is oe_inferencex.common.metrics`, so private
# names, monkeypatches and module state are the one module's. Nothing is imported before an old path is asked for
# (evidence still needs torch only then). The package's own code imports the new paths.
_MOVED = {
    "metrics": "common.metrics", "signals": "common.signals", "stats": "common.stats",
    "assess": "review.assess", "explain": "review.explain", "calibrate": "review.calibrate",
    "olmoearth": "readers.olmoearth", "embeddings": "readers.embeddings", "lcc": "readers.lcc",
    "taskcard": "readers.taskcard",
    "evidence": "research.evidence", "awf": "research.awf", "data": "research.data", "figstyle": "research.figstyle",
    "reliability": "research.reliability",
    "sequential": "certify.sequential", "certificate": "certify.certificate",
    "stratified": "agency.stratified",
}


class _MovedModules(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Finds `oe_inferencex.<old>` of _MOVED and loads it as the module it moved to."""

    def __init__(self):
        self._own = {}      # a moved module's name -> its own spec, package and loader, which exec_module puts back

    def _file(self, tail):
        """The moved module's file, located without importing anything, not even its folder: find_spec runs under
        the import system's global lock, where importing a package could deadlock against another thread."""
        folder = _MOVED[tail].rpartition(".")[0]
        where = [os.path.join(entry, *folder.split(".")) for entry in __path__]
        return importlib.machinery.PathFinder.find_spec(f"{__name__}.{_MOVED[tail]}", where)

    def find_spec(self, fullname, path=None, target=None):
        head, _, tail = fullname.rpartition(".")
        if head != __name__ or tail not in _MOVED:
            return None
        moved = self._file(tail)                         # its origin is argv[0] under -m
        return None if moved is None else importlib.util.spec_from_loader(fullname, self, origin=moved.origin)

    def create_module(self, spec):
        target = f"{__name__}.{_MOVED[spec.name.rpartition('.')[2]]}"
        module = importlib.import_module(target)    # at once, also under a lazy loader: the old path is this module
        if module.__spec__.name == target:          # its own spec, not one an unfinished old-path load left on it
            self._own[target] = module.__spec__, module.__package__, module.__loader__
        return module       # the import system now sets the old path's spec on it, and a lazy loader its loader

    def exec_module(self, module):
        """Gives the moved module back its own name, spec, package and loader, which the import system set to the
        old path's (a reload or the old `load_module` overwrites the name and package too)."""
        old = module.__spec__.name
        target = f"{__name__}.{_MOVED[old.rpartition('.')[2]]}"
        if target not in self._own:
            raise ImportError(f"{old} moved to {target}, whose own spec this finder never saw", name=old)
        module.__spec__, module.__package__, module.__loader__ = self._own[target]
        module.__name__ = target

    def get_code(self, fullname):          # so `python -m oe_inferencex.taskcard` still runs its main block
        tail = fullname.rpartition(".")[2]
        return self._file(tail).loader.get_code(f"{__name__}.{_MOVED[tail]}")


if not any(type(finder).__module__ == __name__ for finder in sys.meta_path):     # once, also after a reload
    sys.meta_path.insert(0, _MovedModules())
```

- **`_MOVED` is the one list.** 18 old names, each with its new path. tests/test_layout.py holds the same mapping and
  checks the two agree.
- **`find_spec` imports nothing.** It answers only `oe_inferencex.<old>` for a name in `_MOVED`. It locates the moved
  file with `importlib.machinery.PathFinder` over the folder under the package's `__path__`, and does not import the
  folder. The import system calls `find_spec` under its global lock, where importing a package could deadlock
  against another thread. It returns a spec named for the old path, with this finder as its loader and the moved
  file as its origin, so under `python -m` argv[0] is the moved file and argparse's prog is still `taskcard.py`. If
  the file is not there, it returns None and the import fails as for any missing module.
- **`create_module` returns the moved module itself.** It imports the new path with `importlib.import_module`, which
  reads `sys.modules` and never a stale package attribute (after a test deletes a key, evidence built against a stub
  torch cannot come back that way). The import system then puts that module in `sys.modules[old]` and sets the
  package attribute `oe_inferencex.<old>` to it. `oe_inferencex.assess` is the very object
  `oe_inferencex.review.assess`: every private name, every dunder, every imported module (`cert.hashlib`) and every
  monkeypatch target, with nothing listed by hand.
- **The moved module keeps its own name, spec, package and loader.** The import system sets the old path's spec on
  the module it gets back, and the standard library's lazy loader sets its own loader; a reload of a module that
  still carries the old path's spec, and the old `load_module`, overwrite its name and package as well. So
  `create_module` records the module's own spec, package and loader in the finder (`self._own`, keyed by the new
  path), but only when the spec is the module's own and not one an unfinished old-path load left on it.
  `exec_module` finds the new path from the old path's spec, which the import system always sets before calling it,
  and puts the name, spec, package and loader back; a module the finder never recorded is an `ImportError`. Relative
  imports and reloads then resolve where the module lives.
- **Under the standard lazy loader an old path is imported at once.** `create_module` must return the moved module
  itself, so it imports it; the lazy loader defers only `exec_module`. To import lazily, lazy-load the new path.
- **When the new module fails to import, the old path fails with the same exception and message.** Examples are
  evidence, awf, data and figstyle without their extra. The old key is then left out of `sys.modules`. A new path
  blocked with `None` in `sys.modules` is refused as missing (`ModuleNotFoundError`).
- **`get_code` serves `python -m oe_inferencex.taskcard`** (taskcard.py:27-28), the documented way to run a moved
  module. It hands over the moved file's code, so its `__main__` block runs; `sys.modules[old]` is not touched.
- **Installed once.** The finder goes in at `sys.meta_path[0]` when the package is imported, and not again after a
  reload of the package.
- **No file exists at any old path,** so there is nothing for `docs/gen_api.py` to skip, and no `__name__` guard is
  needed.

### 4.3 How private names keep resolving

- **Old paths the alias finder answers (18).** The old path is the same module object, so every name resolves, private ones included.
  Examples: `assess._pooled_argmax`, `calibrate._crossfit`, `stratified._stratum_logavg`, `cert._rules_of`,
  `sq._null_count`, `olmoearth._code_name`, `taskcard._audit_settings`, `data._catalog`, `evidence.dawid_skene`.
- **Packages holding the old code (5).** It is the same code, so `compare._graded_against`, `plan._ends` (with
  `.__wrapped__` and `.cache_clear`), `plan._hyper_pmf`, `decide._pc`, `agency._half_unit` and the rest all resolve
  as before.
- **Re-export packages (2).** `estimate/__init__.py` and `cli/__init__.py` name every old name explicitly, private
  ones included, in `from <home> import (...)` lines. Star imports are forbidden: they drop underscore names and
  dunders. Each package keeps its old module docstring (`cli.__doc__` is the parser's description). Shared state
  is re-exported by reference, never copied: `est.RULES_OF is certify.zone.RULES_OF`, and
  `est._hypergeom_interval is common.hypergeom._hypergeom_interval`, so `est._hypergeom_interval.cache_clear()` clears
  the one cache.
- **Names other modules reach through `cli` keep working,** because `cli/__init__` re-exports them:
  - `certificate`: `cli.main`;
  - `demo`: `cli.cmd_assess`;
  - `mcp_server`: `cli.main`, `class_words`, `class_names`, `class_wording`, `result_class_names`,
    `OTHER_DATES_NOTE`, `_sidecar_design`, `_outside_note`;
  - golden `changes.py`: `cli.FILL_INSTRUCTION`;
  - the tests: about 35 names.
- **Names reached through `estimate` keep working the same way:**
  - `assess` (until S3): `_condition_index`;
  - `sequential`: 11 names;
  - `plan`: 16 names;
  - `mcp_server`: 9 names;
  - golden `changes.py`: 4 constants;
  - the tests: about 60 names, 15 of them private;
  - the record: 9 private names.

### 4.4 Rules that keep the old paths honest

- **Code inside the package never imports an old path of a moved module** (R4), `oe_inferencex/__init__.py`
  included: it imports the new paths and binds the 1.18.1 module attributes from them (§3.8). The finder is for
  users, tests and the record. tests/test_layout.py checks that `import oe_inferencex, oe_inferencex.cli` imports no
  old path. Re-export packages (`estimate`, `cli`) and old-code packages (`compare`, `plan`, `decide`, `agency`,
  `scale`) are real packages and may be imported as such.
- No name defined at 1.18.1 on an old path disappears or changes kind. New names, such as submodule attributes
  (`oe_inferencex.estimate.designs`) and registrars, may appear.
- `oe_inferencex.grade` remains the function `agency.grade` after every subpackage has been imported (§6.4).

### 4.5 The `est` handles

Moved code that reads `est.X` keeps the text `est.X`. Each module gets one binding line:

| Module | Binding | Why |
|---|---|---|
| `certify/command.py` | `from oe_inferencex.certify import zone as est` | Every name it reads is certify.zone's (certify_zone, certify_by_condition, ISSUED_NOTE, ZONE_DELTA, ZONE_READING_NOTE, ZONE_READING_SEQUENTIAL). certify does not depend on the estimate package. |
| `certify/certificate.py` `_rules_of` (lazy) | `from oe_inferencex.certify import zone as est` | `est.RULES_OF.set/reset` acts on the one ContextVar. |
| `common/labels.py` | `from oe_inferencex.common import readings as est` | `_also_accept_pairs` and `also_accept_reading` live in common.readings. common does not depend on a feature. |
| `estimate/sample_command.py`, `estimate/estimate_command.py`, `estimate/command_text.py`, `certify/sequential.py`, `plan/__init__.py`, `cli/__init__.py` | `from oe_inferencex import estimate as est` | Their names span several homes; the estimate package binds them all, as estimate.py did. |

Consequence for patching (§4.6): a reader whose `est` is the estimate package sees the package's binding. A patch on
the defining module is not seen through it. This matters to no current test once §4.7 is applied. Renaming these
handles to their defining modules is a follow-up (§7.3).

### 4.6 Patch semantics after the move

A monkeypatch reaches a function only through the globals of the module the function was defined in, or through the
module behind the reader's `est` handle. After 1.19.0, patch here:

| To patch | Patch on |
|---|---|
| zone_pvalue, null_count, zone_pvalue_exact, certify_zone (for zone-internal callers) | `oe_inferencex.certify.zone` |
| _hyper_tail, _hypergeom_interval's inputs | `oe_inferencex.common.hypergeom` (then `cache_clear()`) |
| hypergeom_interval, as `_class_design_exact` and `estimate_per_class` see it | `oe_inferencex.estimate.per_class` |
| hypergeom_interval, as `estimate_error_rate`, `which_map` and `audit` see it | `oe_inferencex.estimate.error_rate`, `.which_map`, `.audit` |
| hypergeom_interval, as `agency.stratified` sees it | `oe_inferencex.common.hypergeom` (its lazy import reads it at call time) |
| _map_windows, as `cmd_estimate` sees it | `oe_inferencex.estimate.estimate_command` |
| _map_windows, as `cmd_certify` sees it | `oe_inferencex.certify.command` |
| _pooled_argmax (assess), _rules_of (certificate), MAX_WORK (stratified), _range (lcc), _get (taskcard), MAX_CONDITIONS / MAX_GRID_BYTES (olmoearth) | the old path, which is the same module object (alias) |
| `cli.main` as mcp_server calls it | `mcp_server.cli` (the cli package): unchanged |
| `oe_inferencex.__version__` | unchanged: `_stamp` and `_write_bundle` read it at call time |

### 4.7 Tests that must change (each with its change)

Six edits. The golden files under `tests/golden/` are records of earlier releases and are not edited. The first two
tests below would otherwise pass without testing anything. As built, `tests/` also gained `tests/test_layout.py`,
which checks the old paths and the alias finder (§8), and the review fixes corrected two test docstrings that named
moved files (`tests/test_agency.py`, `tests/test_sequential.py`); nothing else in `tests/` changed.

**T1, S3: `tests/test_trust_zone.py:638-646`** (`test_the_exact_interval_is_decided_alike_on_every_system`).
`_hypergeom_interval` reads `_hyper_tail` from `common/hypergeom.py`. A patch on `est` would be silent: the test
would still pass, untested. Change it to:

```python
    from fractions import Fraction
    from oe_inferencex.common import hypergeom as hg        # where _hypergeom_interval looks _hyper_tail up (ADR-002)
    assert est._hyper_tail_exact(1, 1, 40, 1, upper=True) == Fraction(1, 40)
    real = hg._hyper_tail
    seen, calls = [], []
    for nudge in (1 - 4e-16, 1 + 4e-16):
        monkeypatch.setattr(hg, "_hyper_tail", lambda *a, nudge=nudge, **kw: calls.append(1) or real(*a, **kw) * nudge)
        hg._hypergeom_interval.cache_clear()
        seen.append(est.hypergeom_interval(1, 1, 40))
    hg._hypergeom_interval.cache_clear()
    assert calls                                            # the nudged tail was used
    assert seen[0] == seen[1] == (2 / 40, 1.0)
```

**T2, S4: `tests/test_trust_zone.py:627-631`** (`test_a_p_value_equal_to_its_level_is_decided_alike_on_every_system`).
`certify_zone` reads `zone_pvalue` from `certify/zone.py`. Silent if missed. Change it to:

```python
    from oe_inferencex.certify import zone                  # where certify_zone looks zone_pvalue up (ADR-002)
    margin = np.arange(5, 0, -1, dtype=float)            # zone order 0..4
    for p in (0.29999999999999977, 0.30000000000000004):
        used = []
        monkeypatch.setattr(zone, "zone_pvalue",
                            lambda k, b, n, a, p=p: used.append((k, b, n)) or (p if (k, b, n) == (1, 3, 5) else 1.0))
        out = est.certify_zone(margin, [0, 1, 2], [1, 0, 0], 0.4, delta=0.3, rule="prefix", grid=(1.0,))
        assert (1, 3, 5) in used                            # the fake p-value was the one decided on
        assert out["levels"][0]["accepted"] is True, p
```

**T3, S4: `tests/test_bughunt_1_17_2.py:806-814`**
(`test_a_certificate_is_rerun_with_the_rules_of_the_release_that_wrote_it`). This one fails loudly if missed.
After `from oe_inferencex import estimate as est` (806), add
`from oe_inferencex.certify import zone as zone_mod` and change line 814 to:

```python
            mp.setattr(zone_mod, "null_count", lambda n, alpha: math.floor(alpha * n) + 1)
```

Line 829 (`est.null_count(n, 0.29)`, outside the patch) stays.

**T4, S5: `tests/test_class_design.py:63-65` and `84`** (`_exact_coverage`). Missing it is loud for the level=0.6
negative control; the level=None test would pass uncached. Change it to:

```python
    from oe_inferencex.estimate import per_class            # where _class_design_exact looks hypergeom_interval up (ADR-002)
    orig = per_class.hypergeom_interval
    cached = functools.lru_cache(maxsize=None)(lambda k, n, M, conf: orig(k, n, M, conf if level is None else level))
    per_class.hypergeom_interval = lambda k, n, M, conf=0.95: cached(k, n, M, conf)
    ...
    finally:
        per_class.hypergeom_interval = orig
```

The call `est._class_design_exact(...)` at 73 stays.

**T5, S10i: `tests/test_also_accept.py:683-685`** (`test_per_class_stays_strict_and_reads_the_map_once`). Loud if
missed. Change it to:

```python
    from oe_inferencex.estimate import estimate_command     # where cmd_estimate looks _map_windows up (ADR-002)
    calls = []
    real = estimate_command._map_windows
    monkeypatch.setattr(estimate_command, "_map_windows", lambda *a, **k: calls.append(a[3]) or real(*a, **k))
```

**T6, S7, completed in review (796dcb1): `tests/test_reliability.py:244-246` at 1.18.1**
(`test_evidence_pooling_crops_a_ragged_edge_and_dawid_skene_refuses_out_of_range_votes`, now lines 238-260). The test
imports evidence against a stub torch. Without a change it still passes but weakens: a cached real module could be
reused, or the stub-bound module could leak. S7 popped and deleted both `sys.modules` keys. The review found that the
stub-bound module still outlived the test, because the import also binds `evidence` as an attribute of
`oe_inferencex` and `oe_inferencex.research`. As built, the test removes both keys and both attributes before the
import (monkeypatch puts back what was there before the test), and a `try/finally` removes the stub-bound module
and the attributes the import bound when the test ends:

```python
    import importlib, sys, types
    import pytest
    import oe_inferencex.research
    keys = ("oe_inferencex.evidence", "oe_inferencex.research.evidence")
    holders = (oe_inferencex, oe_inferencex.research)     # the import binds `evidence` on both
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
    for key in keys:                         # an evidence module cached before the test comes back at teardown
        monkeypatch.delitem(sys.modules, key, raising=False)
    for holder in holders:
        monkeypatch.delattr(holder, "evidence", raising=False)
    try:
        evidence = importlib.import_module("oe_inferencex.evidence")
        assert evidence.pool_to_patches(np.ones((33, 32), bool), 16).shape == (2, 2)
        with pytest.raises(ValueError, match="votes must be classes"):
            evidence.dawid_skene(np.array([[0, -1], [1, 1]]), 2)
    finally:                                 # the module bound to the stub torch does not outlive the test
        for key in keys:
            sys.modules.pop(key, None)
        for holder in holders:
            vars(holder).pop("evidence", None)
```

### 4.8 Tests that keep working without a change, and why

| Site | Why it survives |
|---|---|
| test_bughunt_1_17_2.py:241 `A._pooled_argmax`; test_agency.py:300 `stratified.MAX_WORK`; test_bugfix_periphery.py:159 `lcc._range`; test_taskcard.py:14 `taskcard._get`; test_olmoearth.py:410, 508 | The alias finder: the old path is the real module. `_assess`, `_stratum_logavg`, `LCCTile` methods, `_variants` and `condition_from_inputs` read these names from that module's globals. |
| test_bughunt_1_17_2.py:833, 1152-1153 `cert._rules_of` | The alias finder. `_verify` and `_rules_of` stay in one module. |
| test_bughunt_1_17_2.py:815, 1089, 1151 `oe_inferencex.__version__` | `_stamp` and `_write_bundle` keep their call-time imports. |
| test_bughunt_1_17_2.py:846-857 `cert._rules_of` + `est.null_count` + `cli.class_words` | One `RULES_OF` object, re-exported by reference (R8). |
| test_mcp_server.py:270 `mcp_server.cli.main` | mcp_server is unchanged and calls `cli.main` as an attribute. The cli package defines `main`, and there is no `cli/main.py` submodule. |
| test_mcp_server.py:296-308 (`sys.modules` re-import of `oe_inferencex.mcp_server`) | mcp_server stays the implementing module at that path. `cmd_mcp`'s import stays lazy. |
| test_mcp_server.py:924 `python -m oe_inferencex.cli mcp` | `cli/__main__.py`. |
| test_agent_borrowings.py:37 (reads the file `oe_inferencex/mcp_server.py`) | mcp_server.py stays a real file. |
| test_package.py:12, 16-18, 21, 30, 38, 46 | Versions match; every `__all__` name exists; the nine numpy-only old paths import (through the finder, or packages); the evidence refusal comes through the finder, with the moved module's message; taskcard old path. |
| test_trust_zone.py:644, 646 `est._hypergeom_interval.cache_clear()` | The same lru_cache object (T1 uses `hg.` for clarity). |
| tests/golden/condition_1_3_1/changes.py: 7 `(module path, constant)` pairs, read by test_cli.py:874-908 and test_assess.py:357 | `oe_inferencex.estimate` re-exports PREFIX_NOTE, TILES_WARNING, ISSUED_NOTE, PER_CLASS_EXACT_NOTE. `oe_inferencex.assess` is answered by the alias finder. `oe_inferencex.cli` re-exports FILL_INSTRUCTION. |
| every `from oe_inferencex.cli import main` / `cli.main(...)` (about 25 files), `cli.cmd_assess` / `cli.cmd_sample` with hand-built Namespaces, `cli.build_parser()._actions` | Re-exports. `_condition_args` and `cmd_sample` keep their `getattr` reads of optional attributes. |
| all reads of private names through `est`, `sq`, `plan`, `cert`, `decide_mod`, `assess`, `stratified`, `agency`, `compare`, `olmoearth`, `taskcard`, `calibrate`, `demo` | §4.3. |

### 4.9 Outside the tests

- **`pyproject.toml`.** `oe-inferencex = "oe_inferencex.cli:main"` is unchanged. `packages.find include =
  ["oe_inferencex*"]` picks up every new folder, and each has an `__init__.py`. `package-data` `sample/*` is
  unchanged. Only `version` changes (S14). The comments naming `oe_inferencex.taskcard`, `.awf`, `.data`,
  `.embeddings` and `.mcp_server` stay valid as import paths.
- **Docs.**
  - `docs/gen_api.py` walks every `.py` file. It is unchanged: no file exists at an old path, so there is nothing to
    skip. `cli/__main__.py` is already skipped (it starts with `_`).
  - The API URLs of the 18 moved modules move: `reference/assess/` becomes `reference/review/assess/`. `mkdocs.yml`
    has 18 `redirect_maps` entries, one per moved module, each with an absolute ReadTheDocs URL as its target (for
    example `reference/assess.md: https://olmoearth-inferencex.readthedocs.io/en/latest/reference/review/assess/`), so
    they do not depend on the order in which gen-files writes the pages and the redirects plugin reads them.
  - Same-named packages keep their URL (`reference/estimate/` is now the package index).
  - `docs/Usage.md` "## Modules" (1616-1636) is rewritten as a table of folders. Its code examples (1449-1598) keep
    working through the old paths and stay as they are.
  - README has no module paths, so it is unchanged.
- **`CHANGELOG.md`.** Add a 1.19.0 entry: layout only; old import paths keep working; no behaviour change; the six
  test edits; ADR-002. As built, the entry also states the one output line that names a moved file, the limits of
  the alias finder (§8), `tests/test_layout.py`, and the release check (aicr job 1543225).
- **`oe_inferencex/__init__.py` docstring.** It lists folders instead of modules and cites ADR-002 instead of the
  flat-layout sentence (§3.8).
- **Research record.** It is pinned to 1.12.1 (record `pyproject.toml:10,15`), so nothing changes there now. The
  pin-bump items are in §7.1 (R15).

## 5. Import cycles and the order of the migration

### 5.1 The import graph after the move

**Eager edges (module top), in layers.** No edge points to a higher layer, so the graph is acyclic.

| Layer | Module | Eager package imports |
|---|---|---|
| 0 | common/hypergeom, common/sampling, common/readings, common/metrics, common/signals, common/raster, common/output, common/wording, review/explain, research/reliability, research/awf, research/figstyle, readers/lcc, readers/taskcard, readers/embeddings, decide (package) | none |
| 0 | common/stats | common/metrics, common/signals |
| 0 | common/maps, common/product, common/condition | common/raster |
| 0 | common/labels | common/readings (as `est`) |
| 0 | research/evidence | common/metrics, research/reliability (after torch) |
| 0 | research/data | none (third-party only) |
| 1 | review/assess | common/sampling, common/metrics, common/signals |
| 1 | review/windows | review/assess, common/raster, common/maps, common/product, common/labels, common/output |
| 1 | compare (package) | review/explain, common/stats |
| 1 | review/calibrate | compare, common/metrics |
| 1 | readers/olmoearth | review/assess |
| 1 | scale (package) | review/assess |
| 2 | certify/zone | common/hypergeom, common/sampling |
| 2 | estimate/designs | common/sampling |
| 2 | estimate/intervals | estimate/designs, common/hypergeom |
| 2 | estimate/error_rate | estimate/intervals, estimate/designs, common/hypergeom, common/sampling |
| 2 | estimate/model_assisted | estimate/intervals, estimate/designs |
| 2 | estimate/per_class | estimate/intervals, common/hypergeom, common/sampling |
| 2 | estimate/which_map | common/hypergeom |
| 2 | estimate/audit | estimate/designs, common/hypergeom, common/readings |
| 2 | estimate (package `__init__`) | all of the above, certify/zone, common/{hypergeom, sampling, readings} |
| 2 | agency/stratified | none (numpy) |
| 2 | agency (package) | agency/stratified |
| 3 | certify/sequential | estimate (package), common/sampling |
| 3 | certify/certificate | none at top (every package import is lazy) |
| 3 | plan (package) | estimate (package) |
| 4 | review/command | review/assess, review/explain, common/{raster, maps, product, condition, output} |
| 4 | compare/command | compare, review/assess, review/windows, common/{raster, output} |
| 4 | scale/command | review/assess, common/{raster, maps, product, output} |
| 4 | agency/command | common/output |
| 4 | readers/command | common/output |
| 4 | certify/command | certify/zone, review/windows, common/{labels, output, wording} |
| 4 | estimate/command_text | estimate |
| 4 | estimate/sample_command | estimate, estimate/command_text, review/assess, review/windows, common/{raster, maps, product, condition, output, labels, wording} |
| 4 | estimate/estimate_command | estimate, estimate/command_text, review/windows, common/{product, output, labels, wording} |
| 4 | decide/command | common/{output, wording} |
| 4 | plan/command | review/assess, review/windows, common/{raster, maps, product, output} |
| 4 | cli/demo_command, cli/mcp_command | none |
| 5 | cli (package `__init__`) | every layer-4 module, review/assess, review/explain, estimate, compare, common/* |
| 5 | cli/__main__ | cli |
| 6 | demo | review/assess, common/metrics |
| 6 | mcp_server | oe_inferencex (`__version__`), cli, review/assess, estimate, decide |
| 6 | oe_inferencex (`__init__`) | the new paths: review/assess, agency, review/calibrate, estimate, compare, review/explain, common/metrics, common/signals, agency/stratified, common/stats (never an old path, §3.8); then it installs the alias finder |

Every import of any `oe_inferencex.X` runs the root `__init__` first. That fully imports `estimate`, and with it
`certify.zone`, before anything else. During the root `__init__` itself, nothing imported reaches
`certify.sequential`, `plan` or `cli`.

**Lazy edges.** These are function-level imports and must stay lazy. They are the only upward or sideways edges.

| Site (new home) | Edge | Why lazy | Today |
|---|---|---|---|
| common/wording `class_words` | → certify/zone `RULES_OF`; → certify/certificate `_class_words_1171` | common must not depend on certify at import time; certificate imports wording back (next row) | cli ↔ certificate, lazy |
| certify/certificate `_class_words_1171` | → common/wording `_CLASS_WORDS`, `_PATH_KEYS` | as above | same |
| common/labels `_csv_rows` | → certify/certificate `csv_rows` | layering | cli → certificate, lazy |
| certify/certificate `_redraw`, `_verify` | → cli (`main`) | verify re-runs `sample` and `certify` in process; cli imports certify/command | certificate ↔ cli, lazy |
| certify/certificate (13 other sites) | → common/raster, common/output, common/labels, certify/zone, certify/sequential, compare, oe_inferencex `__version__` | unchanged laziness (the version must be read at call time) | lazy |
| cli/mcp_command `cmd_mcp` | → mcp_server | mcp_server imports cli at module top | same |
| cli/demo_command `cmd_demo` | → demo | demo.run imports cli lazily too; numpy-only either way | same |
| demo `run` | → cli, review/assess | as above | same |
| decide `decide` | → common/wording | was cli; now no cycle at all | decide ↔ cli, lazy |
| decide/command `cmd_decide`, estimate/estimate_command `_estimate_disagreement` | → decide | unchanged | lazy |
| plan/command `cmd_plan`, `_ladder_text` | → plan | unchanged | lazy |
| agency/command `cmd_grade` | → agency | unchanged | lazy |
| scale/command `cmd_scale` | → scale | unchanged | lazy |
| compare/command `cmd_compare` | → compare `dates_reading`, `over_groups` | unchanged | lazy |
| readers/command | → readers/embeddings, readers/olmoearth | extras: zarr, rasterio (R6) | lazy |
| certify/command `cmd_certificate`, `cmd_verify`, `_certify_sequential`; estimate/sample_command `cmd_sample` | → certify/certificate, certify/sequential | unchanged | lazy |
| review/command `cmd_boundary` | → review/assess `boundary_band` | unchanged | lazy |
| review/assess `_assess` | → review/explain `confusion_pairs` | unchanged | lazy |
| estimate/designs `sample_for_estimation` | → common/sampling `sequential_order` | was `.sequential`; now points down a layer | estimate ↔ sequential, lazy |
| estimate/which_map `compare_from_disagreement` | → common/stats | unchanged | lazy |
| agency/stratified `pure_stratum_ratio` | → common/hypergeom | was estimate | lazy |
| research/data `fetch_s2_window` | → common/raster `affine_apply` | was cli | lazy |
| common/output `_stamp` | → oe_inferencex `__version__` | tests patch the version | lazy |
| mcp_server `_class_text`, tool `verify` | → cli, certify/certificate | unchanged | lazy |

**Cycles removed:**

- `estimate` ↔ `sequential`: `sequential_order` moved to common/sampling.
- `decide` ↔ `cli`: decide now imports common/wording.
- The data → cli edge: `affine_apply` now comes from common/raster.

**Cycles that remain, all lazy on at least one side:**

- `certificate` ↔ `cli`, through `main`. This one is inherent.
- `common/wording` ↔ `certify/certificate`. Moving `RULES_OF` and `_class_words_1171` into common would remove it;
  that is a follow-up.
- `mcp_server` ↔ `cli`.
- `demo` ↔ `cli`.

**Package-level cycles.** At the folder level, `review` ↔ `compare` (calibrate → compare → explain) and `estimate` ↔
`certify` (estimate → certify.zone; certify.sequential → estimate) are cycles. They are harmless because the
`__init__`s of `review` and `certify` import nothing (R1), and the module-level graph above is acyclic.

**Rules that keep it acyclic.** Check them at every step:

- **R1.** `common/`, `review/`, `certify/`, `readers/` and `research/` have an `__init__.py` holding a docstring only.
- **R2.** `estimate/__init__.py` imports only estimate's logic modules, `certify.zone` and `common.{hypergeom,
  sampling, readings}`. It never imports a `*_command` module, nor anything that imports `cli`.
- **R3.** `certify/zone.py` and every `common/` module import nothing from a feature folder at module top.
  `certify/zone.py` never imports `oe_inferencex.estimate`: estimate's `__init__` imports zone, so importing the
  estimate package from zone would be a partially-initialised import.
- **R4.** No package module imports an old path of a moved module, `oe_inferencex/__init__.py` included. The alias
  finder answers old paths for users, tests and the record only.
- **R5.** Every lazy import in the table above stays inside its function. `cmd_mcp`'s import of mcp_server is the one
  that would fail first.
- **R6.** Nothing that `cli` imports at module top imports an extra (rasterio, zarr, torch, matplotlib, mcp,
  pydantic), and `readers/__init__` and `research/__init__` import none of their modules. `import oe_inferencex.cli`
  stays numpy-only.
- **R7.** No feature module imports `oe_inferencex.cli` at module top. Feature command modules never import `cli`
  at all; they take shared helpers from `common/` and `review/windows`. The one allowed importer at module top is
  `mcp_server`; `certificate` and `demo` import `cli` only lazily.
- **R8.** These are defined once and re-exported by reference only:
  - `RULES_OF` (certify/zone);
  - `_hypergeom_interval`'s lru_cache (common/hypergeom);
  - `_SIGN_CRIT` (estimate/which_map);
  - `plan._TABLES` and `plan._ends` (plan);
  - `certificate._RUN` (certify/certificate);
  - `mcp_server._CAPTURE` (mcp_server);
  - `taskcard._LISTING_UNREACHABLE` (readers/taskcard);
  - `agency._OTHER` (agency);
  - `stratified.TooLarge`, which is caught by class (agency/stratified).

  Once a folder exists, the old file is deleted. A leftover `estimate.py` next to `estimate/` would be shadowed
  rather than imported, but it would hold a second `RULES_OF` for anyone who reads the source, and docs/gen_api.py
  would render it.

Side effect of R2 and R6: `import oe_inferencex.cli` now also executes the `plan`, `decide` and `scale` package
bodies, which cli imported lazily before. They are numpy- and stdlib-only and have no side effects beyond defining
names, so the only cost is import time (§7.1, R6).

### 5.2 Steps

Each step is one commit and must pass the step job (§6.1). Any gate failure stops the migration until it is
explained. Every step's gate:

1. pyflakes reports no undefined name beyond S0's.
2. Every import statement resolves.
3. The suite is green, with the same numbers of passed and skipped tests as S0, plus the tests of the new
   tests/test_layout.py.
4. The surface check (`eqv_surface.py check`) reports 0 LOST.
5. The verbatim check reports nothing outside the expected list for the steps done so far.

| Step | What moves | Edits (§3.9) | Test changes (§4.7) | Extra checks |
|---|---|---|---|---|
| **S0** baseline | nothing (`987dc7a`) | none | none | Step job with `MODE=baseline`. It runs the suite and `eqv_surface.py dump` into `~/oeix-eqv/surface_before.json`, and records the passed and skipped counts. |
| **S1** common leaves | `common/__init__.py` (new); `git mv` metrics.py, signals.py and stats.py into `common/`; the alias finder in `__init__.py` (`_MOVED`, `_MovedModules`) answers the three old paths, in place of shim files; the root `__init__` imports their new paths (each later step repoints its own) | S1 rows | new `tests/test_layout.py` (old paths, the finder) | none |
| **S2** review | `review/__init__.py` (new); `git mv` assess.py, explain.py and calibrate.py into `review/`; three `_MOVED` entries | S2 rows | none | none |
| **S3** common math out of estimate.py | Create `common/hypergeom.py` (14 names), `common/sampling.py` (6 names, plus `sequential_order` cut from sequential.py) and `common/readings.py` (3 names). estimate.py re-imports all 23 names at its top so its namespace is unchanged. | S3 rows | **T1** | none |
| **S4** certify/zone | `certify/__init__.py` (new). Cut the 32 zone names from estimate.py into `certify/zone.py`; estimate.py re-imports them. | none (new imports only) | **T2, T3** | none |
| **S5** estimate package | `git mv estimate.py estimate/__init__.py`, then cut its definitions into designs, intervals, error_rate, model_assisted, per_class, which_map and audit. `__init__` keeps the docstring, `math`, `np` and the re-export of all 130 names (§3.3.1 order). | none | **T4** | Optional: run the battery (§6.3). |
| **S6** cli plumbing | Cut the 54 shared cli.py names into `common/{raster, output, maps, product, condition, labels, wording}.py` and the two window readers into `review/windows.py`. cli.py imports all of them back, so its namespace is unchanged. | S6 rows | none: `cmd_estimate` is still in cli.py, so test_also_accept:685 still intercepts | none |
| **S7** readers, research | `readers/__init__.py` and `research/__init__.py` (docstring only). `git mv` olmoearth, embeddings, lcc and taskcard into `readers/`; evidence, awf, data, figstyle and reliability into `research/`. Nine `_MOVED` entries; taskcard's old path also runs under `python -m` (the finder's `get_code`, §4.2). | S7 rows | **T6** | `python -m oe_inferencex.taskcard --help` output identical to S0 (in the battery) |
| **S8** certify | `git mv` sequential.py and certificate.py into `certify/`; two `_MOVED` entries | S8 rows | none | none |
| **S9** same-named packages | `git mv` compare.py → `compare/__init__.py`, scale.py → `scale/__init__.py`, plan.py → `plan/__init__.py`, decide.py → `decide/__init__.py`, agency.py → `agency/__init__.py`, stratified.py → `agency/stratified.py` (plus a `_MOVED` entry). May be split into one commit per package. | S9 rows | none | none |
| **S10a-i** command modules, one commit each, while cli.py is still a module | **a** `review/command.py`; **b** `readers/command.py`; **c** `scale/command.py`; **d** `agency/command.py`; **e** `compare/command.py`; **f** `decide/command.py`; **g** `plan/command.py`; **h** `certify/command.py`; **i** `estimate/{command_text, sample_command, estimate_command}.py`. Each step moves the command functions, helpers and constants of §3.1 plus the registrar(s) of §3.2. cli.py imports them back, and in `build_parser` each moved block is replaced by its registrar call at the same position. | S10c row; the `est` bindings of §4.5 | **T5** in S10i | none |
| **S10j** cli package | `git mv cli.py cli/__init__.py`. Create `cli/__main__.py` (§3.2) and `cli/demo_command.py` and `cli/mcp_command.py` (cmd_demo, cmd_mcp and their registrars). `cli/__init__` is left with the docstring, import block, re-exports, `build_parser` and `main`; the old `__main__` guard goes. | none | none | test_mcp_server.py:924 (`-m oe_inferencex.cli`) runs in the suite |
| **S11** MCP (no move) | nothing | none: mcp_server.py's lines 59 and 1728 were changed in S2 and S8 | none | The battery's MCP sections (texts and replies) must be identical. |
| **S12** docs and metadata | docs/gen_api.py unchanged (no shim files to skip); mkdocs.yml redirects (18, absolute targets); Usage.md module table; CHANGELOG entry (heading "1.19.0 (unreleased)"); `__init__.py` docstring | none | none | `mkdocs build --strict` passes (part of the battery, §6.3 step 5) |
| **S13** equivalence | nothing | none | none | **The full battery (§6.3) on the S12 commit, with the version still 1.18.1.** Plus `api_identity.py` (§6.4). All acceptance criteria must hold. |
| **S14** release | `pyproject.toml` version 1.19.0; `CITATION.cff` version and date-released; `__init__.py` fallback "1.19.0"; `uv.lock` line 1239 version "1.19.0"; CHANGELOG date | none | none | Step job (suite). Then the owner merges to `main`, GitHub CI (tests 3.11/3.13, wheel, docs) must be green, and the owner releases. |

Commit messages name the step and the source file of each move, so `git log -C --follow` and `git blame -C -C` can
find the history. No file is left at an old path, so none stands in the way of plain rename detection.

## 6. Verification plan

Everything below runs on aicr. Copy the scripts once, without `scp` (it triggers a Duo push):

```bash
ssh aicr 'mkdir -p ~/oeix-eqv'
for f in step_check.sh resolve_imports.py verbatim_check.py api_identity.py eqv_surface.py eqv_driver.py eqv_job.sh; do
  ssh aicr "cat > ~/oeix-eqv/$f" < "$f"
done
```

Sources:

- `eqv_driver.py`, `eqv_surface.py` and `eqv_job.sh` are in
  `/private/tmp/claude-501/-Users-lucas-Desktop-Github-olmoearth-inferenceX/08a8fdd9-9831-4e63-bef7-8b20afa649c5/scratchpad/featurefolders/`.
- The other four scripts are in Appendices A-D. Save each verbatim under the name given.

If the scratchpad is gone, rebuild `eqv_*` from §6.3 and §6.5.

Before each submission, state its cost. Step job: 8 CPUs, 16 GB, at most 1 h (about 10 min expected), about 3 GB
on `/scratch` (deleted on success), only the Slurm log on `/home`, one `ssh` to submit and one to read the log, no
polling. Battery job: 8 CPUs, 24 GB, at most 3 h, about 10 GB on `/scratch` (deleted after review), same
connections. Run at most one migration job at a time.

### 6.1 The step job (every step)

Each step is pushed to the throwaway branch `cluster-suite`, where CI does not run, and then submitted:

```bash
git push origin HEAD:cluster-suite --force
ssh aicr "sbatch --parsable --export=ALL,SHA=$(git rev-parse HEAD),MODE=step ~/oeix-eqv/step_check.sh"
# about 15 minutes later, once:
ssh aicr 'tail -40 ~/olmoearth_inferenceX/slurm/oeix-step-<JOBID>.out'
```

Checks, in order (Appendix A):

1. **pyflakes 3.2.0 reports no `undefined name` that the S0 baseline does not have.** This catches a from-import
   missing from a split module, which would otherwise surface only as a NameError on a rare path. `imported but unused` lines are listed but do not
   gate.
2. **`resolve_imports.py` (Appendix B).** Every import statement in the package resolves, at module level or inside
   a function. Relative imports are resolved against their file's package; modules that need the encoder extra are
   skipped by their message. This catches broken lazy imports, which fail only at run time on their path.
3. **The suite.** `pytest -q` with the `geo`, `embeddings` and `mcp` extras, as CI runs it. The passed and skipped
   counts must equal S0's, plus the tests of tests/test_layout.py.
4. **`eqv_surface.py check ~/oeix-eqv/surface_before.json`.** Every name of the 27 old paths keeps its fingerprint,
   `__all__` is unchanged with the same kind of object per name, the entry point is unchanged, and
   `python -m oe_inferencex.cli --help` gives the same output. 0 LOST.
5. **`verbatim_check.py` (Appendix C) against a worktree of `987dc7a`.** Every top-level block of the 28 old files
   appears byte-identical somewhere in the new tree, apart from the expected blocks of §3.9 for the steps done so
   far. Every old module docstring appears as a docstring in the new tree (the root `__init__`'s is excluded).

### 6.2 Reviewing the diff

Besides check 5, the reviewer reads each step with
`git diff -M -C --color-moved=dimmed-zebra --color-moved-ws=allow-indentation-change HEAD~1`. Lines that are not
moved must be one of:

- the import edits of §3.9;
- the `est` bindings of §4.5;
- new module docstrings;
- the alias finder in `__init__.py`, its `_MOVED` entries and the root `__init__`'s repointed imports;
- the registrar `def` lines;
- `cli/__init__`'s registrar calls;
- the six test edits, and `tests/test_layout.py`.

### 6.3 The behavioural-equivalence battery (S13; optionally after S5 and S10j)

Set `AFTER=` in the cluster copy of `eqv_job.sh` to the full SHA of the S12 commit, which has been pushed to
`cluster-suite`. Then submit it:

```bash
ssh aicr "sed -i 's/^AFTER=.*/AFTER=$(git rev-parse HEAD)/' ~/oeix-eqv/eqv_job.sh && sbatch --parsable ~/oeix-eqv/eqv_job.sh"
```

**What the job does:**

- It makes fresh clones of BEFORE (`987dc7ac7bc1a5548b48ebe92f0b8c52bcadd1ed`) and AFTER under
  `/scratch/qi_zim_neu/olmoearth_inferenceX/eqv/$SLURM_JOB_ID`. Each gets
  `uv sync --frozen --python 3.12 --extra geo --extra embeddings --extra mcp`, and the two `uv.lock` files are
  compared with `cmp`.
- The environment is `COLUMNS=100 PYTHONHASHSEED=0 TZ=UTC MPLBACKEND=Agg`.
- `eqv_driver.py FIXTURES WORK OUT` runs three times: **before**, **before2** (the control) and **after**. WORK is
  the same absolute path each time. FIXTURES is always the BEFORE clone, so both sides read the same fixture code.
- Every command goes through `oe_inferencex.cli.main`, the old path. The driver records stdout, stderr, the exit
  code or refusal text, and any exception (type and message), and it copies WORK.
- `oe_inferencex.__version__` is forced to `"1.18.1"` before anything else is imported. That makes `written_by` and
  the zone.json digests inside certificate.json identical on both sides. On the pre-bump commit it is 1.18.1 anyway.
- The only normalisation is certificate.json's `"created"`, which becomes `<CREATED>`.

**The commands.** `qs` is the work directory for the quick-start map. `scores.tif`, `other.tif` and `truth.tif` come
from `examples/quickstart_map.py write_maps()`, and labels are filled by its `label()`.

**A. Golden 1.3.1 battery.** This is `tests/golden/condition_1_3_1/generate.py produce(WORK/golden_1_3_1)`:

- `assess` on the Dynamic World tile, on the 128×128 scene (`--reference --budgets 0.05 0.10 --order
  boundary_first`) and on the 3-class logits (`--logits --reference`).
- `sample` on dw with the confidence, proportional, random and tiles designs (budgets 200, 200, 300 and 160; `--tile
  8 --per-tile 16`). Confidence and random on the scene (`--seed 3`, `--seed 5`). Random with `--logits` on the
  logits (`--seed 2`).
- On each sample: `estimate`, `estimate --per-class`, and `certify` with prefix 0.12, bonferroni 0.15, whole 0.3,
  none 0.02, and `--delta 0.2` at 0.1. Prefix certify also runs on the scene and the logits.
- The six `assess_prediction` / `assess_classmap` summaries.

**B. Quick-start battery** (cwd `qs`):

- **help:** `oe-inferencex --help`, and `<cmd> --help` for all 16 commands.
- **assess:**
  - `assess scores.tif --out a_plain`
  - `assess scores.tif --reference truth.tif --order boundary_first --budgets 0.02 0.1 --out a_ref`
  - `assess scores.tif --patch 8 --out a_p8`
  - `assess scores.tif --condition band.tif --out a_cond`
- **boundary:**
  - `boundary scores.tif --width-px 2 --out band.tif`
  - `boundary scores.tif --width-m 20 --out band_m.tif`
- **scale:**
  - `scale scores.tif --reference truth.tif --out scale.json`
  - `scale scores.tif --patches 2 4 8`
- **compare:**
  - `compare scores.tif other.tif --out cmp --labels truth.tif`
  - the same with `--date-a 2024-06-01 --date-b 2024-07-01 --labels-date 2024-06-15`
- **sample** (`--out s_<k>.csv` each):
  - `--budget 200`
  - `--design proportional`
  - `--budget 400 --design random`
  - `--budget 160 --design tiles --tile 8 --per-tile 16`
  - `--budget 200 --condition band.tif`
  - `--budget 1200 --design class`
  - `--design audit --per-class 20`
  - `--budget 200 --design sequential --alpha 0.1`, then `--budget 300 --extend s_seq.csv --out s_seq.csv`
  - `--budget 315 --design random --target 0.5 --seed-from-map --alpha 0.05`
  - `--budget 200 --other other.tif`
  - two refusals: no budget, and an unlabelled sample
- **estimate:**
  - every sample;
  - `--per-class` on the conf, prop, rand, class and cond samples;
  - `--per-class --classes 4`;
  - `--reviewer-false-alarm 0.01 --reviewer-miss 0.05`;
  - `--also-accept accept.json` (where accept.json is `{"1":[2]}`), plain, with `--per-class`, and on the audit;
  - the audit with `--alpha 0.1`.
- **certify:**
  - on s_rand: `--alpha 0.05`, `--rule bonferroni`, `--level-cut ramp`, `--alpha 0.1 --delta 0.2`, and
    `--also-accept accept.json`;
  - target (alpha from the sidecar);
  - seq `--alpha 0.1`;
  - class `--alpha 0.2`;
  - cond `--alpha 0.1`;
  - refusals: a confidence sample, and no alpha.

**C. Certificates:**

- New bundles:
  - `certificate s_target_zone.json --date 2024-06-01 --out cert_target`
  - class: `sample scores.tif --design class --budget 1200 --seed-from-map --alpha 0.2 --out sc_class.csv`, label it,
    `certify sc_class.csv`, `certificate sc_class_zone.json --date 2024-06-01 --out cert_class`
  - sequential (`--budget 300 --seed-from-map --alpha 0.1`, date `2024-06-01/2024-08-31`)
  - named classes (`--condition-names 0=input 1=condition "2=good condition" 3=x`)
  - .npy (`sample map.npy --design random --budget 300 --seed-from-map --alpha 0.1` → certify → certificate →
    `verify cert_npy/certificate.json --scores map.npy --at-pixel 2 2`)
- Each new bundle is checked three ways: `verify cert_<k>/certificate.json --scores scores.tif --out v_<k>.json`; the
  same without `--scores`; and `--at 500020 4999980`.
- The 1.17.1 golden bundles (`tests/golden/certificates_1_17_1/{random,class,sequential,class_named}`):
  `verify <kind>/certificate.json --scores scores.tif --out vg_<kind>.json`. These verifies exercise `_rules_of`,
  `RULES_OF`, `null_count` and `class_words` together.

**D. The remaining commands:**

- **decide:**
  - `decide s_rand_estimate.json --ask error_rate_below=0.1`
  - `decide s_rand_perclass.json --ask user_accuracy_above=0.85 --ask producer_accuracy_above=0.85`
  - `decide z_rand_prefix.json --ask trusted_share --ask trusted_share_at_least=0.5`
  - `decide s_pairs_estimate.json --ask more_accurate`
  - `decide cmp/comparison.json --ask share_differs`
  - `decide s_rand_estimate.json` (lists the questions)
  - `decide sc_class_zone.json --ask trusted_share`
- **plan:**
  - `plan scores.tif --width 0.1`
  - `plan --windows 4096 --width 0.05 --error-rate 0.2`
  - `plan scores.tif --other other.tif --difference 0.02`
  - `plan --windows 4096 --differing 500 --difference 0.05 --two-class`
  - `plan scores.tif --coverage 0.5 --alpha 0.05 --zone-error 0.02`
  - `plan --windows 4096 --coverage 0.8 --alpha 0.1`
  - `plan --windows 4096 --class-share 0.1 --classes 4 --drawn-at 0.5`
  - the refusal `plan scores.tif`
- **grade** (the Olofsson points from tests/test_agency.py, weights to six decimals):
  - `grade points.csv --stratum stratum --reference ref --map map --map other --weight weight --pure 1=1 2=2 3=3 4=4`
  - `--strata sizes.csv`
  - `--alternate other --weight weight`
  - `--strata sizes.csv --sizes area`
  - a refusal
- **from-olmoearth** (fixtures from tests/test_olmoearth.py):
  - `from-olmoearth ds_in --out fo_in --conditions` and its printed `next:` command
  - `from-olmoearth ds_cls --out fo_cls` and its `next:` command
  - the refusal `--conditions` on a classification
- **from-embeddings** (the local zarr from tests/test_embeddings.py):
  - `from-embeddings --store emb.zarr --points points.csv --bbox <_bbox()> --year 2025 --out emb_scores.tif`, then
    `assess emb_scores.tif`
  - the refusal `--year 2023`
- **demo:** `demo --out demo_real`; `demo --made-up --seed 3 --out demo_made_up`.
- **MCP, in process:**
  - dump `__doc__`, `__version__`, INSTRUCTIONS, CARDS, TITLES, QUESTIONS, HARD_RULES, MCP_SCOPE, BUDGET_300, the
    tool signatures and the Field descriptions;
  - call guide, assess, compare, estimate (plain and per_class), certify, decide, plan, verify and sample on the
    same files;
  - also run `python -m oe_inferencex.taskcard --help`.

**E. Surface, suites, the bare wheel and docs:**

- `eqv_surface.py dump` at BEFORE and `check` at AFTER.
- `pytest -q` in both clones.
- `uv build`, then a numpy-only venv. In it: import the core old paths; `import oe_inferencex.evidence` must refuse
  naming `olmoearth-inferencex[encoder]`; `python -m oe_inferencex.cli --help` and `oe-inferencex demo` are compared
  before and after.
- `mkdocs build --strict` at AFTER.

**Acceptance (all must hold):**

1. The driver exits 0 for before, before2 and after.
2. **The control diff `diff -r out-before out-before2` is empty.** Otherwise the battery is not deterministic and
   says nothing about the move.
3. **`diff -r out-before out-after` is empty.** Two differences are path artefacts rather than behaviour, and are
   accepted only after the following check:
   - a line that differs only by the clone directory (`…/src-before/…` against `…/src-after/…`);
   - a Python warning's `file:line` prefix, which names the moved file.

   For these, rerun the diff with the clone directory replaced by `<SRC>` and each warning's location replaced by
   `<WHERE>`. Every remaining line must be identical, and every normalised line is listed in the merge note.
4. `uv.lock`s identical (`cmp`).
5. The surface check reports 0 LOST.
6. Both suites are green, and AFTER's passed and skipped counts equal S0's plus the tests of tests/test_layout.py
   (as built: 1,296 and 1,319, the 23 new ones in test_layout.py; job 1543225).
7. Bare wheel:
   - "core imports ok";
   - the evidence refusal names the extra;
   - the `--help` diff is empty;
   - the demo output tree diff is empty;
   - the demo stdout diff is empty, with the out directory normalised.
8. The docs build exits 0.
9. `api_identity.py` (§6.4) exits 0.

### 6.4 Public API identity (S13, and in the step job from S9 on)

`api_identity.py` (Appendix D) runs on the AFTER env against a worktree of BEFORE.

1. It imports every subpackage of the new tree, skipping the ones that need the encoder extra.
2. It checks `list(oe_inferencex.__all__)` equals 1.18.1's, element by element and in order. That also implies
   sorted equality.
3. For each of the 84 names except `__version__`, `getattr(oe_inferencex, n) is getattr(<its 1.18.1 home>, n)`. The
   home is read from BEFORE's `__init__.py` import lines: for example `oe_inferencex.assess` for `assess_prediction`,
   `oe_inferencex.agency` for `grade`. Each object's `__module__` must start with `oe_inferencex.`.
4. `callable(oe_inferencex.grade)` holds after all subpackages are imported.
5. The state objects are one each:
   - `estimate.RULES_OF is certify.zone.RULES_OF`
   - `estimate._hypergeom_interval is common.hypergeom._hypergeom_interval`
   - `estimate._SIGN_CRIT is estimate.which_map._SIGN_CRIT`
   - `certificate is certify.certificate`
   - `sequential is certify.sequential`
   - `stratified is agency.stratified`
   - `assess is review.assess`
   - `taskcard is readers.taskcard`
   - `cli.est is estimate`
6. `cli.main.__module__ == "oe_inferencex.cli"`, and `importlib.metadata` resolves the console entry point to that
   same function.

### 6.5 Old-path import check

- **`eqv_surface.py check`** covers the 27 old paths: agency, assess, awf, calibrate, certificate, cli, compare,
  data, decide, demo, embeddings, estimate, evidence, explain, figstyle, lcc, mcp_server, metrics, olmoearth, plan,
  reliability, scale, sequential, signals, stats, stratified and taskcard.
  - Every name in each path's `dir()` at 1.18.1 must still be there with the same fingerprint: a callable's
    signature, a constant's repr, a class's qualname.
  - That covers every private name, `__version__`, `cert.hashlib`, `mcp_server.cli` and `mcp_server.Field`.
  - `mcp_server.__doc__` and `cli.__doc__` must be unchanged.
  - The 1.18.1 import errors without extras must be the same exception with the same message.
- **The bare wheel** (numpy, pyyaml and huggingface_hub only) checks two things:
  - These 23 old paths import: assess, explain, calibrate, compare, estimate, metrics, stats, signals, reliability,
    cli, certificate, sequential, plan, decide, scale, agency, stratified, demo, taskcard, lcc, olmoearth,
    embeddings, mcp_server.
  - These 4 refuse with their 1.18.1 messages: evidence, awf, data, figstyle.
- **Entry points:** `python -m oe_inferencex.cli --help` and `python -m oe_inferencex.taskcard --help` match
  1.18.1 byte for byte, and `oe-inferencex --help` (the console script) does too.

## 7. Risks, scope and consequences

### 7.1 Risks

| # | Risk | How it would show | Mitigation |
|---|---|---|---|
| R1 | A test patch silently stops reaching the moved code. | test_trust_zone:629 and :643 would still pass, untested. | Retarget in the same step (T1-T5), with call assertions added to the two silent tests. §4.6 lists where to patch from now on. |
| R2 | A lazy relative import points at a missing module after the move. | An ImportError on one CLI branch only, at run time. | All of them are listed in §3.9; `resolve_imports.py` checks every import statement at every step. |
| R3 | A split module misses a from-import. | A NameError on a rare path (for example `_class_design_exact` without `math`). | pyflakes gates on undefined names at every step; the battery runs every route. |
| R4 | Shared state is duplicated. | A second `RULES_OF` breaks verify of 1.17.x certificates; a second cache misses `cache_clear`. | Aliases rather than copies; re-export by reference; old files deleted (R8); identity checks in §6.4. |
| R5 | Name collisions. | `oe_inferencex.grade` becomes a module, `oe_inferencex.decide.decide` becomes a module, `cli.main` becomes a module, or a folder named after the SDK's `mcp`, `sample` or `data` misbehaves. | `agency/` (D2); no `decide/decide.py`, `agency/grade.py` or `cli/main.py`; mcp_server stays (D3); names checked against `.gitignore`. `api_identity` checks `callable(grade)` after importing everything. |
| R6 | Import eagerness. | cli now imports plan, decide and scale at import; a feature `__init__` imports an extra. | All three are numpy- or stdlib-only, so the cost is import time only. R1 and R6 keep readers and research empty. The bare-wheel import of `oe_inferencex.cli` gates it. |
| R7 | CLI help or order drifts. | `--help` text, metavars or subcommand order differ, and with them the MCP schema text. | Registrars hold verbatim blocks called in the fixed order (§3.2); the battery diffs every `--help`. |
| R8 | The parser description changes. | `build_parser` moved away from `cli.__doc__`. | `build_parser` stays in `cli/__init__.py` with the same docstring. |
| R9 | `written_by` and certificate digests differ because of the version. | The whole battery differs. | Run it on the pre-bump commit (S13), with the driver forcing 1.18.1 in any case. Never force a non-release string: verify parses it. |
| R10 | The moved files' paths show up in output. | A warning's `file:line`, or an install path, in stderr. | The normalisation in §6.3 acceptance 3, with every such line listed. |
| R11 | `python -m` and runpy on an old path. | `python -m oe_inferencex.taskcard` does nothing, or names the wrong program. | The finder's `get_code` serves the moved file's code, and the spec's origin is the moved file, so argv[0] and argparse's prog stay `taskcard.py` (§4.2; tests/test_layout.py runs it). The limit, stated in the CHANGELOG: in one process, once an old path is imported, `runpy.run_module` refuses it (the module's loader answers to its new name); `python -m`, and runpy before the import, still run it. |
| R12 | `__module__` and `__qualname__` of moved classes change. | Reprs, pickles and mkdocs anchors name the new modules (Cue, Fusion, LCCTile, TaskCard, Design, TooLarge, Unreadable). | No runtime output prints them (checked: only `type(x).__name__` is used). Old pickles still load, because the old paths alias. Pickles written by 1.19.0 name the new paths and do not load in 1.18.1. Noted in CHANGELOG. |
| R13 | Docs under `--strict`. | Old URLs break. | gen_api.py is unchanged, since no file exists at an old path; 18 redirects with absolute targets; the docs build is in the battery. |
| R14 | History lookups. | `git log --follow` loses a file that was split or edited on the move. | No file is left at an old path. Use `git log -C --follow` and `git blame -C -C`; commit messages name the source. |
| R15 | The research record at a later pin bump. | `scripts/upstream_revision.py` scans only the package's top-level `*.py`; `exp/out/upstream_revisions.json` cites `oe_inferencex/taskcard.py:379` and `:323`, and `tests/test_upstream_revisions.py:83` asserts the `oe_inferencex/taskcard.py:` prefix; `exp/jobs/e98_read.sh` and `e99_read.sh` `test -f oe_inferencex/olmoearth.py`; `tests/test_exp89.py:1165` patches `awf.ROOT`; `exp116:64` reassigns `est.hypergeom_interval` as a speed cache. | For the scan, its test, test_exp89 and exp116, nothing changes while the pin is 1.12.1. At a bump: make the scan recursive, re-cite the taskcard lines as `oe_inferencex/readers/taskcard.py`, and update the test's prefix. No file is left at `oe_inferencex/olmoearth.py`, and 1.19.0 does not break e98/e99: their unpinned route (a checkout of the package's `origin/main`) has not worked since the repository split of 14 Oct, because `exp/out` lives in the record, and their pinned routes (`E98_SHA`, `E99_SHA`) check out old commits where `oe_inferencex/olmoearth.py` exists. They do not use the record's pin: each runs in a checkout of the package. Before either is run at a package commit from S7 (c643776) on, including `origin/main` once 1.19.0 is merged, its `test -f oe_inferencex/olmoearth.py` must become `oe_inferencex/readers/olmoearth.py` or an import check. The `awf` alias keeps `ROOT` patchable. exp116's cache becomes a no-op: slower, same numbers. |
| R16 | Environment drift between the two clones. | GDAL or numpy bytes differ. | Same `uv.lock` (`cmp`), `--frozen`, the before/before2 control. |
| R17 | Stale local `build/lib/oe_inferencex/` (20 flat modules). | It could leak into an in-tree wheel built on the Mac. | Nothing is built on the Mac. The owner deletes `build/` before any local build. |
| R18 | Cluster quota and access. | /home over quota, or a lockout from polling. | Scratch-only clones deleted on success; one job at a time; no polling; the cost stated before each submission. |
| R19 | The alias finder outside a plain import. *(Added as built.)* | A deadlock when `find_spec` imports under the import system's global lock; a wrong exception for a blocked module; the standard lazy loader, a reload of a module that still carries an old path's spec, or the old `load_module` losing the moved module's own name or spec; tools that resolve imports from files not finding an old path. | `find_spec` imports nothing (it locates the file with `PathFinder`); the finder keeps each moved module's own spec, package and loader in its own dict, and `exec_module` finds them from the old path's spec and restores the name too; tests/test_layout.py checks finding without importing, the lazy loader, a reload of a module carrying an old path's spec, a blocked module, a pickle that names an old path and a reload of the package. Limits stated in the CHANGELOG: static tools (type checkers, editors, PyInstaller) do not resolve old paths, and the lazy loader imports an old path at once. |

### 7.2 Out of scope (1.19.0 does none of this)

- **Any change to behaviour or output,** including bug fixes noticed during the move. Note them and fix them in a
  later release. As built, one output line changed in review: the task-card page names
  `oe_inferencex/readers/taskcard.py` (§8, CHANGELOG 1.19.0).
- **Renaming any public or private name, CLI option, MCP tool or text.**
- **Merging same-named duplicates** (`_check_patch`, `PROB_TOLERANCE`, `Z95`, `_code`, `INTERVAL_NOTE`, `QUESTIONS`,
  `_pc`), or deleting dead code (`_band_count`, the unused `boundary_indicator` import).
- **Further splits:** `mcp_server.py` by feature, `certificate.py` into issue and verify, or `cli/__init__`'s
  re-export list.
- **Renaming the `est` handles,** or rewriting comments and docstrings that cite `cli.py:NNN`, `estimate.py` or
  `sequential.sequential_order` (including the stale comment at demo.py:165).
- **Removing any old path or the alias finder.** That is a later major-release decision with a deprecation period.
- **Test edits beyond T1-T6,** and any edit to the golden files. As built, the new `tests/test_layout.py` and two
  test docstrings that named moved files were added on top (§4.7).
- **The research record:** its code, docs, claims and pins. ADR-001 is not edited. Filing this ADR in the record,
  with a status line on ADR-001, is the owner's act, done with the release, before it goes out.
- **README and `skills/oe-inferencex/SKILL.md`,** which name no module paths.

### 7.3 Consequences

- **Easier.**
  - A command's code sits next to its logic. A new command touches one folder and one line in
    `cli/__init__.build_parser`.
  - `estimate`, `certify` and `plan` each read as a feature.
  - The shared arithmetic (`common/hypergeom`), sampling primitives and command plumbing have one home each.
  - Three lazy cycles are gone: estimate ↔ sequential, decide ↔ cli, and data → cli.
- **Harder.**
  - Paths are longer.
  - The re-export lists in `cli/__init__` and `estimate/__init__` must be kept in step by hand until the old paths
    are retired; the surface check guards them. So must the finder's `_MOVED`; tests/test_layout.py checks it
    against its own copy.
  - Tests that patch globals must patch the defining module (§4.6).
  - Static tools (type checkers, editors, PyInstaller) do not resolve the 18 old paths, which have no file.
- **Follow-ups, each its own decision:**
  - rename the `est` handles to their defining modules;
  - move `RULES_OF` and `_class_words_1171` to common, which removes the last common ↔ certify lazy cycle;
  - point the remaining internal imports at their homes;
  - deprecate the old paths, then remove them in 2.0;
  - update the docstrings that cite old file:line positions;
  - resolve the duplicates listed in §7.2.

## Appendix A: `step_check.sh` (cluster only)

```bash
#!/bin/bash
# One migration step of ADR-002 on aicr: a fresh clone of the step's commit, static checks, the suite, the old-path
# surface, the verbatim check. CLUSTER ONLY. Submit:
#   ssh aicr "sbatch --parsable --export=ALL,SHA=<full sha>,MODE=step ~/oeix-eqv/step_check.sh"
# MODE=baseline at 987dc7a writes ~/oeix-eqv/surface_before.json and the baseline test counts instead of checking.
#SBATCH -A p2026_0089_neu
#SBATCH -p cpu
#SBATCH -J oeix-step
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH -o /home/qi_zim_neu/olmoearth_inferenceX/slurm/%x-%j.out
BEFORE=987dc7ac7bc1a5548b48ebe92f0b8c52bcadd1ed
REPO=https://github.com/2imi9/olmoearth_inferenceX
SCRIPTS=$HOME/oeix-eqv
BASE=/scratch/qi_zim_neu/olmoearth_inferenceX/step/$SLURM_JOB_ID
export UV_CACHE_DIR=/scratch/qi_zim_neu/olmoearth_inferenceX/uv-cache PATH=$HOME/.local/bin:$PATH
export COLUMNS=100 PYTHONHASHSEED=0 TZ=UTC PYTHONUNBUFFERED=1 MPLBACKEND=Agg
test ${#SHA} = 40 || { echo "SHA must be the full 40-character sha"; exit 1; }
mkdir -p "$BASE" || exit 1
git clone -q "$REPO" "$BASE/src" || exit 1
git -C "$BASE/src" fetch -q origin cluster-suite || true
git -C "$BASE/src" checkout -q "$SHA" || exit 1
test "$(git -C "$BASE/src" rev-parse HEAD)" = "$SHA" || exit 1
cd "$BASE/src" || exit 1
uv sync -q --frozen --python 3.12 --extra geo --extra embeddings --extra mcp || exit 1
P="$BASE/src/.venv/bin/python"
fail=0

echo "== 1 pyflakes: no undefined name beyond the 1.18.1 baseline"
uvx -q --python 3.12 --from pyflakes==3.2.0 pyflakes oe_inferencex > "$BASE/pyflakes.txt"
grep -E "undefined name" "$BASE/pyflakes.txt"                                  # all of them, for the log
grep -oE "undefined name '[^']+'" "$BASE/pyflakes.txt" | sort -u > "$BASE/undefined.txt"
if [ "$MODE" = baseline ]; then cp "$BASE/undefined.txt" "$SCRIPTS/undefined_baseline.txt"; fi
comm -23 "$BASE/undefined.txt" "$SCRIPTS/undefined_baseline.txt" | grep . && { echo "NEW undefined names"; fail=1; }
echo "imported but unused (not gated): $(grep -c 'imported but unused' "$BASE/pyflakes.txt")"

echo "== 2 every import statement resolves"
"$P" "$SCRIPTS/resolve_imports.py" "$BASE/src" || fail=1

echo "== 3 the suite"
uv run -q --frozen --extra geo --extra embeddings --extra mcp pytest -q -p no:cacheprovider > "$BASE/pytest.txt" 2>&1 || fail=1
grep -E "^(FAILED|ERROR)" "$BASE/pytest.txt"; tail -2 "$BASE/pytest.txt"
if [ "$MODE" = baseline ]; then tail -1 "$BASE/pytest.txt" > "$SCRIPTS/pytest_baseline.txt"; fi
echo "baseline: $(cat "$SCRIPTS/pytest_baseline.txt" 2>/dev/null)"   # compare passed/skipped counts by eye

echo "== 4 old import paths"
if [ "$MODE" = baseline ]; then
  "$P" "$SCRIPTS/eqv_surface.py" dump "$SCRIPTS/surface_before.json" || fail=1
else
  "$P" "$SCRIPTS/eqv_surface.py" check "$SCRIPTS/surface_before.json" || fail=1
fi

if [ "$MODE" != baseline ]; then
  echo "== 5 verbatim moves against 1.18.1"
  git -C "$BASE/src" worktree add -q "$BASE/before" "$BEFORE" || exit 1
  "$P" "$SCRIPTS/verbatim_check.py" "$BASE/before" "$BASE/src"     # reviewed against ADR-002 3.9
  echo "== 6 public API identity"
  "$P" "$SCRIPTS/api_identity.py" "$BASE/before" || echo "(api_identity must exit 0 from S9 on; earlier its new paths do not exist yet)"
fi

du -sh "$BASE"
if [ $fail = 0 ]; then rm -rf "$BASE"; echo "STEP OK"; else echo "STEP FAILED; kept $BASE"; fi
exit $fail
```

## Appendix B: `resolve_imports.py` (cluster only)

```python
"""Every import statement of the package resolves, at module level or inside a function (ADR-002, check 2).
CLUSTER ONLY. Usage: python resolve_imports.py <checkout>. Reads the sources with ast; imports only package modules."""
import ast
import importlib
import pathlib
import sys

root = pathlib.Path(sys.argv[1]).resolve() / "oe_inferencex"
sys.path.insert(0, str(root.parent))
bad = []


def check(path, lineno, base, names):
    try:
        mod = importlib.import_module(base)
    except ImportError as e:
        if "olmoearth-inferencex[" in str(e):          # an extra this environment lacks (the encoder stack)
            return
        bad.append(f"{path}:{lineno} {base}: {type(e).__name__}: {e}")
        return
    for name in names:
        if name != "*" and not hasattr(mod, name):
            try:
                importlib.import_module(f"{base}.{name}")
            except ImportError:
                bad.append(f"{path}:{lineno} {base} has no {name}")


for path in sorted(root.rglob("*.py")):
    package = ".".join(path.relative_to(root.parent).with_suffix("").parts[:-1])
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom):
            if node.level:
                parts = package.split(".")
                parent = ".".join(parts[:len(parts) - node.level + 1])
                base = f"{parent}.{node.module}" if node.module else parent
            else:
                base = node.module or ""
            if base.split(".")[0] == "oe_inferencex":
                check(path, node.lineno, base, [a.name for a in node.names])
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] == "oe_inferencex":
                    check(path, node.lineno, a.name, [])
print("\n".join(bad) or "every package import resolves")
sys.exit(1 if bad else 0)
```

## Appendix C: `verbatim_check.py` (cluster only)

The script is kept as it ran. The docstring's 32 is the plan's miscount; the list in §3.9 has 33 blocks (§8).

```python
"""Every top-level block of the 1.18.1 package appears byte-identical in the new tree (ADR-002, check 5).
CLUSTER ONLY. Usage: python verbatim_check.py <before checkout> <after checkout>. Reads files; imports nothing.
Prints the blocks not found verbatim; compare them with ADR-002 3.9's expected list (32 blocks at the end)."""
import ast
import pathlib
import sys


def blocks(path):
    text = path.read_text(encoding="utf-8")
    tree, lines = ast.parse(text), text.splitlines(keepends=True)
    out = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names = [node.name]
        elif isinstance(node, ast.Assign):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)] + \
                    [e.id for t in node.targets if isinstance(t, ast.Tuple) for e in t.elts if isinstance(e, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names = [node.target.id]
        else:
            continue
        start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
        segment = "".join(lines[start - 1:node.end_lineno])
        for n in names:
            out[n] = segment
    return out, ast.get_docstring(tree, clean=False)


before, after = (pathlib.Path(a) / "oe_inferencex" for a in sys.argv[1:3])
segments, docs = set(), set()
for p in after.rglob("*.py"):
    b, d = blocks(p)
    segments |= set(b.values())
    docs.add(d)
missing = []
for p in sorted(before.glob("*.py")):
    b, d = blocks(p)
    if p.name != "__init__.py" and d not in docs:
        missing.append(f"{p.name}: module docstring")
    missing += [f"{p.name}: {n}" for n, s in b.items() if s not in segments]
print("\n".join(missing))
print(f"{len(missing)} blocks not found verbatim")
```

## Appendix D: `api_identity.py` (cluster only)

```python
"""The public API and the shared state after the move (ADR-002 6.4). CLUSTER ONLY.
Usage: python api_identity.py <before checkout>   (run with the AFTER environment)."""
import ast
import importlib
import importlib.metadata
import pathlib
import pkgutil
import sys

tree = ast.parse((pathlib.Path(sys.argv[1]) / "oe_inferencex" / "__init__.py").read_text(encoding="utf-8"))
homes = {a.asname or a.name: f"oe_inferencex.{n.module}" for n in tree.body
         if isinstance(n, ast.ImportFrom) and n.level == 1 and n.module for a in n.names}
before_all = next(ast.literal_eval(n.value) for n in tree.body
                  if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) == "__all__")

import oe_inferencex as ox                                  # noqa: E402
for info in pkgutil.walk_packages(ox.__path__, "oe_inferencex."):
    if info.name.endswith("__main__"):
        continue
    try:
        importlib.import_module(info.name)
    except ImportError as e:
        if "olmoearth-inferencex[" not in str(e):
            raise
bad = []
if list(ox.__all__) != list(before_all):
    bad.append("__all__ differs from 1.18.1")
for name in ox.__all__:
    if name == "__version__":
        continue
    obj, home = getattr(ox, name), importlib.import_module(homes[name])
    if getattr(home, name) is not obj:
        bad.append(f"oe_inferencex.{name} is not {homes[name]}.{name}")
    if not str(getattr(obj, "__module__", "oe_inferencex")).startswith("oe_inferencex"):
        bad.append(f"{name}: __module__ {obj.__module__}")
if not callable(ox.grade):
    bad.append("oe_inferencex.grade is no longer the function")
m = lambda n: importlib.import_module(f"oe_inferencex.{n}")
same = [("estimate", "RULES_OF", "certify.zone"), ("estimate", "_hypergeom_interval", "common.hypergeom"),
        ("estimate", "_SIGN_CRIT", "estimate.which_map")]
bad += [f"{a}.{x} is not {b}.{x}" for a, x, b in same if getattr(m(a), x) is not getattr(m(b), x)]
for old, new in [("certificate", "certify.certificate"), ("sequential", "certify.sequential"),
                 ("stratified", "agency.stratified"), ("assess", "review.assess"), ("taskcard", "readers.taskcard")]:
    if m(old) is not m(new):
        bad.append(f"oe_inferencex.{old} is not oe_inferencex.{new}")
if m("cli").est is not m("estimate"):
    bad.append("cli.est is not the estimate package")
eps = [e for e in importlib.metadata.entry_points(group="console_scripts") if e.name == "oe-inferencex"]
if not eps or eps[0].load() is not m("cli").main or m("cli").main.__module__ != "oe_inferencex.cli":
    bad.append("the console entry point does not load oe_inferencex.cli.main")
print("\n".join(bad) or "public API and shared state identical")
sys.exit(1 if bad else 0)
```

## 8. Outcome (2026-10-27)

**What was built.** The layout of §2, on the branch `feature-folders`: 25 commits from S1 (d0bb7a9) to the release
check (443281c), the fixes of two reviews (796dcb1, ebd78e5) among them. S11 and S13 have no commit of their own: S11 moves nothing and S13 is the battery. Against the plan:

- **One alias finder instead of 18 shim files.** This was the owner's choice, made on 26 Oct before the migration
  started. The 18 old paths are answered by `_MOVED` and `_MovedModules` in `oe_inferencex/__init__.py` (§4.2). No
  file exists at an old path. `docs/gen_api.py` is unchanged, since there are no shims to skip. `mkdocs.yml`
  redirects the API pages of the 18 moved modules. The root `__init__` imports the new paths, and no package module
  imports an old one.
- **33 verbatim blocks, not 32.** The expected list in §3.9 has 33 blocks; the plan miscounted it. The step check
  matched all 33.
- **One more output line.** The review fixes made the task-card page name `oe_inferencex/readers/taskcard.py`, and
  changed two docstrings that named moved files (`cmd_grade`, `cmd_scale`). With `to_markdown` and `cmd_grade`, two
  blocks beyond the 33 no longer match 1.18.1 verbatim.
- **Tests.** T1-T6 as in §4.7, T6 as completed in review. New: `tests/test_layout.py`.

**The review.** A read-only review of 26-27 Oct (workflow wf_a0fb8b41-d9e) found the following. Commit 796dcb1 fixed
each:

- `find_spec` imported the target's folder under the import system's global lock: a deadlock risk, and an
  `AttributeError` for a blocked module. It now locates the file with `PathFinder` and imports nothing; a blocked
  module is refused as `ModuleNotFoundError`.
- The moved module's own spec and package were kept on the alias spec's `loader_state`, which broke the standard
  library's `importlib.util.LazyLoader`. The finder now keeps them, with the loader, in its own dict.
- test_reliability's evidence test (T6) let a module bound to the stub torch outlive the test. A `try/finally` now
  removes it and the attributes the import bound.
- Text: the package docstring gives `decide` its own row; the docstring and Usage say the shared window readers are
  in `review`; docstrings, two test docstrings and the task-card page name the moved files' paths; the CHANGELOG
  counts 25 modules, states the limits below and the one output line, and dates the release 27 Oct. Tests were added
  for runpy, the lazy loader, blocking, pickling and a reload.

A second read-only review, of 796dcb1 (workflow wf_96920dc5-7cc), found two more. Commit ebd78e5 fixed both:

- `exec_module` looked the module up by its `__name__`, which a reload of a module still carrying an old path's
  spec, or the old `load_module`, has already overwritten: a `KeyError`, and the module left renamed. It now finds the
  new path from the old path's spec, which the import system always sets, restores the name too, and raises
  `ImportError` for a module it never recorded.
- Under the standard lazy loader an old path is imported at once, since `create_module` must return the moved module
  itself. The CHANGELOG now states this, and the test asserts it instead of implying a lazy load.

It also corrected the release-check counts (25 tests in test_layout.py, 1,321 in all) and Usage's account of
`common`: shared helpers live there, and features also use one another's logic. It refuted a third candidate: that
locating the moved file with `PathFinder` breaks installs whose modules no file-based finder serves; no supported
install (the wheel, the setuptools editable install, a zip) is one, and the CHANGELOG already says that tools which
resolve imports from files do not find the old paths.

The review refuted one candidate finding: that 1.19.0 breaks `exp/jobs/e98_read.sh` and `e99_read.sh` in
the record. Their unpinned route has not worked since the repository split of 14 Oct, and their pinned routes check
out old commits where `oe_inferencex/olmoearth.py` exists. The change they need before they run at a 1.19.0 commit is in R15.

**Limits, as the CHANGELOG's 1.19.0 entry states them.** Pickles written by 1.19.0 name the new paths and do not
load in 1.18.1; pickles written by 1.18.1 still load. Static tools (type checkers, editors, PyInstaller) do not
resolve the old paths. In one process, once an old path is imported, `runpy.run_module` refuses it, because the
module's loader answers to its new name; `python -m` with an old path, and runpy before the old path is imported,
still run it. Under the standard lazy loader an old path is imported at once; lazy-load the new path.

**Checks on aicr** (cpu partition, sbatch):

| Job | Commit | What it showed |
|---|---|---|
| 1543223 | 987dc7a (1.18.1) | the baseline |
| 1543224 | 88b4884 (S12) | the step check: 33 verbatim blocks matched the expected list |
| 1543225 | 88b4884 (S12) | the battery: 157 steps and 418 files byte-identical between 1.18.1 and the move, with 1.18.1 run twice as the control; pytest 1,296 passed before and 1,319 after; the public surface, `--help`, the bare demo and the docs agree apart from the checkouts' absolute paths |
| 1544004, 1544005 | 796dcb1 | the battery: as 1543225, with pytest 1,321 after; the suite on 3.11 and 3.13: 1,321 passed on each, the test order that let the stub-bound module leak passes, `python -m oe_inferencex.taskcard` runs |
| 1544286, 1544287 | ebd78e5 | the same two, on the code released: 157 steps and 418 files byte-identical, pytest 1,296 before and 1,321 after on 3.12, 1,321 on 3.11 and 3.13; the surface, `--help`, the bare demo and the docs agree apart from the checkouts' paths and the version stamp. 443281c adds only these numbers to the CHANGELOG |
