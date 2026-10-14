# Repository cleanup audit

olmoearth_inferenceX at commit c5f25ec, 17 September 2026. Every one of the 808 tracked files was reviewed. Tier 1 and Tier 2 were applied the same day at the author's choice (see the changelog); Tier 3 stays, except exp21's figure, which went with the Word report.

## How the list was made

- Nine auditors covered all 808 files, each accounting for every file in its scope.
- Two independent refuters checked every candidate: one hunted for consumers in code, tests, the docs build and the claim ledger; the other checked the research record (docs, lab log, report, README, recorded decisions). A candidate kept the most cautious verdict either gave.
- A completeness critic ran twice; the first pass added 7 candidates and the second found nothing new.
- The safe set was then deleted in a throwaway git worktree. The four live TikZ diagrams compiled, 309 tests passed, the ledger showed only its two superseded claims, and the strict docs build was clean.

## Summary

| Tier | Files | Size | What removal needs |
|---|---|---|---|
| 1. Safe to remove now | 28 | 19.7 MB | nothing; verified by deletion |
| 2. Removable, your call | 413 | 68.2 MB | the edits listed, and accepting what is lost |
| 3. Proposed, then kept | 11 | 18.4 MB | not removable |

The other 356 tracked files were never candidates: the package, tests, scripts, live docs and figures, claim-pinned outputs, and the scripts behind recorded experiments.

## Tier 1: safe to remove now

### Abandoned overview figure set (6 files, 558 KB)

Unlinked since 8 September and still carrying wording corrected on 9 September. The six files mention only each other, so they go together.

- `docs/figures/inferencex_overview.png`
- `docs/figures/inferencex_overview.py`
- `docs/figures/tikz/inferencex_overview.tex`
- `docs/figures/tikz/inferencex_overview.pdf`
- `docs/figures/tikz/README.md`
- `docs/figures/tikz/PROMPT.md`

### Retarget draft figures (6 files, 982 KB)

Drawn on 7 September to illustrate the re-targeting idea. No page, README or report has ever embedded them.

- `docs/figures/latent_mim_retarget_idea.png`
- `docs/figures/latent_mim_retarget_idea.py`
- `docs/figures/olmoearth_pretraining_vs_retarget.png`
- `docs/figures/olmoearth_pretraining_vs_retarget.py`
- `docs/figures/proposal_retargeted_residual.png`
- `docs/figures/proposal_retargeted_residual.py`

### Leftovers from early drafts of the comparison figure (7 files, 58 KB)

Written only by superseded versions of the raster script; the other draft rasters were already deleted and these were missed.

- `docs/figures/tikz/rasters/cmp_s1pre.png`
- `docs/figures/tikz/rasters/cmp_s1post.png`
- `docs/figures/tikz/rasters/cmp_s2.png`
- `docs/figures/tikz/rasters/cmp_s2post.png`
- `docs/figures/tikz/rasters/cmp_sens_diff.tex`
- `docs/figures/tikz/rasters/cmp_sens_error.tex`
- `docs/figures/tikz/rasters/cmp_time_diff.tex`

### Raster outputs no diagram reads (3 files, 6 KB)

The raster script still writes them, so a rerun recreates them untracked; dropping those three save lines is an optional tidy-up.

- `docs/figures/tikz/rasters/bolivia_pred.png`
- `docs/figures/tikz/rasters/scene_dihedral.png`
- `docs/figures/tikz/rasters/scene_errors.tex`

### PDF build products of the four live diagrams (4 files, 479 KB)

Pages, README and report use the exported PNGs. pdflatex recreates each PDF exactly from tracked sources.

- `docs/figures/tikz/audit_pipeline.pdf`
- `docs/figures/tikz/compare_inferences.pdf`
- `docs/figures/tikz/explanation_layer.pdf`
- `docs/figures/tikz/protocol.pdf`

### A resume checkpoint of exp04 (1 files, 17.6 MB)

A partial checkpoint whose 1,440 rows are bit-identical to the first rows of the committed exp04_feats.npz. The script reads it only when that file is missing.

- `exp/out/exp04_ckpt.npz`

### The figure of a superseded run (1 files, 80 KB)

exp09 was overturned by exp11 and exp13. Every plotted value is in the kept exp09_multiscene.csv, and no page embeds the figure.

- `exp/out/exp09_multiscene.png`

## Tier 2: removable, applied the same day at the author's choice

### Flat CSV copies of claim-pinned summaries (16 files, 77 KB)

Every number is in the pinned summary json of the same experiment, checked cell by cell. You lose only a one-table view. No edit is required; adding each to .gitignore stops a rerun leaving an untracked copy. Five of them are named only by the paper outline below.

- `exp/out/exp46_shared_error_sources.csv`
- `exp/out/exp47_served_ranker.csv`
- `exp/out/exp49_shrug_signals.csv`
- `exp/out/exp51_their_probe.csv`
- `exp/out/exp52_finetune.csv`
- `exp/out/exp53_shift_tta.csv`
- `exp/out/exp54_multiclass.csv`
- `exp/out/exp55_geoid_flood.csv`
- `exp/out/exp56_label_efficiency.csv`
- `exp/out/exp60_two_periods.csv`
- `exp/out/exp61_residue.csv`
- `exp/out/exp62_fourth_cell.csv`
- `exp/out/exp63_multiclass_atlas.csv`
- `exp/out/exp65_calibrate.csv`
- `exp/out/exp66_dfc2020.csv`
- `exp/out/exp67_dynamic_world.csv`

### Resume checkpoints of exp73 and exp74 (384 files, 441 KB)

All 384 files were parsed and are identical to the pinned summaries. You lose the per-pair files and the ability to resume those scripts. Edits: the Outputs lines in comparisons.md and the jobs README, and the sentence '306 kept checkpoints' in the exp74 section must say they are at tag v1.0.0.

- `exp/out/exp73_parts/` (24 files)
- `exp/out/exp74_parts/` (360 files)

### The largest file in the repository (1 files, 64.5 MB)

exp16's cache of dense encoder token grids, 66 MB. Only exp16 reads it, as a cache; its results are in exp16_summary.json (four claims) and exp16_awf_boundary.csv. You lose rerunning exp16's analysis without an encoder pass.

- `exp/out/exp16_dense_val.npz`

### The early Word report and the figures only it embeds (4 files, 2.4 MB)

report/main.pdf replaces the Word report, and it stays in tag v1.0.0. ADR-001 kept it, so removal needs a dated note in the ADR. Three figures (exp23, exp24, exp25) have no other user; two more (exp21, exp22) came back KEEP only because the Word report embeds them, so all five become removable with it.

- `report/2026-09-02_experiments_exp01-25.docx`
- `exp/out/exp23_reference_instability.png`
- `exp/out/exp24_year2021.png`
- `exp/out/exp25_seasonal_water.png`

### Figures no page shows (2 files, 762 KB)

exp05_hard_scenes.png is the only view of a superseded run, since exp05 writes no table. summary_transfer.png was pulled from the README on 4 September as misleading, and its script says it is diagnostic only.

- `exp/out/exp05_hard_scenes.png`
- `exp/out/summary_transfer.png`

### The workshop paper outline (1 files, 35 KB)

Superseded by the technical report, and stale inside: its newest experiment is exp67. Edits: the nav line in mkdocs.yml, and one SHRUG-FM sentence worth moving to the exp49 section first. The refuter disputed the auditor's reason, not the mechanics.

- `docs/plan/paper_outline.md`

### Scripts ADR-001 kept on purpose (5 files, 45 KB)

Each is dead: a draft never run past a pilot, a download no run used, a sweep with no job id, a first 27B run that was superseded, and a native serving probe that failed. Removal reverses a decision taken yesterday, so each needs a jobs-README row edit and a dated note in the ADR.

- `exp/exp27_decoder_worldcover.py`
- `exp/jobs/dl38.sh`
- `exp/jobs/e64full.sh`
- `exp/jobs/e64run27.sh`
- `exp/jobs/vllm38.sh`

## Tier 3: proposed, then kept after a refuter found a reason

| File | Why it stays |
|---|---|
| `docs/plan/readout_design_theory.md` | The preregistration of exp48, and GitHub issue 13 for exp48 is still open. |
| `exp/exp64_serve.py` | Three committed job scripts call it, so removing it breaks the kept pilots. |
| `exp/jobs/e64armE.sh` | The only committed source of a job id that the results pages rely on. |
| `exp/jobs/e64pilot.sh` | The only committed source for a limitation cited in the exp64 results section. |
| `exp/out/exp05_cache.npz` | Three recorded scripts read it as input, even though its arrays duplicate exp11_scenes.npz. |
| `exp/out/exp09_cache.npz` | exp10 reads it under a claim pin, even though its arrays duplicate exp11_scenes.npz. Pointing exp10 at that file would free 10 MB, but that edits a recorded script. |
| `exp/out/exp21_finetuned_awf.png` | Was embedded only in the Word report; removed together with it. |
| `exp/out/exp22_lcc_striping.png` | Embedded in the Word report, and the only store of the full periodogram curves. |
| `exp/out/exp39_cache.npz, exp40_cache.npz, exp41_cache.npz` | Nothing reads them, but the lab log names each as the store of per-window scores that no other committed file holds. |

## Side findings

- **A false sentence in exp/jobs/README.md.** The rows for `e64run27.sh` and `e64E27.sh` say two superseded answer files are kept in `exp/out/`. They were kept on the cluster and never committed, so the rows should say that. I wrote those rows yesterday.
- **ADR-001's context counts are dated.** It says 143 claims and 203 artifacts; today the ledger has 146 claims and there are 565 tracked outputs. The counts describe the state when the decision was taken, so a dated note is enough.
- **Local clutter outside git:** `build/`, `dist/`, `site/`, the egg-info folder and LaTeX build files in `report/`. They are gitignored, not in the repository, and safe to delete locally.

## Evidence

Each candidate's full reasoning, the commands its auditor and refuters ran, and every proposed edit were delivered with this list as `audit_result.json`; that file is not committed.
