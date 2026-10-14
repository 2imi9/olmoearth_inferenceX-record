# OlmoEarth Agent trial through Studio, 24 September 2026 (co-development log)

Kept as a record beside the audits; not an experiment, and nothing here is a ledger claim. The fixes it lists are
being made in the OlmoEarth Agent repository (branch 2imi9/feature-studio-review-and-inferencex-1-3) and, for T1, in
docs/Usage.md here.


Goal: use the package the way an outside user would, through the OlmoEarth Agent (main at a26a5c7, PR 155 merged)
driving OlmoEarth Studio, with Qwen3.8-27B-NVFP4 served on aicr. A friction log, not an experiment: nothing here
enters the claim ledger unless it is later preregistered and rerun.

## Constraints known in advance
1. Model availability. GPU jobs 1041135 (rtx/b200-batch, 8 h) and 1041141 (rtx/b200-devel, 4 h) are queued on a
   shared cluster; the start time is not ours to set. The one that starts second is cancelled. Start-up ~3.5 min.
   When the job's time runs out the model disappears mid-conversation: plan each session inside the window.
2. Speed. ~63 tokens/s single stream on an RTX PRO 6000, reasoning on: a turn with tool calls can take tens of
   seconds. Slowness alone is not a bug.
3. Studio. Predictions are submitted and polled; they take minutes. The key is the user's, entered in the page and
   kept in the browser. Rate limits and HTTP errors from Studio are Studio's, not the agent's.
4. Version gap. The agent carries its own port of the review-set logic from PR 155 (18 September): review set,
   compare, grade a rule, budget ceiling. It has none of 1.3.0's estimate, per-class, certify, or the dates in
   compare. Missing capability of that kind is an integration lag, not a defect in either repository.
5. Inputs. A Studio prediction may come without per-window confidence (the LCC export has no class confidence).
   Without a confidence the review set can only fall back to boundary triage; whether the agent says so is part
   of the test.
6. Fair use. The GPU is released (scancel) as soon as a session ends.

## How each problem is attributed
- **harness**: the agent's loop, prompts, UI, tool routing, Studio client, parsing of model output.
- **tool**: the review-set / compare logic computing or refusing something wrongly (the agent's port or
  `oe_inferencex` itself; say which).
- **model**: Qwen3.8 choosing a wrong tool, inventing a number, ignoring a tool result.
- **Studio**: the service's errors, limits, missing fields.
- **ours (infra)**: the tunnel, the vLLM job, the token.
A problem is logged with the brief, the tool calls and arguments, the output, and the attribution with its reason.

## Briefs, in order (each builds on the one before)
1. "Which fine-tuned OlmoEarth models can I run, and what does each predict?" (task cards; Studio listing)
2. "Run <model> on <small area> and tell me which windows a reviewer should check first, and why."
   (end to end: prediction, confidence, review set, reasons)
3. "Compare two predictions of the same area (two models, or two dates) and tell me where they differ and
   which is right." (compare; the date caution is expected to be missing, see 4 above)
4. "How wrong is this map? I can label 300 windows." (not in the agent: does it say so, or invent an answer?)

## What counts as success
The agent reaches the tool on its own, reports the tool's numbers without inventing any, and declines what the tool
cannot answer (exp64's decline). Every departure is logged and attributed.

## Found before the model was up (24 September, with the user's new Studio key; read-only calls)
- The key works (users/me). The account sees one project (PA Karst) and three models: KarstEmbedding (embeddings),
  KarstNumber and KarstBinary (fine-tuned, prediction_type per_pixel_regression; properties sample_number,
  sample_karst_score). One completed 2025 prediction each.
- A prediction result carries PNG map tiles (tile_urls) and a point lookup (`/prediction-results/{id}/pixel-value`,
  raw_value for regression, classification for categorical); `output_files` is empty: no raster download.
  **Studio**: no raster of raw scores or class probabilities to hand to the review set.
- KarstBinary is a regression score, not class probabilities; using it as a binary probability is an assumption the
  agent must state. **model/harness** if it does not.
- User's point: the agent may bridge this itself (sample the score on a window grid through pixel-value, or the
  qgis tool's GDAL WMS description of the tiles), then call the review set. Brief 2 tests whether it does so
  unprompted; if not, the missing grid-sampling step is **harness** (a tool the agent repo can add), not Studio.

## Runs (Qwen3.8-27B-NVFP4 on aicr job 1041141, node a0018, RTX PRO 6000; agent a26a5c7)
- 1a (11:33) infra: the tunnel's SSH connection was closed by the login node; the agent's first LLM call failed.
  Fixed with a reconnecting tunnel loop (tunnel.log).
- 1b, 1c (11:41) infra: the Mac's resolver had cached a failed lookup of olmoearth.allenai.org after a transient DNS
  failure on the campus resolvers; nslookup answered, getaddrinfo did not. The user flushed the cache (sudo). The
  agent reported the failure honestly and invented nothing (harness and model behaved correctly).
- 1d (11:44-11:46, 80 s, 5 tool calls ok): the three models listed correctly with ids and output layers.
  Minor: calls KarstBinary's regression score a "continuous confidence layer" (model; it never read prediction_type);
  lists only the user's models, not Ai2's public fine-tuned ones, though a task-card tool exists (harness/model).
- 2 (11:46-11:53, 7.5 min, 27 tool calls ok, 7 turns): sampled the existing KarstBinary result with 21 pixel_value
  calls (a 4x4 grid over the whole AOI), used ensemble_uncertainty against KarstNumber, and wrote a review plan.
  + bridged the missing raster itself (the user's hypothesis); + stated the score is not a calibrated probability.
  - never called olmoearth_review_set (harness: no tool turns a Studio prediction into its input; no routing).
  - ranked the 0.97/0.99 cells FIRST ("the only karst calls"), the inverse of the method it claimed to follow
    (low margin = suspect: 0.23 and 0.31 should lead) (model; prevented if the tool ranked).
  - 16 single-pixel samples over a state are not windows (harness: needs a grid-sampling tool at window scale).
  - "cross-run disagreement" between a binary score and a count regression (model; harness let the tool compare
    different targets).
  - inferenceX: no documented path for a binary regression score; could accept it with a stated mapping
    (margin = |2s-1|) and say so in the output.
- 3 (11:54-11:56, 2 min, 5 tool calls ok): olmoearth_compare_results on KarstBinary vs KarstNumber, 6x6 grid.
  + declined "which is right" without labels (exp64's decline holds in the shipped agent).
  - BUG (harness, agent repo: analysis/raster_compare.compare_numeric via tools/predict._compare_results): Studio's
    pixel-value returns the no-data sentinel -1 as a value, not None; only None is dropped. Reproduced with the
    tool's own sampling: 11 of 36 points are -1 in both maps; correlation 0.946 as shipped vs -0.017 with no-data
    removed, agreement 31% vs 0%, KarstBinary mean -0.24 vs 0.10. The narrated "strongly correlated, karst in
    largely the same places" is produced by the shared no-data. Fix: drop the model's nodata_value (wizard
    answers carry it) or any -1 sentinel before the statistics, and report how many were dropped.
  - compared a binary score with a count regression without saying they measure different things (model;
    harness could refuse different property semantics).
