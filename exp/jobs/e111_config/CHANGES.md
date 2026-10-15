# exp111 configs: what differs from exp98's (and so from Ai2's AWF deployment)

Base: `exp/jobs/e98_config/` (Ai2's `allenai/olmoearth_projects` at `f3c9b0c89c7670b3525647dda4c14b76245d682d`,
`olmoearth_run_data/awf/`, with exp98's two changes; `exp/jobs/e98_config/CHANGES.md`).

| File | Here | Change |
|---|---|---|
| `e98_config/dataset.json` (`ad076afb...d328`) | `dataset.json` | one key added: `layers.sentinel2.data_source.query_config.per_period_mosaic_reverse_time_order: true` |
| `e98_config/scl_dataset.json` (`e3d8de8b...7a01`) | not copied | used as it is (the sidecar never runs `prepare`; its item groups are the input's, copied in order) |
| `e102_config/model.yaml` (`b8af519a...5e9e`) | not copied | used as it is: exp98's `model.yaml` (Ai2's plus `output_probs` and the float32 `p0..p9` writer layer) with `model_id: ${MODEL_ID}` and a CSVLogger; one file for both checkpoints |

`exp/exp111_jobs.py check-configs` refuses unless `dataset.json` equals exp98's after removing exactly that key, and
the other two files hash as above.

## Why the one key

The deployment's 12 item groups are newest first: rslearn 0.0.27, which exp98 and exp99 ran, returned the
`PER_PERIOD_MOSAIC` groups in reverse time order (`exp/jobs/e98_env.sh`). rslearn 0.1.17 still defaults
`per_period_mosaic_reverse_time_order` to True, and its docstring says the default "will change to False". The model
reads the groups in the order they are stored (`load_all_item_groups`), so a silent default change would feed both
checkpoints a reversed time series. Setting the key to True states the deployment's order instead of inheriting it.
rslearn 0.1.17 validates it (`QueryConfig`, `extra="forbid"`), and `check-dataset` asserts the parsed value.

Consequence for the fallback route (v1 in rslearn 0.0.29, owner's call only): if that version's `QueryConfig` does not
know the key, its `config.json` must be the one without it (`check-configs --reverse` writes it).

## What the time range means (rslearn 0.1.17, `rslearn/data_sources/utils.py:419-482`)

The windows' time range is the year's request range, `[YYYY-01-01T00:00Z, YYYY-12-31T00:00Z]`, as exp98's and exp99's
deployment windows hold it (`check-dataset` and `deploy-time-range` compare). Periods of 30 days are cut BACKWARDS
from the end: the newest is `[12-01, 12-31)`, the twelfth `[01-05, 02-04)`. A period without any scene covering the
window is skipped, so a window with an empty period gets 11 groups and is rejected (`min_matches = 12`).
