# Cluster job scripts

The Slurm scripts behind the recorded runs that needed a served language model or the Hub's encoder set
(exp64, exp72, exp73, exp74), copied verbatim from the home directory on the aicr cluster on 2026-09-17
(ADR-001, docs/plan/adr-001-repository-layout.md). The account (`p2026_0089_neu`), the partitions and every
path under `/home/qi_zim_neu` and `/scratch/qi_zim_neu` are site-specific; nothing here runs elsewhere
unchanged. Earlier experiments (exp01 to exp70) were submitted with sbatch files that were not kept in the
home directory; their job ids are in the lab log (`exp/NOTES.md`) and in the commit that recorded each one,
and `slurm_log_index_2026-09-17.txt` is the listing of every job log (job name, id, date) at the time of the copy.
Thirteen of the seventeen scripts copied that day remain: four superseded setup and pilot scripts (`dl38.sh`,
`vllm38.sh`, `e64full.sh` and `e64run27.sh`, the last being job 876751, the first 27B run that the token budget cut short)
were removed later the same day after the full-tree audit and remain in tag v1.0.0.

Every script follows one pattern: `cd ~/olmoearth_inferenceX`, reset the checkout to `origin/main`, point
`HF_HOME` at the scratch cache, run one experiment script, and, for exp64, grade. The served-model jobs
start vLLM from the Apptainer image, wait for the endpoint, and run a two-turn tool-call preflight before
the arms. `OLMOEARTH_API_KEY=exp64-no-studio-access` is not a credential: the agent's client refuses to start
with the variable empty, and the benchmark never calls the Studio API. Job names are the `-J` values; each
job's log is `slurm/<name>-<jobid>.out`, and `slurm/serve-<jobid>.log` for the model server, kept outside
the repository (`slurm/` is gitignored).

| Script | What it runs | Jobs (name, ids) | Result |
|---|---|---|---|
| `pullvllm.sh` | pulls `vllm/vllm-openai:nightly` into an Apptainer image (`vllm-openai-nightly.sif`) and prints its vllm, nvcc and Python versions | pullvllm 876450 | the image every served-model job below uses |
| `vllm38c.sh` | serves `nvidia/Qwen3.8-27B-NVFP4` from the container (the natively installed vLLM fails on these nodes, which have a driver but no CUDA toolkit; job 876444), one chat completion and a throughput probe | vllm38c 876677 | up in about 210 s; 63 tokens per second single stream on an RTX Pro 6000 |
| `e64cards.sh` | builds the exp64 cards (`--stage cards --sources geoid,sen1`) | e64cards 836588, 876757 | `exp/out/exp64_cards.csv` (40 cards) |
| `e64pilot.sh`, `e64armE.sh` | pilots on `Qwen/Qwen2.5-7B-Instruct` through the transformers shim (`exp/exp64_serve.py`) | e64pilot 842234, 842489, 842712, 842885, 876097; e64armE 876216 | superseded: the shim double-encoded tool arguments and dropped braces; nothing recorded |
| `e64AD27.sh` | exp64 arms A to D on `nvidia/Qwen3.8-27B-NVFP4` with the 8,000-token budget and the final answer turn, then grade | e64AD27 877661 | `exp/out/exp64_summary.json`, `exp64_answers.jsonl` (commit 36f2dd7) |
| `e64E27.sh` | exp64 arms E and E_forced (the OlmoEarth Agent on its branch `2imi9/feature-inferencex-review-set`) at 27B, then grade | e64E27 877101, 877753 | `exp/out/exp64_answers_E.jsonl`, `exp64_agent_tool_use.json` (36f2dd7); the earlier E output, from before the driver wrote the second inference, was set aside on the cluster checkout as exp64_run1_E_nocompare_answers.jsonl, never committed |
| `e64_7b.sh` | the preregistered model-size ablation: arms A to D and E on `Qwen/Qwen2.5-7B-Instruct` (32k context) | e64_7b 880180 | `exp/out/exp64_qwen25_7b/` (commit 94815cd); its E arm was refused on every request, the client's 32,768-token default not fitting beside the prompt |
| `e64E7b.sh` | arm E at 7B again with an 8,192-token completion budget (`--max-output-tokens 8192`) | e64E7b 881794 | `exp/out/exp64_qwen25_7b/exp64_answers_E.jsonl` (94815cd) |
| `e72.sh` | exp72: the agent skill's pure-Python port against the package on Ai2's real logits | e72xcheck 841999, 842048 | `exp/out/exp72_agent_crosscheck.json` (commit da0b536) |
| `dl74.sh` | downloads the embeddings of the encoders exp74 scores, and `eval_settings/`, from `allenai/olmoearth-paper-embeddings` | dl74 880196 | the cache exp74 reads |
| `e73.sh` | exp73: the strong alternatives (ensemble, nearest-neighbour typicality, Mahalanobis) on the 24-task suite | e73alts 876749, 877293, 879619, 879638 | `exp/out/exp73_summary.json`, `exp73_tasks.csv` (commit 4f593cc, from 877293 and 879638; the other two stopped early on memory and then on the time limit, which is why the script checkpoints per task) |
| `e74.sh` | exp74: the suite's protocol under 15 encoders | e74enc 880197, 881793 | `exp/out/exp74_summary.json`, `exp74_encoders.csv` (recorded 2026-09-17 from 881793; 880197 is the run whose 54 absences were the loader's index error) |
| `e77demo.sh` | `scripts/make_demo_sample.py`: the real map of `oe-inferencex demo`, the lower-median Dynamic World test tile by error capture among the fully annotated ones | e77demo 995922 | `oe_inferencex/sample/dynamic_world_tile.npz`, `exp/out/demo_sample_selection.json` |
| `e77.sh` | exp77: is a confident error a window that agrees with its scene? Seven segmentation tasks, the suite's own probe, arm A against arm B on tokens with the tile-mean direction projected out | e77scene 1003587 | `exp/out/exp77_summary.json`, `exp77_tasks.csv` |
| `e78.sh` | exp78 stage one: fit exp70's probe on the 24 tasks and export per-unit margin, p1, decision and error, so every later reanalysis of the suite costs no cluster time | e78export (pending) | `exp/out/exp78_units/*.npz`, `manifest.json` |
| `e88.sh` | exp88 (docs/plan/missing_modality.md): the numpy and torch smokes, then `align` (identifiers, shapes and split sizes of the three inputs of each family; allowed before the plan is frozen) or `run` (the S1+S2 probe read on S1+S2, S1-only and S2-only test embeddings; the script refuses it until the plan page says frozen) | e88modality (align 1107181, run 1107807) | `exp/out/exp88_alignment.json`; `exp/out/exp88_summary.json`, `exp88_units.npz` |
| `e98.sh` | exp98 (docs/plan/awf_deployment.md), the record's re-run of Ai2's FT-AWF deployment configuration: the smoke, then `inv` (label points inside the deployed FT-AWF map by split, how each was mapped, time ranges and, with `E98_DEPLOY`, the 10 m inputs and Sentinel-2 configuration at the label pixels; no class or probability read out; allowed before the plan is frozen) or `run` (the script refuses it until the plan page says frozen). Reads the part directories `oe-inferencex from-olmoearth` wrote (`E98_SCORES`, by `e98_read.sh`) as one map, and exp89's AWF extraction; chained with `--dependency=afterok` on `e98_read.sh` | not submitted | `exp/out/exp98_inventory.json`; `exp/out/exp98_summary.json`, `exp98_units.npz` |
