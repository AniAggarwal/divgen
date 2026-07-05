# GPU results campaign — outcomes (living document)

One paragraph per experiment: outcome, where the data lives, which paper
claims changed. Statuses update as the campaign progresses; final version
committed when E1–E11 conclude. Data root: `/workspace/runs/paperprep/<exp>`
mirrored to `s3://anirud-dev/evodiv-divgen-fork/2026-07-04/paperprep-<exp>.tar.gz`.

## E1 — same-harness gradient baseline (553 GenEval prompts) — RUNNING
The repository's own optimizer (`configs/geneval_sdxl_white.yaml`: DPP+HPS,
lr 6, 150 iters, B=4) over all 553 prompts with timing + peak-memory
instrumentation. Solo-window timing (prompts 0–9, GPU otherwise idle) is the
paper's s/iter number. Data: `e1_gradient/`. Paper: adds the same-hardware
gradient row to Table 1 (via the E5 unified re-score), plus measured s/iter
and peak memory to the efficiency discussion.

## E2 — matched-compute random-search control — RUNNING (capped at 256 prompts)
Per-prompt render budgets extracted from the archived full552+ext+ext2
histories (median 21.1k renders/prompt; see `render_budget_by_prompt.json`).
Same selection rule, surrogate profile, and exact final scoring as breeding.
**Scope decision (2026-07-05, flagged for verification):** capped at the first
256 prompts rather than 553. At matched compute the breeding-vs-random gap is
large (~0.08 DINO), so 256 paired prompts give an overwhelmingly powered
Wilcoxon while costing ~1/3 the GPU time; the ~10 freed GPU-hours fund the
diverse night-queue experiments the user asked for. The paired test uses the
256-prompt intersection with `geneval_merged`. Resume to 553 anytime with
`EXPECT=553 E2_END=553 bash e2_random/run_e2.sh`. Paper: the "is it selection
or luck" row of Table 1.

## E3 — CMA-DCT at 553 prompts + exact re-scoring — QUEUED
The exploration winner scaled 40→553 prompts with the 40-prompt script's flaw
fixed (its "exact re-score" was dead code; winners were surrogate-scored).
Winner latents tracked, exact-rescored every 10 iters and at the end; final
covariance eigen-spectrum + coordinate variances dumped per prompt (feeds
E10a). Paper: Table-1 row promotion decided by the 553-prompt paired
Wilcoxon (stats.py, Holm-corrected).

## E4 — FLUX.1-schnell — PROBES DONE; BREEDING RUNNING
Latent packing implemented (`evodiv/run.py:flux_pack`, spatial genome packed
2×2 at the render boundary; verified byte-identical to the upstream packing).
Memory probes complete (`e4_flux/memory_table.json`): forward-only 37.9 GB
(B=4) / 44.4 GB (B=16) vs gradient 57.9 GB / 113.8 GB — the wall is the
*scaling* (+6.5 GB vs +55.9 GB when B goes 4→16). Breeding run over 60
GenEval prompts in progress (search at 1 step, init/best reported at 4 steps
with a matched i.i.d. 4-step reference). Paper: Sec. "Scaling to FLUX" +
Table 3 (measured, no longer an assertion).

## E5 — preference-model judging — WEIGHTS PREFETCHED
ImageReward + PickScore + HPSv2.1 over the final image sets of
bred/gradient/random/CMA-DCT, plus a unified re-score of all methods' saved
jpgs with the repo's own metric code (removes harness asymmetry for E1's
Table-1 row). Evaluation only — never bred objectives. Runs as each method's
images land.

## E6 — HPSv2-floored DPP — QUEUED (config smoke-tested)
## E7 — DPG-Bench subset (40 dense prompts) — QUEUED (config smoke-tested)
## E8 — Vendi long-budget probe (150 gens, 20 prompts) — QUEUED
## E9 — LPIPS probe (20 prompts) — QUEUED
## E10 — CMA covariance spectrum (from E3) + transfer band decomposition — QUEUED
## E11 — seed variance (20 prompts × 3 seeds) — QUEUED

## Scheduling / cost notes (2026-07-05)
Measured on the B200: the gradient baseline (E1) runs at ~120 s/prompt **solo
but only 23% GPU utilization** — latency-bound by its small B=4 batch and
per-iteration Python/optimizer gaps. Running E1 concurrently with E2 (the
random control) fills those idle cycles: E1 slows to ~180 s/prompt but E2
progresses in parallel, and the pair completes in ~21.6 h vs ~27.7 h serial
(~6 h / ~$25 saved), so **E1∥E2 run concurrently**. The heavier, higher-util
stages (E3, tail) run one-at-a-time; the IO-bound E5 judging sub-runs overlap
whatever compute stage is active. `pipeline_driver.sh` enforces this and never
lets the GPU idle until the night queue drains.

## Infrastructure notes
- `paper/full/GPU_PLAN.md` is a marked reconstruction (original never reached
  any remote; see its provenance note).
- Every paper number flows runs → `analysis/aggregate.py` →
  `paper_numbers.json` → `make_numbers_tex.py` → `numbers.tex`;
  `verify_tex_numbers.py` enforces no bare numerics in either tex. The
  archived-headline sanity gate (`aggregate.py --check`) reproduces the
  committed 3-pager exactly.
- Model caches moved to tmpfs (/dev/shm) after two disk-full incidents; disk
  now holds only code + run outputs.
