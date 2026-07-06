# GPU results campaign — outcomes (living document)

One paragraph per experiment: outcome, where the data lives, which paper
claims changed. Statuses update as the campaign progresses; final version
committed when E1–E11 conclude. Data root: `/workspace/runs/paperprep/<exp>`
mirrored to `s3://anirud-dev/evodiv-divgen-fork/2026-07-04/paperprep-<exp>.tar.gz`.

## E1 — same-harness gradient baseline (553 GenEval prompts) — DONE
**Outcome (important nuance):** the repo's gradient optimizer via
`geneval_sdxl_white.yaml` optimizes **DPP + HPS** (its Tab-2 objective), not the
DINO+CLIP that breeding uses. Rescored by the identical metric code over its
553 image sets: DINO 0.772, DreamSim 0.460, LPIPS 0.752, CLIP 0.367, Vendi
3.99, DPP 0.999; 0.82 s/iter, 18.8 GB peak on B200. So on the same hardware and
scoring the two methods **split the metrics along their objectives**: breeding
leads DINO (+0.020, p<1e-19) and CLIP (+0.029, p<1e-77); the gradient run leads
the set-level diversity it directly maximizes (DPP, Vendi) and with it
DreamSim/LPIPS. Neither dominates. **Claim change:** the "ahead on 3 of 4"
headline is kept *only against the published Harrington numbers* (still true);
the same-harness row is presented as an objective-split, not a clean win, with
Table-1 bold = true column max (gradient's DreamSim is bolded). Independent
judges (E5) break the tie in breeding's favor — see E5. s/iter + peak-mem enter
the efficiency discussion. Data: `e1_gradient/`, `stats_results.json`.

### E5 (partial) — bred vs same-harness gradient on independent judges
Breeding beats the gradient run on **ImageReward** (0.671 vs 0.583, p<1e-2,
paired) and ties **PickScore** (22.54 vs 22.53, n.s.); the gradient run leads
only **HPSv2.1** (0.286 vs 0.267) — the metric it directly optimized. On
preference models neither method's breeding objective targeted, breeding is
ahead-or-even. Rand/CMA-DCT judging queued behind their runs.

## (historical) E1 setup note
The repository's own optimizer (`configs/geneval_sdxl_white.yaml`: DPP+HPS,
lr 6, 150 iters, B=4) over all 553 prompts with timing + peak-memory
instrumentation. Solo-window timing (prompts 0–9, GPU otherwise idle) is the
paper's s/iter number. Data: `e1_gradient/`. Paper: adds the same-hardware
gradient row to Table 1 (via the E5 unified re-score), plus measured s/iter
and peak memory to the efficiency discussion.

## E2 — matched-compute random-search control — DONE (256 prompts)
**Outcome:** random search at the matched per-prompt render budget reaches
DINO 0.683 / DreamSim 0.319 / LPIPS 0.683 / CLIP 0.359 / Vendi 2.21 —
decisively **below** breeding (0.790 / 0.437 / 0.745 / 0.393 / 2.85) on every
axis. Paired Wilcoxon over the 256-prompt intersection with `geneval_merged`:
all five metrics significant at p<10⁻²⁵ (Holm-corrected; DINO gap +0.098).
Blind luck at equal compute does not reach the bred result — structured
selection is doing real work. Data: `e2_random/`, `stats_results.json`.
Paper: activates the "Random search (matched)" row in Table 1 and the
"is it selection, or is the landscape just easy?" paragraph. Claim unchanged
(breeding > random), now with a hard significance number behind it.

### (historical) E2 setup note — capped at 256 prompts
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

## E3 — CMA-DCT at 553 prompts + exact re-scoring — DONE
**Outcome:** 1024-dim CMA-ES (8×8 low-freq DCT block/channel, 64× fewer params)
over 553 prompts, exact-rescored: DINO 0.767, DreamSim 0.420, LPIPS **0.842**,
CLIP **0.401**, Vendi 2.89, DPP 0.884 — the **highest CLIP and LPIPS of any
method**, within 0.023 DINO of full-space breeding.
**Decision rule applied:** CMA-DCT is significantly *below* breeding on DINO
(−0.023, paired Wilcoxon p<10⁻³³ at n=553), so it is **NOT** promoted to a
headline DINO win. Kept the "matches at 64× fewer dimensions" framing — which
is if anything stronger than expected (it leads on CLIP+LPIPS at 64× fewer
params). Table-1 bold = column max: CMA-DCT bolded on CLIP+LPIPS, breeding on
DINO, gradient on DreamSim.
**E10a (covariance dump):** honest finding — the genome is already low-freq
restricted; within the 8×8 block the adapted variance is ~uniform (lowest
third holds 23%, ≈ its 23% coefficient share), i.e. search uses the whole
low-freq block, not just the lowest modes. Claim corrected from "concentrates
in low freq" (would be an overclaim vs the uniform baseline) to "exploits the
whole low-freq block." Figure: `cma_spectrum.png`. Data: `e3_cmadct/` (incl.
per-prompt `cma_cov.npz`), `stats_results.json`.

## (queued) E3 setup note
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

## Tail experiments (E6–E9, E11) — DONE
- **E6 (HPS-floored DPP, 40 prompts):** breeding the paper's Tab-2 objective
  pair (DPP set-diversity + HPS floor) without a gradient lifts raw DPP log-det
  2.22→2.59 at HPS 0.307. Non-differentiable set objective enters selection
  directly. Data: `e6_dpp_hps/`.
- **E7 (DPG-Bench subset, 40 dense prompts):** DINO 0.618→0.738 at CLIP 0.412 —
  the recipe is not GenEval-specific; it generalizes to dense, paragraph-length
  captions. Data: `e7_dpg/`.
- **E8 (Vendi long budget, 20 prompts × 150 gens):** bred Vendi 2.25→3.34,
  plateauing ~gen 48 — a longer selection budget *does* keep buying the metric
  the source paper reports as saturating, up to a plateau. Figure
  `vendi_budget.png`. Data: `e8_vendi/`.
- **E9 (LPIPS probe, 20 prompts):** breeding LPIPS directly reaches **0.968**
  (vs 0.745 under DINO selection) at CLIP 0.342 — our one behind-the-gradient
  metric in Table 1 is a choice of objective, not a limit of the search.
  **Claim strengthened:** the LPIPS deficit is fully selectable. Data:
  `e9_lpips/`.
- **E11 (seed variance, 20 prompts × 3 seeds):** per-metric std across seeds is
  0.004 (DINO) / 0.002 (CLIP) — an order of magnitude below every gap discussed
  in the paper. Table-1 numbers are stable. Data: `e11_seeds/`.

## E10b — transfer band decomposition — DONE
**Outcome:** SDXL-bred noise replayed zero-shot through PixArt lifts DINO
diversity **+26.3%** over i.i.d.; band-limiting the transferred delta shows the
effect is **almost entirely low-frequency** — low-band-only recovers +25.6%
(97% of the full effect), high-band-only only +1.6%. Clean confirmation of the
source paper's spectral story: what transfers across models is the
low-frequency structure of the bred noise, not model-specific detail. (Two
bugs fixed en route: a log KeyError, and objective weights left at their 0.0
default that made the first run a no-op — both caught before integration.)
Data: `e10_transfer_bands/`. Paper: activates the band decomposition in the
"bred noise transfers across models" paragraph.
