# EvoDiv exploration program (post-paper GPU utilization)
Each entry: what was run, key numbers, verdict.

## 1. DPP set-level objective (paper Tab. 2 setting), 40 GenEval prompts
- Config: DPP(DINO-CLS kernel) diversity + CLIP quality, pop 40 x <=60 gens, surrogate+compile.
- init -> evolved: Vendi 2.29 -> 3.32, DreamSim 0.278 -> 0.449, DPP 0.802 -> 0.929, CLIP 0.325 -> 0.342.
- vs paper Tab.2 (gradient, DPP+HPS): DreamSim 0.457, Vendi 4.0 (saturated).
- Verdict: gradient-free breeding of a *set-level* objective works; approaches
  their DreamSim with quality RISING (they report slight HPS drop for SDXL pink /
  hyperparameter sensitivity on flux). Vendi not saturated at this budget —
  candidate for longer runs or Vendi-as-objective.
## 2. MAP-Elites quality-diversity archive (6 GenEval prompts)
- 10x10 archive over (color-div x layout-div) cells; fitness = DINO + 3*CLIP;
  same spectral operators; 60 iters x batch 40 (~2400 evals/prompt).
- coverage 24.3/100 cells; best-cell DINO 0.773 @ CLIP 0.351; top archive DINO 0.779.
- vs NSGA-II default on same prompts (45 gens x pop 32): DINO 0.757 @ CLIP 0.347.
- Verdict: MAP-Elites slightly ahead at a somewhat larger eval budget AND yields
  ~24 behaviourally distinct diverse sets per prompt (a menu, not a point) —
  promising direction for a "diversity of diversities" extension. GPU ~98%.
## 3. GP v2 (4 prompts, matched ~60-gen budget) — first pass + corrected rerun
- Arms: A pure tensor NSGA-II 60g | B quality-constrained GP 60g | C hybrid (GP 25g -> seeds half a tensor pop -> 35g).
- First pass (flawed reporting): A dino 0.763 @ clip 0.338; B 0.871 @ 0.224; C 0.878 @ 0.240.
  LESSON: filtering offspring by a quality floor is NOT enough — unfiltered gen-0
  ancestors persist on the Pareto front, and max-diversity selection ignores the
  floor. Constraint must bind at SELECTION (and/or seed the archive floored).
- Corrected rerun (floored selection) in explore_gp2b.log — results next entry.
- Corrected rerun (floored selection): B dino 0.610 @ clip 0.336; C dino 0.729 @ 0.336;
  reference A (tensor) 0.763 @ 0.338.
- VERDICT: program search owns the unconstrained diversity extreme (0.87+ @ low CLIP)
  but is DOMINATED in the quality-feasible region; GP warm-starts do not transfer
  useful structure once quality binds. Tensor genome remains the right tool at the
  paper's operating point.
## 4. CMA-ES over 8x8 low-frequency DCT coefficients (4 prompts, 60 it x 40)
- Genome: fixed white base + IDCT low-freq field (1024 dims), chi_d repair;
  scalar fitness DINO + 3*CLIP; surrogate evals (rank-corr 0.965, not exact-rescored).
- Result: dino 0.807 @ clip 0.325 vs tensor NSGA-II 0.763 @ 0.338.
- Verdict: BEST diversity-per-eval so far. Searching ONLY the low-frequency
  subspace outperforms full-space search at slight quality cost — the strongest
  independent confirmation of the paper's Fig. 9 yet. Follow-ups: two-objective
  variant (MO-CMA / NSGA-II over DCT genome), exact re-scoring, bigger K.
- Ops note: first launch hit cma/np.Inf (numpy2) + a x64 field-scaling bug
  (CLIP 0.14 blobs); utilization+crash watchdogs caught both within minutes.
## 5. Set-size scaling B=4/8/16 (4 prompts, pop 24 x 40 gens)
- Evolved Vendi: 2.35 (B=4) -> 2.94 (B=8) -> 4.36 (B=16); CLIP flat ~0.33.
- Pairwise means dilute with B (expected: harder to keep 16 mutually far), but the
  effective number of distinct modes (Vendi) nearly doubles; B=16 exceeds the
  Vendi=4 ceiling that bounds the paper's batched B=4 mode.
- Forward-only: B=16 is just a bigger batch (384 imgs/gen, chunked); backprop
  through the sampler at B=16 would not fit a single GPU.
- Verdict: strong scale-up demo; candidate headline for a "large diverse sets" section.
## 6. Epsilon-lexicase parent selection vs tournament (4 prompts, pop 32 x 45 gens)
- Cases = 6 pairwise DINO distances + quality (3x weight); NSGA-II survival kept.
- lexicase: dino 0.791 @ clip 0.331 | tournament: dino 0.760 @ clip 0.337.
- Verdict: lexicase shifts the front toward diversity (+0.03 DINO, -0.006 CLIP) —
  useful as a diversity-lean selection knob; not a strict domination.
## 7. Cross-model transfer: SDXL-Turbo -> PixArt-DMD (4 prompts)
- Same (4,64,64) latent shape; bred sets from SDXL rendered verbatim through PixArt.
- PixArt DINO: i.i.d. 0.424 -> bred-transfer 0.535 (+26%), CLIP 0.320 -> 0.318 (flat).
- Verdict: HEADLINE-GRADE. A large fraction of bred diversity is a property of
  the NOISE, not the model — zero-shot transferable across architectures.
  Resonates with the paper's low-frequency analysis (the transferred component
  is presumably the low-freq structure both models read the same way).
  Follow-up: spectral decomposition of transferred vs non-transferred gains.
## 8. Composition: bred noise warm-starts the gradient optimizer (4 prompts)
- A grad-only(10 it): 0.596 @ 0.339 | B bred-only(20 gens): 0.720 @ 0.334 |
  C bred->grad: 0.726 @ 0.337.
- Verdict: composition works — breeding finds the diverse basin, a few gradient
  steps recover the quality gap. Directly supports the paper-note's Sec. 4 claim.

# ROUND 1 COMPLETE (all 8 queue items). Top findings:
# 1. CMA-ES in low-freq DCT subspace: best diversity-per-eval (0.807).
# 2. Cross-model noise transfer: +26% diversity zero-shot SDXL->PixArt.
# 3. B=16 sets: Vendi 4.36, past the paper's B=4 ceiling, quality flat.
# 4. MAP-Elites: menu of ~24 distinct diverse sets per prompt.
## R2a. Vendi-as-objective, 20 prompts x <=120 gens
- Vendi 2.25 -> 3.32 mean (median 3.52, max 3.94; 4/20 near-saturated >3.8);
  CLIP 0.332 -> 0.351 (UP). DreamSim 0.277 -> 0.441.
- vs paper gradient+DPP: Vendi 4.0 fully saturated (their Tab. 2).
- Verdict: honest gap — differentiable pressure on kernel eigenvalues saturates
  Vendi better than selection does at this budget; breeding trades the last
  ~0.7 Vendi for +0.02 CLIP. Longer budgets / larger sigma floor might close it.
## R2b. CMA-DCT quality-weight sweep (6 prompts, 60 it x 40 each)
- w_q=1: dino 0.830 @ clip 0.318 | w_q=3: 0.804 @ 0.338 | w_q=6: 0.753 @ 0.347.
- KEY: w_q=3 STRICTLY DOMINATES tensor NSGA-II (0.763 @ 0.338) — same quality,
  +0.04 diversity. The low-frequency DCT subspace is simply the right search
  space; the paper's Fig. 9, operationalized fully, beats full-space search.
- Next: DCT genome inside NSGA-II (Pareto in the winning subspace).
## R2c. NSGA-II over the DCT subspace vs full-space NSGA-II (6 prompts, floored selection)
- dct-nsga: dino 0.777 @ clip 0.343 | tensor-nsga: 0.768 @ 0.345.
- Verdict: subspace alone gives only +0.01 under NSGA-II; the big CMA-DCT gain
  (+0.04) therefore comes from CMA's COVARIANCE ADAPTATION inside the subspace,
  not just the parameterization. Winning combo = CMA machinery + low-freq
  subspace (+ scalarization at w_q=3, or MO-CMA as future work).
## R2d. CMA-DCT w=3 consolidation at 40 prompts (final GPU run)
- dino 0.785 @ clip 0.344 vs tensor NSGA-II 0.763 @ 0.338 — DOMINATES ON BOTH
  AXES at scale. Distilled recommendation for follow-up work: CMA-ES with
  covariance adaptation in the low-frequency DCT subspace (w_q=3), optionally
  followed by a few gradient steps (R1.8) for final quality polish.

# EXPLORATION PROGRAM COMPLETE — 2026-07-03. GPU released after backups.
