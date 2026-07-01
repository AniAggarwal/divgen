# EvoDiv — Evolutionary Noise Optimization for Collapse Recovery

An **evolutionary (NSGA-II) reformulation** of
[*It's Never Too Late: Noise Optimization for Collapse Recovery in Trained
Diffusion Models*](https://github.com/anneharrington/divgen) (Harrington,
Koepke et al., CVPR 2026), built by transplanting the multi-objective genetic
algorithm from [**ECAD** — *Evolutionary Caching to Accelerate Your
Off-the-Shelf Diffusion Model*](https://research.aniaggarwal.com/ecad) (ICLR
2026).

The base paper recovers from mode collapse by **backpropagating through the
frozen sampler** to optimize the initial noise for a diversity objective.
EvoDiv instead **evolves the noise gradient-free**, treating *quality* and
*diversity* as two objectives on a Pareto front — the exact shape ECAD uses for
its *quality* vs *speedup* front.

## Why evolutionary?

| | ECAD (ICLR 2026) | Never-Too-Late (CVPR 2026) | **EvoDiv (this work)** |
|---|---|---|---|
| Engine | NSGA-II GA | gradient descent on noise | **NSGA-II GA** |
| Chromosome | binary cache decisions | — | **a set of `B` noise latents** (+ self-adaptive genes) |
| Objective 1 | quality (ImageReward) | quality (CLIP/HPSv2) | **quality (CLIP/HPSv2)** |
| Objective 2 | speedup / MACs | diversity (DINO/DPP/Vendi) | **set diversity (DINO/DPP/Vendi)** |
| Search | forward-only | backprop through sampler | **forward-only** |

Concrete advantages of the evolutionary formulation:

1. **No backprop through the diffusion model.** Gradient noise-optimization must
   differentiate through the entire frozen sampler — the memory wall on a single
   GPU for a 10B model like FLUX. EvoDiv only ever runs *forward* passes.
2. **Non-differentiable objectives are first-class.** True DPP log-det and Vendi
   scores, black-box rewards, or discrete metrics can drive the search directly.
3. **The Pareto front is the output.** You get an explicit quality–diversity
   trade-off curve per prompt, not a single point tied to loss weights.
4. **Generation 0 *is* the paper's i.i.d. baseline**, so every run reports the
   evolutionary lift against it directly.

## Method

**Genome.** Diversity is a *set-level* property (the paper's Tab. 1/2 metrics
are averaged over a set of 4 images), so the unit of selection is a whole
**set of `B` noise latents** for one prompt, not a single noise. A genome also
carries two self-adaptive strategy genes (below).

**Objectives** (both reuse divgen's own code, so numbers are comparable):
- *quality* = mean per-image **CLIPScore** (Tab. 1) or **HPSv2** (Tab. 2). ImageReward is **not** used.
- *diversity* = a set-level statistic: patchwise/CLS **DINOv2** (Tab. 1), **DPP**, or **Vendi** (Tab. 2).
Held-out metrics (DreamSim, LPIPS, Color, tiny-L2, SSCD-Vendi) are logged but
never optimized, to show generalization (the paper's claim).

**Operators**, each grounded in a paper finding:
- **Spectral, low-frequency-biased mutation.** The paper's Fig. 9 shows
  optimization acts mostly on the low third of the spectrum. Mutations are
  colored with a `1/(1+f)^beta` filter so proposals move low frequencies more —
  the same filter behind their pink-noise init.
- **χ_d-norm repair.** Their soft regularizer `K(ε)` pins `‖ε‖` to the χ_d
  radius; we enforce it *exactly* by standardizing each latent (mean 0, std 1)
  after every operator, so noises stay on the Gaussian prior's high-density shell.
- **Set-level uniform crossover.** Each of a child's `B` noise slots is inherited
  whole from either parent — recombination in the space of *sets*.
- **Self-adaptive mutation rate + step (idea 3).** ECAD's Table 9 shows 1%
  mutation stalls and 15% is too slow, with the sweet spot unknown. Each genome
  carries its own rate and step, evolved by log-normal self-adaptation (as in
  Evolution Strategies) — the population discovers its own schedule and anneals
  exploration→exploitation on its own.

**Selection.** Standard NSGA-II: pymoo's fast non-dominated sorting (the routine
behind ECAD's `NSGA2`) + crowding-distance survival + binary-tournament
selection, in a (μ+λ) loop.

**Efficiency.** The whole population shares one prompt, so every genome's
latents render in a *single batched forward pass* (chunked to a memory cap). On
one B200, SDXL-Turbo runs ~12 ms/image steady-state, so a pop-40 × 60-generation
run over a prompt is ~90 s.

## Extensions (the three requested ideas)

1. **Island model with prompt-typed subpopulations** (`islands.py`). Landscape
   paintings, 5-word prompts, and GPT-generated prompts each get their own
   island; top genomes migrate between islands (ring) every N generations.
   Because a genome is prompt-agnostic noise, a migrant is re-scored on the
   destination's prompts — so a genome that keeps winning after migrating has
   diversity that *generalizes across prompt types*. Reported at the end.
2. **Genetic programming** (`gp_noise.py`). Evolve a *program* (a frequency-op
   expression tree: `white`, `pink(α)`, `grating(k,θ,φ)`, `add`, `mix`,
   `spectral`) that *generates* each noise, instead of the raw tensor. Compact,
   interpretable, and it bakes the paper's frequency findings into the grammar.
3. **Self-adaptive mutation** — the default operator, see above.

## Running

```bash
# one prompt (paper Tab. 1 setting: DINOv2 diversity + CLIP quality)
python -m evodiv.run --config configs/evo_sdxl_single.yaml --task.prompt "A photo of a cat"

# GenEval sweep
python -m evodiv.run --config configs/evo_sdxl_geneval.yaml

# set-level objective (Vendi story): DPP diversity + CLIP quality
python -m evodiv.run --config configs/evo_sdxl_dpp.yaml

# island model (cross-prompt generalization)
python -m evodiv.run --config configs/evo_sdxl_islands.yaml

# genetic-programming variant
python -m evodiv.run --config configs/evo_sdxl_gp.yaml

# background launcher (nohup, like ECAD)
./scripts/run_single_gpu.sh configs/evo_sdxl_geneval.yaml

# mutation-rate ablation (fixed 1% / 10% / 15% vs self-adaptive; ECAD Table 9)
./scripts/run_mutation_ablation.sh

# aggregate + plot a finished run
python -m evodiv.analyze --run_dir runs/geneval_main/evodiv/sdxl-turbo/geneval --out report
```

Every `--evolution.*` / `--rewards.*` / `--diversity.*` field is a CLI flag
(inherited from divgen's config system) and can override the YAML.

## Scaling up

The single-B200 runs here are a proof of convergence. To scale:
- **More prompts / full GenEval:** drop `--task.prompt_end_index`; runs are
  embarrassingly parallel across prompts — shard by `--task.prompt_start_index /
  _end_index` across GPUs.
- **Bigger populations / generations:** `--evolution.pop_size`,
  `--evolution.n_generations`. Memory scales with `pop_size × set_size`; raise
  `--evolution.render_chunk` on larger cards.
- **Larger models (FLUX.1-schnell, SANA-Sprint):** wire the latent packing in
  `evodiv/run.py:_latent_spec` (the base repo's `generate_latents(...,
  flux_schnell_pack=True)` already handles FLUX packing). Being forward-only,
  EvoDiv fits FLUX where gradient noise-optimization would OOM.
- **HPSv2 quality (Tab. 2):** `--rewards.hps.enable --no-rewards.clip.enable`
  (needs the HPSv2 checkpoint downloaded).

See `EVODIV_RESULTS.md` at the repo root for measured numbers and plots.
