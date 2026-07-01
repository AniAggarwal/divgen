"""Genetic-programming noise generator (requested idea 2).

Instead of evolving the raw noise tensor (as in ``genome.py``), we evolve a
small *program* -- an expression tree -- that *generates* each noise latent
from primitive operations. The genotype is a compact, human-readable recipe;
the phenotype is the noise field it renders. This is attractive for three
reasons:

  * It bakes the paper's frequency findings into the primitive set (pink
    filtering, low-frequency oriented bases), so the search operates in a space
    where "more low-frequency structure" is a single mutation away.
  * Programs are far lower-dimensional than a 4x64x64 latent, so crossover and
    mutation are more meaningful (subtree swaps recombine *structure*).
  * A discovered program is interpretable and transferable -- e.g. "pink(alpha)
    + 0.3 * lowfreq_grating(k, theta)" -- unlike an opaque noise tensor.

Grammar (all nodes return a (C,H,W) field, later chi_d-repaired):
    terminals:
        white(seed)                 i.i.d. Gaussian
        pink(alpha)                 1/(1+f)^alpha filtered Gaussian
        grating(k, theta, phase)    low-frequency oriented sinusoid (per channel)
    functions:
        add(a, b)                   a + b
        mix(a, b, w)                (1-w)*a + w*b
        spectral(a, alpha)          re-colour a with 1/(1+f)^alpha

Operators: point mutation (perturb a node's numeric params or retype a
terminal), subtree mutation (replace a random subtree with a fresh one), and
subtree crossover (swap random subtrees between two parents). Bloat is bounded
by a max depth.

This module is deliberately self-contained: it reuses the NSGA-II survival /
tournament primitives and the FitnessEvaluator, but keeps its own tree
representation and evolve loop so the tensor genome stays simple.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import torch

from evodiv.genome import repair, spectral_color, GenomeSpec
from evodiv.nsga2 import EvoResult, binary_tournament, rank_and_crowding, survival


TERMINALS = ["white", "pink", "grating"]
FUNCTIONS = ["add", "mix", "spectral"]


@dataclass
class Node:
    op: str
    params: dict
    children: List["Node"]

    def clone(self) -> "Node":
        return Node(self.op, dict(self.params), [c.clone() for c in self.children])

    def size(self) -> int:
        return 1 + sum(c.size() for c in self.children)

    def to_str(self) -> str:
        if self.op == "white":
            return f"white(s={self.params['seed']})"
        if self.op == "pink":
            return f"pink(a={self.params['alpha']:.2f})"
        if self.op == "grating":
            return (f"grating(k={self.params['k']:.1f},th={self.params['theta']:.2f},"
                    f"ph={self.params['phase']:.2f})")
        if self.op == "mix":
            return f"mix({self.children[0].to_str()},{self.children[1].to_str()},w={self.params['w']:.2f})"
        if self.op == "spectral":
            return f"spectral({self.children[0].to_str()},a={self.params['alpha']:.2f})"
        return f"add({self.children[0].to_str()},{self.children[1].to_str()})"


# --------------------------------------------------------------------------- #
# Random program construction                                                 #
# --------------------------------------------------------------------------- #


def _rand(gen: torch.Generator) -> float:
    return torch.rand(1, generator=gen, device=gen.device).item()


def _rand_terminal(gen: torch.Generator) -> Node:
    op = TERMINALS[int(_rand(gen) * len(TERMINALS)) % len(TERMINALS)]
    if op == "white":
        return Node("white", {"seed": int(_rand(gen) * 1e6)}, [])
    if op == "pink":
        return Node("pink", {"alpha": 0.1 + 0.9 * _rand(gen)}, [])
    return Node("grating", {"k": 1 + 6 * _rand(gen), "theta": math.pi * _rand(gen),
                            "phase": 2 * math.pi * _rand(gen)}, [])


def random_tree(gen: torch.Generator, max_depth: int, depth: int = 0) -> Node:
    if depth >= max_depth or _rand(gen) < 0.4:
        return _rand_terminal(gen)
    op = FUNCTIONS[int(_rand(gen) * len(FUNCTIONS)) % len(FUNCTIONS)]
    if op == "spectral":
        return Node("spectral", {"alpha": 0.1 + 0.9 * _rand(gen)},
                    [random_tree(gen, max_depth, depth + 1)])
    if op == "mix":
        return Node("mix", {"w": _rand(gen)},
                    [random_tree(gen, max_depth, depth + 1), random_tree(gen, max_depth, depth + 1)])
    return Node("add", {}, [random_tree(gen, max_depth, depth + 1),
                            random_tree(gen, max_depth, depth + 1)])


# --------------------------------------------------------------------------- #
# Program evaluation -> noise field                                            #
# --------------------------------------------------------------------------- #


def _grating(spec: GenomeSpec, k: float, theta: float, phase: float, device) -> torch.Tensor:
    ys = torch.linspace(0, 2 * math.pi, spec.height, device=device).view(spec.height, 1)
    xs = torch.linspace(0, 2 * math.pi, spec.width, device=device).view(1, spec.width)
    grid = torch.sin(k * (math.cos(theta) * xs + math.sin(theta) * ys) + phase)
    return grid.unsqueeze(0).expand(spec.channels, spec.height, spec.width).contiguous()


def eval_tree(node: Node, spec: GenomeSpec, device, base_seed: int) -> torch.Tensor:
    """Render a program to a (C,H,W) field (pre-repair)."""
    if node.op == "white":
        g = torch.Generator(device=device).manual_seed((base_seed * 1000003 + node.params["seed"]) % (2**31))
        return torch.randn(spec.channels, spec.height, spec.width, device=device, generator=g)
    if node.op == "pink":
        g = torch.Generator(device=device).manual_seed((base_seed * 1000003 + 7) % (2**31))
        w = torch.randn(spec.channels, spec.height, spec.width, device=device, generator=g)
        return spectral_color(w, node.params["alpha"])
    if node.op == "grating":
        return _grating(spec, node.params["k"], node.params["theta"], node.params["phase"], device)
    if node.op == "add":
        return eval_tree(node.children[0], spec, device, base_seed) + \
               eval_tree(node.children[1], spec, device, base_seed)
    if node.op == "mix":
        w = node.params["w"]
        return (1 - w) * eval_tree(node.children[0], spec, device, base_seed) + \
               w * eval_tree(node.children[1], spec, device, base_seed)
    if node.op == "spectral":
        return spectral_color(eval_tree(node.children[0], spec, device, base_seed), node.params["alpha"])
    raise ValueError(node.op)


# --------------------------------------------------------------------------- #
# GP genome: a set of B programs                                              #
# --------------------------------------------------------------------------- #


class GPGenome:
    """A set genome whose B latents are each produced by an expression tree."""

    def __init__(self, trees: List[Node]):
        self.trees = trees  # length B

    def clone(self) -> "GPGenome":
        return GPGenome([t.clone() for t in self.trees])

    def render_latents(self, spec: GenomeSpec, device) -> torch.Tensor:
        lat = torch.stack([
            repair(eval_tree(t, spec, device, base_seed=i)) for i, t in enumerate(self.trees)
        ], 0)
        return lat  # (B, C, H, W)


def _all_nodes(node: Node) -> List[Node]:
    out = [node]
    for c in node.children:
        out.extend(_all_nodes(c))
    return out


def mutate_tree(tree: Node, gen: torch.Generator, spec_max_depth: int, p_point: float = 0.5) -> Node:
    t = tree.clone()
    nodes = _all_nodes(t)
    target = nodes[int(_rand(gen) * len(nodes)) % len(nodes)]
    if _rand(gen) < p_point and target.params:
        # point mutation: jitter a numeric parameter
        key = list(target.params.keys())[int(_rand(gen) * len(target.params)) % len(target.params)]
        if key == "seed":
            target.params[key] = int(_rand(gen) * 1e6)
        elif key in ("alpha", "w"):
            target.params[key] = float(min(1.0, max(0.05, target.params[key] + 0.3 * (_rand(gen) - 0.5))))
        elif key == "k":
            target.params[key] = float(min(8.0, max(1.0, target.params[key] + 2 * (_rand(gen) - 0.5))))
        else:  # theta, phase
            target.params[key] = float(target.params[key] + math.pi * (_rand(gen) - 0.5))
        return t
    # subtree mutation: replace target with a fresh subtree (in-place via rebuild)
    fresh = random_tree(gen, spec_max_depth)
    return _replace_node(t, target, fresh)


def _replace_node(root: Node, target: Node, repl: Node) -> Node:
    if root is target:
        return repl
    root.children = [_replace_node(c, target, repl) for c in root.children]
    return root


def crossover_trees(a: Node, b: Node, gen: torch.Generator) -> Node:
    child = a.clone()
    nodes_c = _all_nodes(child)
    nodes_b = _all_nodes(b)
    pt_c = nodes_c[int(_rand(gen) * len(nodes_c)) % len(nodes_c)]
    pt_b = nodes_b[int(_rand(gen) * len(nodes_b)) % len(nodes_b)].clone()
    return _replace_node(child, pt_c, pt_b)


# --------------------------------------------------------------------------- #
# GP population + NSGA-II loop (mirrors nsga2.evolve, tree operators)          #
# --------------------------------------------------------------------------- #


class GPPopulation:
    """Adapter exposing the same surface FitnessEvaluator / nsga2 expect.

    ``latents`` is materialised on demand from the trees so the evaluator's
    batched renderer works unchanged.
    """

    def __init__(self, genomes: List[GPGenome], spec: GenomeSpec, device):
        self.genomes = genomes
        self.spec = spec
        self.device = device
        self.F: Optional[torch.Tensor] = None
        self.raw: Optional[list] = None

    @property
    def size(self):
        return len(self.genomes)

    @property
    def latents(self) -> torch.Tensor:
        return torch.stack([g.render_latents(self.spec, self.device) for g in self.genomes], 0)

    def select(self, idx) -> "GPPopulation":
        p = GPPopulation([self.genomes[i].clone() for i in idx.tolist()], self.spec, self.device)
        if self.F is not None:
            p.F = self.F[idx].clone()
        if self.raw is not None:
            p.raw = [self.raw[i] for i in idx.tolist()]
        return p

    @staticmethod
    def concat(a: "GPPopulation", b: "GPPopulation") -> "GPPopulation":
        p = GPPopulation([g.clone() for g in a.genomes] + [g.clone() for g in b.genomes], a.spec, a.device)
        if a.F is not None and b.F is not None:
            p.F = torch.cat([a.F, b.F], 0)
        if a.raw is not None and b.raw is not None:
            p.raw = list(a.raw) + list(b.raw)
        return p

    def sigma(self):
        return torch.zeros(1)   # GP has no ES step size; keep the logging interface happy

    def rate(self):
        return torch.zeros(1)


def evolve_gp(spec, evaluator, prompt, pop_size, n_generations, device, generator,
              max_depth: int = 4, log_fn: Optional[Callable[[str], None]] = None) -> EvoResult:
    log = log_fn or (lambda s: None)
    genomes = [GPGenome([random_tree(generator, max_depth) for _ in range(spec.set_size)])
               for _ in range(pop_size)]
    pop = GPPopulation(genomes, spec, device)
    evaluator.evaluate(pop, prompt, log_all=True)
    history = []

    def record(gen):
        div = [r["_diversity"] for r in pop.raw]
        qual = [r["_quality"] for r in pop.raw]
        rank, _, _ = rank_and_crowding(pop.F)
        rec = {"gen": gen, "best_diversity": max(div), "mean_diversity": sum(div) / len(div),
               "best_quality": max(qual), "mean_quality": sum(qual) / len(qual),
               "pareto_size": int((rank == 0).sum().item()), "n_renders": evaluator.n_renders}
        best_i = max(range(len(pop.raw)), key=lambda i: pop.raw[i]["_diversity"])
        for k, v in pop.raw[best_i].items():
            if not k.startswith("_"):
                rec[f"best/{k}"] = v
        rec["best_program"] = pop.genomes[best_i].trees[0].to_str()
        history.append(rec)
        log(f"[GP gen {gen:03d}] div(best={rec['best_diversity']:.4f} mean={rec['mean_diversity']:.4f}) "
            f"qual(best={rec['best_quality']:.4f}) pareto={rec['pareto_size']} renders={rec['n_renders']}")

    record(0)
    for gen in range(1, n_generations + 1):
        rank, cd, _ = rank_and_crowding(pop.F)
        pa = binary_tournament(rank, cd, pop_size, generator)
        pb = binary_tournament(rank, cd, pop_size, generator)
        children = []
        for i in range(pop_size):
            ga, gb = pop.genomes[pa[i].item()], pop.genomes[pb[i].item()]
            trees = []
            for s in range(spec.set_size):
                if _rand(generator) < 0.9:
                    child_tree = crossover_trees(ga.trees[s], gb.trees[s], generator)
                else:
                    child_tree = ga.trees[s].clone()
                child_tree = mutate_tree(child_tree, generator, max_depth)
                trees.append(child_tree)
            children.append(GPGenome(trees))
        off = GPPopulation(children, spec, device)
        evaluator.evaluate(off, prompt, log_all=(gen == n_generations))
        pop = survival(GPPopulation.concat(pop, off), pop_size)
        record(gen)

    evaluator.evaluate(pop, prompt, log_all=True)
    return EvoResult(population=pop, history=history)
