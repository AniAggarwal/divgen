"""EvoDiv: evolutionary noise optimisation for collapse recovery.

An evolutionary (NSGA-II) reformulation of "It's Never Too Late: Noise
Optimization for Collapse Recovery" that reuses the ECAD multi-objective
genetic-algorithm framework. Quality and diversity are treated as two
objectives on a Pareto front; the initial noise is evolved gradient-free
instead of optimised by backprop.
"""

__all__ = [
    "genome",
    "fitness",
    "nsga2",
    "islands",
    "config",
    "io_utils",
]
