"""Factories for parameter sweeps and randomized simulations.

- :func:`uniform_squeezings` / :func:`uniform_squeezing_angles` produce the
  per-node dicts consumed by :class:`~cvflow.utils.run.RunConfig`.
- :func:`squeezing_factory` mutates an existing pattern in place (kept for
  one-off pattern tweaks; the sweep pipeline uses the dict factories instead).
- :func:`random_measurements` decorates an existing pattern by drawing fresh
  measurement parameters from a sampler.
"""
from abc import ABC, abstractmethod
from collections.abc import Iterable

import numpy as np

from cvflow.command import CommandKind, Node
from cvflow.pattern import Pattern


def uniform_squeezings(nodes: Iterable[Node], r: float) -> dict[int, float]:
    """Build a ``{node: r}`` map suitable for ``RunConfig(squeezings=...)``.

    Parameters
    ----------
    nodes : Iterable[Node]
        Nodes to apply squeezing to (typically ``graph.non_input_nodes``).
    r : float
        Squeezing ratio applied uniformly across ``nodes``.
    """
    return {int(n): float(r) for n in nodes}

def uniform_squeezing_angles(
    nodes: Iterable[Node],
    angle: float = float(np.pi / 2),
) -> dict[int, float]:
    """Build a ``{node: angle}`` map suitable for ``RunConfig(squeezing_angles=...)``.

    Parameters
    ----------
    nodes : Iterable[Node]
        Nodes to set an angle on (typically ``graph.non_input_nodes``).
    angle : float, optional
        Squeezing angle. Defaults to $\\pi/2$ (p eigenstate).
    """
    return {int(n): float(angle) for n in nodes}

# ========= Measurements ========
class MeasurementSampler(ABC):
    """Base class for measurement samplers.

    Subclasses must implement :meth:`__call__` to draw
    $(\\alpha, \\beta, \\gamma)$ for a given node.
    """
    @abstractmethod
    def __call__(self, rng: np.random.Generator, node: Node) -> tuple[float, float, float]:
        """Draw $(\\alpha, \\beta, \\gamma)$ for a measurement on ``node``."""

    def to_dict(self) -> dict:
        """Serialise this sampler to a JSON-friendly ``{"type", "params"}`` dict.

        The default implementation records the class name and an empty
        ``params`` mapping. Subclasses should override to include the init
        kwargs needed to rebuild an equivalent instance.
        """
        return {"type": type(self).__name__, "params": {}}

    def __str__(self) -> str:
        return "Abstract MeasurementSampler"

def random_measurements(
    pattern: Pattern,
    *,
    sampler: MeasurementSampler,
    rng: np.random.Generator,
) -> dict[int, tuple[float, float, float]]:
    """Redraw each ``M`` command's parameters from ``sampler`` in place.

    Mutates ``pattern`` and returns it (no clone, no revalidation) so the same
    pattern object can be reused across many shots in a tight loop. Call
    :meth:`Pattern.clone` first if you need to preserve the original parameters.

    Parameters
    ----------
    pattern : Pattern
        The pattern to update in-place.
    sampler : MeasurementSampler
        The sampler to draw new measurement parameters from.
    rng : np.random.Generator
        The random number generator to use for sampling.

    Returns
    -------
    dict[int, tuple[float, float, float]]
        A mapping from measured node to the new $(\\alpha, \\beta, \\gamma)$
        parameters drawn for that node.
    """
    overrides = {
        cmd.node: sampler(rng, cmd.node) for cmd in pattern if cmd.kind == CommandKind.M
    }
    pattern.set_measurements(overrides)
    return overrides

class UniformSampler(MeasurementSampler):
    """Sampler that draws $\\alpha, \\beta, \\gamma$ uniformly from bounded ranges.

    Parameters
    ----------
    alpha_bounds : tuple[float, float]
        ``(low, high)`` for the $\\alpha$ measurement parameter.
    beta_bounds : tuple[float, float]
        ``(low, high)`` for the $\\beta$ measurement parameter.
    """

    def __init__(
        self,
        alpha_bounds: tuple[float, float],
        beta_bounds: tuple[float, float],
    ) -> None:
        self.alpha_bounds = alpha_bounds
        self.beta_bounds = beta_bounds

    def __call__(self, rng: np.random.Generator, node: Node) -> tuple[float, float, float]:
        return (
            rng.uniform(*self.alpha_bounds),
            rng.uniform(*self.beta_bounds),
            0.0,
        )

    def to_dict(self) -> dict:
        return {
            "type": "UniformSampler",
            "params": {
                "alpha_bounds": list(self.alpha_bounds),
                "beta_bounds": list(self.beta_bounds),
            },
        }

    def __str__(self) -> str:
        return f"UniformSampler(alpha_bounds={self.alpha_bounds}, beta_bounds={self.beta_bounds})"

def uniform_sampler(
    alpha_bounds: tuple[float, float], beta_bounds: tuple[float, float]
) -> UniformSampler:
    """Return a :class:`UniformSampler` with the given bounds."""
    return UniformSampler(alpha_bounds, beta_bounds)
