"""Pattern factories for parameter sweeps and randomized simulations.

A factory is any ``Callable[..., Pattern]``. The helpers below cover the two
common building blocks:

- :func:`squeezing_factory` decorates a pattern with a chosen
  squeezing setting on every prepared node.
- :func:`random_measurements` decorates an existing pattern by drawing fresh
  measurement parameters from a sampler.
"""
from abc import ABC, abstractmethod

import numpy as np

from cvflow.command import CommandKind, Node
from cvflow.pattern import Pattern


class MeasurementSampler(ABC):
    """Base class for measurement samplers.

    Subclasses must implement :meth:`__call__` to draw
    $(\\alpha, \\beta, \\gamma)$ for a given node.
    """

    @abstractmethod
    def __call__(self, rng: np.random.Generator, node: Node) -> tuple[float, float, float]:
        """Draw $(\\alpha, \\beta, \\gamma)$ for a measurement on ``node``."""


def squeezing_factory(
    pattern: Pattern,
    r: float,
    theta: float = np.pi / 2,
) -> None:
    """Update in-place a pattern with uniform squeezing on every ancilla node.

    Parameters
    ----------
    pattern : Pattern
        The base pattern to update in-place.
    r : float
        Squeezing ratio applied to every non-input node.
    theta : float, optional
        Squeezing angle. Defaults to $\\pi/2$ (p eigenstate).
    """
    squeezing_params = {n: (r, theta) for n in pattern.non_input_nodes}
    pattern.set_squeezing(squeezing_params)


def random_measurements(
    pattern: Pattern,
    *,
    sampler: MeasurementSampler,
    rng: np.random.Generator,
) -> None:
    """Redraw each ``M`` command's parameters from ``sampler`` in place.

    Mutates ``pattern`` and returns it (no clone, no revalidation) so the same
    pattern object can be reused across many shots in a tight loop. Call
    :meth:`Pattern.clone` first if you need to preserve the original parameters.
    """
    overrides = {
        cmd.node: sampler(rng, cmd.node) for cmd in pattern if cmd.kind == CommandKind.M
    }
    pattern.set_measurements(overrides)


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


def uniform_sampler(
    alpha_bounds: tuple[float, float], beta_bounds: tuple[float, float]
) -> UniformSampler:
    """Return a :class:`UniformSampler` with the given bounds."""
    return UniformSampler(alpha_bounds, beta_bounds)
