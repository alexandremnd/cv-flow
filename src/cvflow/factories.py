"""Pattern factories for parameter sweeps and randomized simulations.

A factory is any ``Callable[..., Pattern]``. The helpers below cover the two
common building blocks:

- :func:`squeezing_factory` decorates a pattern with a chosen
  squeezing setting on every prepared node.
- :func:`random_measurements` decorates an existing pattern by drawing fresh
  measurement parameters from a sampler.
"""
from collections.abc import Callable
from typing import TypeAlias

import numpy as np

from cvflow.command import CommandKind, Node
from cvflow.pattern import Pattern


MeasurementSampler: TypeAlias = Callable[
    [np.random.Generator, Node], tuple[float, float, float]
]
"""Draw $(\\alpha, \\beta, \\gamma)$ for a measurement on ``node`` using ``rng``."""


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

def uniform_sampler(
    alpha_bounds: tuple[float, float], beta_bounds: tuple[float, float]
) -> MeasurementSampler:
    """Sampler that draws $\\alpha, \\beta, \\gamma$ uniformly in ``[low, high]``

    Parameters
    ----------
    alpha_bounds : tuple[float, float]
        Bounds for the uniform distribution of the $\\alpha$ measurement parameter.
    beta_bounds : tuple[float, float]
        Bounds for the uniform distribution of the $\\beta$ measurement parameter.
    gamma_bounds : tuple[float, float]
        Bounds for the uniform distribution of the $\\gamma$ measurement parameter.

    Returns
    -------
    MeasurementSampler
            A sampler that can be passed to :func:`random_measurements` to draw
            measurement parameters uniformly from the specified bounds.
    """
    def _sample(rng: np.random.Generator, _node: Node) -> tuple[float, float, float]:
        sample = rng.uniform(alpha_bounds[0], alpha_bounds[1]), rng.uniform(beta_bounds[0], beta_bounds[1]), 0.0
        return sample
    return _sample
