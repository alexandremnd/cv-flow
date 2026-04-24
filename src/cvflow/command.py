import dataclasses
import enum
from enum import Enum
from typing import ClassVar, Literal

Node = int

class CommandKind(Enum):
    """Tag for command kind."""

    N = enum.auto()
    M = enum.auto()
    E = enum.auto()
    X = enum.auto()
    Z = enum.auto()

@dataclasses.dataclass(repr=False)
class N():
    """Preparation command.

    Attributes
    ----------
    node : Node
        The node to which this command is applied.
    squeezing_ratio : float
        The ratio controlling the squeezing magnitude.
        Defaults to 10.0. Higher values increase the squeezing effect.
    squeezing_angle : float
        The angle in degrees controlling the direction of the squeezing
        transformation. Defaults to 0.0.
    """
    node: Node
    squeezing_ratio: float = 10.0
    squeezing_angle: float = 0.0
    kind: ClassVar[Literal[CommandKind.N]] = CommandKind.N

    def __str__(self) -> str:
        return f"N({self.node}, r={self.squeezing_ratio}, θ={self.squeezing_angle})"

@dataclasses.dataclass(repr=False)
class M():
    """Measurement command.

    A measurement command is represented by a momentum homodyne measurement
    after applying the unitary transformation:

    .. math::

        U(\\alpha, \\beta, \\gamma) = e^{i \\alpha Q} e^{i \\beta Q^2} e^{i \\gamma Q^3}

    Attributes
    ----------
    node : Node
        The node to which this command is applied.
    alpha : float
        The amplitude of the unitary. Defaults to 0.0.
    beta : float
        The angle in radians for the measurement basis. Defaults to 0.0.
    gamma : float
        The angle in radians for the measurement basis. Defaults to 0.0.
    x_domain : dict[Node, float]
        A dictionary giving how the correction depends on the measurement outcomes.
        The keys are the measured nodes that influence the current correction and the values are the corresponding amplitudes.
        Defaults to an empty dictionary.
    z_domain : dict[Node, float]
        A dictionary giving how the correction depends on the measurement outcomes.
        The keys are the measured nodes that influence the current correction and the values are the corresponding amplitudes.
        Defaults to an empty dictionary.
    """
    node: Node
    alpha: float = 0.0
    beta: float = 0.0
    gamma: float = 0.0
    x_domain: dict[Node, float] = dataclasses.field(default_factory=dict)
    z_domain: dict[Node, float] = dataclasses.field(default_factory=dict)
    kind: ClassVar[Literal[CommandKind.M]] = CommandKind.M

    def __str__(self) -> str:
        parts = [str(self.node)]
        if self.alpha != 0.0:
            parts.append(f"α={self.alpha}")
        if self.beta != 0.0:
            parts.append(f"β={self.beta}")
        if self.gamma != 0.0:
            parts.append(f"γ={self.gamma}")
        if self.x_domain:
            parts.append(f"x_domain={self.x_domain}")
        if self.z_domain:
            parts.append(f"z_domain={self.z_domain}")
        return f"M({', '.join(parts)})"

@dataclasses.dataclass(repr=False)
class E():
    """Entanglement command represented by the following unitary transformation:

    .. math::

        E_{ij}(w) = e^{i w Q_i \\otimes Q_j}

    Attributes
    ----------
    nodes : tuple[Node, Node]
        The pair of nodes to entangle.
    weight : float
        The weight of the edge representing the entanglement. Defaults to 1.0.
    """
    nodes: tuple[Node, Node]
    weight: float = 1.0
    kind: ClassVar[Literal[CommandKind.E]] = CommandKind.E

    def __str__(self) -> str:
        return f"E({self.nodes[0]}, {self.nodes[1]}, w={self.weight})"

@dataclasses.dataclass(repr=False)
class X():
    """X correction command represented by the following unitary transformation:

    .. math::

        X_i(s) = e^{- i s P_i}

    Attributes
    ----------
    node : Node
        The node to which this command is applied.
    amplitude : float
        The amplitude of the X correction. Defaults to 0.0.
    x_domain : dict[Node, float]
        A dictionary giving how the correction depends on the measurement outcomes.
        The keys are the measured nodes that influence the current correction and the values are the corresponding amplitudes.
        Defaults to an empty dictionary.
    """
    node: Node
    amplitude: float = 0.0
    x_domain: dict[Node, float] = dataclasses.field(default_factory=dict)
    kind: ClassVar[Literal[CommandKind.X]] = CommandKind.X

    def __str__(self) -> str:
        parts = [str(self.node)]
        if self.amplitude != 0.0:
            parts.append(f"a={self.amplitude}")
        if self.x_domain:
            parts.append(f"domain={self.x_domain}")
        return f"X({', '.join(parts)})"

@dataclasses.dataclass(repr=False)
class Z():
    """Z correction command represented by the following unitary transformation:

    .. math::

        Z_i(s) = e^{i s Q_i}

    Attributes
    ----------
    node : Node
        The node to which this command is applied.
    amplitude : float
        The amplitude of the Z correction. Defaults to 0.0.
    z_domain : dict[Node, float]
        A dictionary giving how the correction depends on the measurement outcomes.
        The keys are the measured nodes that influence the current correction and the values are the corresponding amplitudes.
        Defaults to an empty dictionary.
    """
    node: Node
    amplitude: float = 0.0
    z_domain: dict[Node, float] = dataclasses.field(default_factory=dict)
    kind: ClassVar[Literal[CommandKind.Z]] = CommandKind.Z

    def __str__(self) -> str:
        parts = [str(self.node)]
        if self.amplitude != 0.0:
            parts.append(f"a={self.amplitude}")
        if self.z_domain:
            parts.append(f"domain={self.z_domain}")
        return f"Z({', '.join(parts)})"

Command = N | M | E | X | Z
Correction = X | Z
