from abc import ABC, abstractmethod

from cvflow.command import CommandKind
from cvflow.pattern import Pattern


class AbstractBackend(ABC):
    """Abstract base class for CV-MBQC simulation backends.

    Subclass this and implement the five abstract methods to plug in any
    physical or numerical backend.  Call :meth:`run` to execute a full
    :class:`~cvflow.pattern.Pattern`.
    """
    def __init__(self):
        self.measurement_results : dict[int, float] = {}

    def run(self, pattern: Pattern) -> None:
        """Execute every command in *pattern* sequentially.

        Parameters
        ----------
        pattern : Pattern
            The pattern to run.
        """
        for cmd in pattern._commands:
            match cmd.kind:
                case CommandKind.N:
                    self.prepare_mode(cmd.node, cmd.squeezing_ratio, cmd.squeezing_angle)
                case CommandKind.E:
                    self.entangle_modes(cmd.nodes, cmd.weight)
                case CommandKind.M:
                    outcome = self.measure_mode(cmd.node, cmd.alpha, cmd.beta, cmd.gamma)
                    self.measurement_results[cmd.node] = outcome
                case CommandKind.X:
                    for node, amplitude in cmd.x_domain.items():
                        cmd.amplitude -= self.measurement_results[node] * amplitude
                    self.apply_x_correction(cmd.node, cmd.amplitude)
                case CommandKind.Z:
                    for node, amplitude in cmd.z_domain.items():
                        cmd.amplitude -= self.measurement_results[node] * amplitude
                    self.apply_z_correction(cmd.node, cmd.amplitude)

    # ------------------------------------------------------------------
    # Abstract interface — one method per command kind
    # ------------------------------------------------------------------

    @abstractmethod
    def prepare_mode(self, node: int, squeezing_ratio: float, squeezing_angle: float) -> None:
        """Prepare *node* as a squeezed vacuum state (N command).

        Parameters
        ----------
        node : int
            Mode index.
        squeezing_ratio : float
            Squeezing magnitude.
        squeezing_angle : float
            Squeezing angle in degrees.
        """

    @abstractmethod
    def entangle_modes(self, nodes: tuple[int, int], weight: float) -> None:
        """Apply a weighted CZ gate between *nodes* (E command).

        Parameters
        ----------
        nodes : tuple[int, int]
            The two mode indices to entangle.
        weight : float
            Edge weight (coupling strength).
        """

    @abstractmethod
    def measure_mode(self, node: int, alpha: float, beta: float, gamma: float) -> float:
        """Measure *node* in the adaptive basis (M command).

        The measurement corresponds to a momentum homodyne detection after
        applying the unitary $U(\\alpha, \\beta, \\gamma) = \\exp(i\\alpha Q) \\exp(i\\beta Q^2) \\exp(i\\gamma Q^3)$.

        Parameters
        ----------
        node : int
            Mode index to measure.
        alpha, beta, gamma : float
            Unitary parameters.

        Returns
        -------
        float
            The measurement outcome.
        """

    @abstractmethod
    def apply_x_correction(self, node: int, amplitude: float) -> None:
        """Apply an X (position displacement) correction to *node* (X command).

        Parameters
        ----------
        node : int
            Mode index.
        amplitude : float
            Displacement amplitude.
        """

    @abstractmethod
    def apply_z_correction(self, node: int, amplitude: float) -> None:
        """Apply a Z (momentum displacement) correction to *node* (Z command).

        Parameters
        ----------
        node : int
            Mode index.
        amplitude : float
            Displacement amplitude.
        """
