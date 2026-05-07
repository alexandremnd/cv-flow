from abc import ABC, abstractmethod
from collections.abc import Mapping

import numpy as np

from cvflow.command import CommandKind, Node
from cvflow.pattern import Pattern


class AbstractBackend(ABC):
    """Abstract base class for CV-MBQC simulation backends.

    Subclass this and implement the abstract methods to plug in any
    physical or numerical backend.  Call :meth:`run` to execute a full
    :class:`~cvflow.pattern.Pattern`.
    """
    def __init__(self):
        self._measurement_results: dict[int, float] = {}

    def run(self, pattern: Pattern, input_state) -> None:
        """Execute every command in *pattern* sequentially on the backend.

        Parameters
        ----------
        pattern : Pattern
            The pattern to run.
        input_state : any
            The input state to the pattern. The type and interpretation of this
            object is backend-dependent.
        """
        self._compute_expected_state(pattern, input_state)
        self._insert_input_state(input_state)

        for cmd in pattern:
            match cmd.kind:
                case CommandKind.N:
                    self._prepare_mode(cmd.node, cmd.squeezing_ratio, cmd.squeezing_angle)
                case CommandKind.E:
                    self._entangle_modes(cmd.nodes, cmd.weight)
                case CommandKind.M:
                    outcome = self._measure_mode(cmd.node, cmd.alpha, cmd.beta, cmd.gamma)
                    self._measurement_results[cmd.node] = outcome
                case CommandKind.X:
                    correction_amplitude = self._correction_amplitude(cmd.amplitude, cmd.x_domain)
                    self._apply_x_correction(cmd.node, correction_amplitude)
                case CommandKind.Z:
                    correction_amplitude = self._correction_amplitude(cmd.amplitude, cmd.z_domain)
                    self._apply_z_correction(cmd.node, correction_amplitude)

    def _correction_amplitude(self, amplitude: float, domain: Mapping[Node, float]) -> float:
        for node, weight in domain.items():
            amplitude += self._measurement_results[node] * weight
        return amplitude

    def _compute_expected_state(self, pattern: Pattern, input_state):
        """Compute the expected output state of *pattern* given *input_state*
        in the infinite squeezing limit. This function guarantees that the backend
        is reset after it is called.

        Parameters
        ----------
        pattern : Pattern
            The pattern to simulate.
        input_state : any
            The input state to use for the simulation. The type and interpretation
            of this object is backend-dependent.
        """
        self._reset()
        self._insert_input_state(input_state)

        for cmd in pattern:
            match cmd.kind:
                case CommandKind.N:
                    self._prepare_mode(cmd.node, 10, np.pi)
                case CommandKind.E:
                    self._entangle_modes(cmd.nodes, cmd.weight)
                case CommandKind.M:
                    self._measure_mode_with_outcome(cmd.node, cmd.alpha, cmd.beta, cmd.gamma, outcome=0)
                case CommandKind.X:
                    pass
                case CommandKind.Z:
                    pass

        self._store_expected_state()
        self._reset()

    # ------------------------------------------------------------------
    # Abstract interface — one method per command kind
    # ------------------------------------------------------------------
    @abstractmethod
    def _store_expected_state(self) -> None:
        """Save the expected output state after running the pattern with ideal
        infinite squeezing. This is used for benchmarking purposes, to compare
        with the actual output state of the backend after running the pattern.
        The type and interpretation of the expected state is backend-dependent.
        """

    @abstractmethod
    def compute_fidelity(self) -> float:
        """Compute the fidelity between the actual output state of the backend after running a
        pattern and the expected output state computed in the infinite squeezing case.

        This method is used for benchmarking purposes, to compare the actual
        output state of the backend after running a pattern with the expected
        output state in the infinite squeezing case.
        The type and interpretation of the input states is backend-dependent.
        """

    @abstractmethod
    def _reset(self) -> None:
        """Reset the backend to its initial state, clearing any stored quantum
        state or measurement results.
        """

    @abstractmethod
    def _insert_input_state(self, state) -> None:
        """Set the input state.

        Parameters
        ----------
        node : int
            Mode index.
        state : any
            State to insert.  The type and interpretation of this object is
            backend-dependent.
         """

    @abstractmethod
    def _prepare_mode(self, node: int, squeezing_ratio: float, squeezing_angle: float) -> None:
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
    def _entangle_modes(self, nodes: tuple[int, int], weight: float) -> None:
        """Apply a weighted CZ gate between *nodes* (E command).

        Parameters
        ----------
        nodes : tuple[int, int]
            The two mode indices to entangle.
        weight : float
            Edge weight (coupling strength).
        """

    @abstractmethod
    def _measure_mode(self, node: int, alpha: float, beta: float, gamma: float) -> float:
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
    def _measure_mode_with_outcome(self, node: int, alpha: float, beta: float, gamma: float, outcome: float) -> None:
        """Measure *node* in the adaptive basis (M command) with a predefined outcome.

        This method is used for benchmarking purposes, to compute the expected output state of a pattern
        in the ideal case of infinite squeezing, where we can assume that all measurement outcomes are 0.

        The measurement corresponds to a momentum homodyne detection after
        applying the unitary $U(\\alpha, \\beta, \\gamma) = \\exp(i\\alpha Q) \\exp(i\\beta Q^2) \\exp(i\\gamma Q^3)$.

        Parameters
        ----------
        node : int
            Mode index to measure.
        alpha, beta, gamma : float
            Unitary parameters.
        outcome : float
            The predefined measurement outcome to use instead of sampling.
        """

    @abstractmethod
    def _apply_x_correction(self, node: int, amplitude: float) -> None:
        """Apply an X (position displacement) correction to *node* (X command).

        Parameters
        ----------
        node : int
            Mode index.
        amplitude : float
            Displacement amplitude.
        """

    @abstractmethod
    def _apply_z_correction(self, node: int, amplitude: float) -> None:
        """Apply a Z (momentum displacement) correction to *node* (Z command).

        Parameters
        ----------
        node : int
            Mode index.
        amplitude : float
            Displacement amplitude.
        """
