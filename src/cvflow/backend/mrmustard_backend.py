from cvflow.backend.abstract_backend import AbstractBackend
from cvflow.command import Node

from mrmustard.lab.states import SqueezedVacuum, QuadratureEigenstate, Vacuum
from mrmustard.lab.transformations import CZgate, Dgate, Pgate
from mrmustard.lab import CircuitComponent, HomodyneSampler
import numpy as np

INV_SQRT_2 = 1 / 1.414213562

class MRMustardBackend(AbstractBackend):
    """CV-MBQC simulation backend using the MRMustard library (^1.0.0.a1).
    """
    def __init__(self, homodyne_bounds: tuple[float, float] = (-70, 70), homodyne_num: int =1000):
        """Initialize the MRMustardBackend.

        Parameters
        ----------
        homodyne_bounds : tuple[float, float], optional
            Lower and upper bounds used for homodyne sampling, by default (-70, 70)
        homodyne_num : int, optional
            Number of sample points used by the homodyne sampler, by default 1000
        """
        self._qr: CircuitComponent = Vacuum(0)
        self._expected_state = None
        self._homodyne_bounds: tuple[float, float] = homodyne_bounds
        self._homodyne_num: int = int(homodyne_num)
        self._homodyne_sampler: HomodyneSampler = HomodyneSampler(phi=np.pi/2, bounds=homodyne_bounds, num=homodyne_num)
        self._is_qr_initialized = False

        super().__init__()

    def _reset(self) -> None:
        self._qr = Vacuum(0)
        self._is_qr_initialized = False
        self._measurement_results.clear()

    def _insert_input_state(self, state) -> None:
        if not self._is_qr_initialized:
            self._qr = state
            self._is_qr_initialized = True
            return
        self._qr = self._qr >> state # type: ignore

    def _prepare_mode(self, node: int, squeezing_ratio: float, squeezing_angle: float) -> None:
        if not self._is_qr_initialized:
            self._qr = SqueezedVacuum(mode=node, r=squeezing_ratio, phi=2*squeezing_angle)
            self._is_qr_initialized = True
            return
        self._qr = self._qr >> SqueezedVacuum(mode=node, r=squeezing_ratio, phi=2*squeezing_angle) # type: ignore

    def _entangle_modes(self, nodes: tuple[int, int], weight: float) -> None:
        self._qr = self._qr >> CZgate(modes=nodes, s=weight) # type: ignore

    def _measure_mode(self, node: int, alpha: float, beta: float, gamma: float) -> float:
        if gamma != 0:
            raise NotImplementedError("MRMustardBackend does not support non-zero gamma for measurement commands.")

        self._qr = self._qr >> Dgate(mode=node, x=-alpha * INV_SQRT_2) >> Pgate(mode=node, shearing=4*beta) # type: ignore

        m_outcome_array = self._homodyne_sampler.sample(self._qr[node], n_samples=1) # type: ignore
        m_outcome = float(m_outcome_array[0, 0])  # Extract scalar from batched result
        self._qr = (self._qr >> QuadratureEigenstate(mode=node, x=m_outcome, phi=np.pi/2).dual).normalize() # type: ignore
        return m_outcome

    def _measure_mode_with_outcome(self, node: Node, alpha: float, beta: float, gamma: float, outcome: float) -> None:
        if gamma != 0:
            raise NotImplementedError("MRMustardBackend does not support non-zero gamma for measurement commands.")

        self._qr = self._qr >> Dgate(mode=node, x=-alpha * INV_SQRT_2) >> Pgate(mode=node, shearing=4*beta) # type: ignore
        self._qr = (self._qr >> QuadratureEigenstate(mode=node, x=outcome, phi=np.pi/2).dual).normalize() # type: ignore

    def _apply_x_correction(self, node: int, amplitude: float) -> None:
        self._qr = self._qr >> Dgate(mode=node, x=amplitude*INV_SQRT_2) # type: ignore

    def _apply_z_correction(self, node: int, amplitude: float) -> None:
        self._qr = self._qr >> Dgate(mode=node, y=amplitude*INV_SQRT_2) # type: ignore

    def compute_fidelity(self) -> float:
        return np.real(self._qr.fidelity(self._expected_state)) # type: ignore

    def compute_fidelity_with_state(self, target_state) -> float:
        return np.real(self._qr.fidelity(target_state)) # type: ignore

    def _store_expected_state(self) -> None:
        self._expected_state = self._qr

    def get_expected_state(self):
        return self._expected_state