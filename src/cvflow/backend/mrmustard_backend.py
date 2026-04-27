from cvflow.backend.abstract_backend import AbstractBackend

from mrmustard.lab.states import Coherent, SqueezedVacuum, QuadratureEigenstate, Number, Vacuum, Ket
from mrmustard.lab.transformations import CZgate, Dgate, Pgate
from mrmustard.lab import CircuitComponent, HomodyneSampler
import mrmustard.math as math
import numpy as np

INV_SQRT_2 = 1 / 1.414213562

class MRMustardBackend(AbstractBackend):
    """CV-MBQC simulation backend using the MRMustard library (^1.0.0.a1).
    """
    def __init__(self):
        self._qr: CircuitComponent = Vacuum(0)
        self._homodyne_sampler: HomodyneSampler = HomodyneSampler(phi=np.pi/2, num=5000)
        self.is_qr_initialized = False

        super().__init__()

    def prepare_mode(self, node: int, squeezing_ratio: float, squeezing_angle: float) -> None:
        """Prepare *node* as a squeezed vacuum state (N command)."""
        if not self.is_qr_initialized:
            self._qr = SqueezedVacuum(mode=node, r=squeezing_ratio, phi=2*squeezing_angle)
            self.is_qr_initialized = True
            return
        self._qr = self._qr >> SqueezedVacuum(mode=node, r=squeezing_ratio, phi=2*squeezing_angle) # type: ignore

    def entangle_modes(self, nodes: tuple[int, int], weight: float) -> None:
        """Apply a weighted CZ gate between *nodes* (E command)."""
        self._qr = self._qr >> CZgate(modes=nodes, s=weight) # type: ignore

    def measure_mode(self, node: int, alpha: float, beta: float, gamma: float) -> float:
        """Measure *node* in the adaptive basis (M command)."""
        if gamma != 0:
            raise NotImplementedError("MRMustardBackend does not support non-zero gamma for measurement commands.")

        self._qr = self._qr >> Dgate(mode=node, x=alpha) >> Pgate(mode=node, shearing=beta) # type: ignore


        m_outcome_array = self._homodyne_sampler.sample(self._qr, n_samples=1) # type: ignore
        m_outcome = float(m_outcome_array[0, 0])  # Extract scalar from batched result
        self._qr = (self._qr >> QuadratureEigenstate(mode=node, x=m_outcome, phi=np.pi/2).dual).normalize() # type: ignore
        return m_outcome


    def apply_x_correction(self, node: int, amplitude: float) -> None:
        """Apply an X correction to *node* with the given amplitude (X command)."""
        self._qr = self._qr >> Dgate(mode=node, x=amplitude*INV_SQRT_2) # type: ignore

    def apply_z_correction(self, node: int, amplitude: float) -> None:
        """Apply a Z correction to *node* with the given amplitude (Z command)."""
        self._qr = self._qr >> Dgate(mode=node, y=amplitude*INV_SQRT_2) # type: ignore
