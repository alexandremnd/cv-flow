# import numpy as np

# from mrmustard.lab import Dgate, SqueezedVacuum, Rgate, State
# from mrmustard.physics import fock

# def gaussian_marginal(state: State, x: np.ndarray, theta: float) -> np.ndarray:
#     """
#     Return the marginal distribution of a Gaussian state along a quadrature defined by angle theta.

#     The quadrature is defined as:

#     .. math::
#         x_\\phi = x \\cos(\\theta) + p \\sin(\\theta)

#     For :math:`\\theta = 0`, this corresponds to the marginal distribution along the :math:`x` quadrature,
#     and for :math:`\\theta = \\pi/2`, this corresponds to the marginal distribution along the :math:`p` quadrature.

#     Parameters
#     ----------
#     state : State
#         State for which to compute the marginal distribution. Must be a single-mode Gaussian state.
#     x : np.ndarray
#         Points at which to evaluate the marginal distribution.
#     theta : float
#         Angle defining the quadrature along which to compute the marginal distribution.

#     Returns
#     -------
#     np.ndarray
#         Marginal distribution evaluated at the points in x, normalized to sum to 1.

#     Notes
#     -----
#     The computation is done by rotating the covariance matrix and means of the state,
#     and then evaluating the Gaussian distribution at the points in x. No numerical integration is performed.
#     """
#     if state.num_modes != 1:
#         raise ValueError("State must be single-mode")
#     if not state.is_gaussian:
#         raise ValueError("State must be Gaussian")

#     sxx = state._cov[0, 0]
#     spp = state._cov[1, 1]
#     sxp = state._cov[0, 1]

#     # Just because I derived the formula as a positive rotation instead of a negative one
#     cos_theta = np.cos(-theta)
#     sin_theta = np.sin(-theta)

#     rotated_sqq = cos_theta**2 * sxx + sin_theta**2 * spp - 2 * cos_theta * sin_theta * sxp
#     # rotated_spp = cos_theta**2 * spp + sin_theta**2 * sxx + 2 * cos_theta * sin_theta * sxp
#     # rotated_sqp = (cos_theta**2 - sin_theta**2) * sxp - sin_theta * cos_theta * (spp - sxx)

#     rotated_q = cos_theta * state._means[0] - sin_theta * state._means[1]
#     # rotated_p = sin_theta * state._means[0] + cos_theta * state._means[1]

#     marginal = np.exp(-0.5/rotated_sqq * (x - rotated_q)**2)
#     marginal /= np.sum(marginal)

#     return marginal

# def fock_marginal(state: State, x: np.ndarray, theta: float) -> np.ndarray:
#     """
#     Return the marginal distribution of a state along a quadrature defined by angle theta.

#     The quadrature is defined as:

#     .. math::
#         x_\\phi = x \\cos(\\theta) + p \\sin(\\theta)

#     For :math:`\\theta = 0`, this corresponds to the marginal distribution along the :math:`x` quadrature,
#     and for :math:`\\theta = \\pi/2`, this corresponds to the marginal distribution along the :math:`p` quadrature.

#     Parameters
#     ----------
#     state : State
#         State for which to compute the marginal distribution. Must be a single-mode state.
#     x : np.ndarray
#         Points at which to evaluate the marginal distribution.
#     theta : float
#         Angle defining the quadrature along which to compute the marginal distribution.

#     Returns
#     -------
#     np.ndarray
#         Marginal distribution evaluated at the points in x.

#     Notes
#     -----
#     The computation is done by converting the state to a density matrix in the Fock basis
#     and using the `fock.quadrature_distribution` function from MrMustard.
#     """
#     if state.num_modes != 1:
#         raise ValueError("State must be single-mode")

#     dm = state.dm(cutoffs=state.cutoffs)
#     _, x_theta_probability = fock.quadrature_distribution(dm, theta, x)

#     return x_theta_probability

# def random_projection(total_state: State, mode: int, theta: float = 0.0, r: float = 10.0) -> tuple[State, float]:
#     """
#     Generate a random projection measurement outcome along a quadrature.

#     This function simulates a homodyne measurement on the given state along the quadrature
#     defined by angle theta. It samples a measurement outcome from the marginal distribution
#     and returns the corresponding post-measurement state as a squeezed vacuum displaced to
#     the measurement outcome and rotated to the measurement quadrature.

#     Parameters
#     ----------
#     total_state : State
#         State to be measured. Can be multi-mode, but the measurement will be performed only on the specified mode.
#     mode : int
#         Mode along which to perform the measurement.
#     theta : float, optional
#         Angle defining the quadrature along which the homodyne measurement is performed, by default 0.0.
#     r : float, optional
#         Squeezing parameter for the measurement state onto which we project, by default 10.0.

#     Returns
#     -------
#     State
#         Post-measurement state represented as a squeezed vacuum displaced to the
#         measurement outcome and rotated to the measurement quadrature.

#     Notes
#     -----
#     The function uses the marginal distribution along the specified quadrature to sample
#     a measurement outcome. A small amount of noise is added to account for the finite
#     grid resolution used in the computation.
#     """
#     x = np.linspace(-10, 10, 200)
#     dx = x[1] - x[0]
#     state = total_state.get_modes(mode)

#     marginal = None
#     if state.is_gaussian:
#         marginal = gaussian_marginal(state, x=x, theta=theta)
#     else:
#         marginal = fock_marginal(state, x=x, theta=theta)

#     x_sample = np.random.choice(x, p=marginal)
#     x_sample += 0.5 * np.random.normal(scale=dx) # we have low grid resolution, so we add some noise

#     # I thought it would be faster to give directly the mean and covariance but in fact it's not.
#     # Thus, better to have a clear code than a slightly "faster" code.
#     # mu, variance = get_squeezed_matrix(r=r, theta=theta, x_theta=x_sample)

#     return state << (SqueezedVacuum(r=r, phi=0, modes=[mode]) >> Dgate(x=x_sample/2, modes=[mode]) >> Rgate(angle=theta, modes=[mode])), x_sample