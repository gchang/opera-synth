"""Module for simulating stacks of SLCs.

Uses the Cholesky decomposition of a covariance matrix at each pixel
to correlate complex circular Gaussian (CCG) samples over time.
"""

from functools import partial

import jax.numpy as jnp
import numpy as np
from jax import Array, jit, random
from jax.scipy.linalg import cho_factor, cho_solve
from numpy.typing import NDArray


@partial(jit, static_argnums=(1,))
def ccg_noise_jax(key: Array, N: int) -> Array:
    """Create N samples of standard complex circular Gaussian noise using JAX.

    Parameters
    ----------
    key : Array
        JAX random key for generating random numbers.
        Created with jax.random.key()
    N : int
        Number of complex samples to generate.

    Returns
    -------
    jnp.ndarray
        Array of N complex64 samples.

    Notes
    -----
    This function generates complex circular Gaussian noise with zero mean
    and unit variance. The real and imaginary parts are independent and
    each has a standard deviation of 1/sqrt(2).

    """
    # Split the key for generating real and imaginary parts
    key_real, key_imag = random.split(key)

    # Generate real and imaginary parts separately
    real_part = random.normal(key_real, shape=(N,), dtype=jnp.float32) / jnp.sqrt(2.0)
    imag_part = random.normal(key_imag, shape=(N,), dtype=jnp.float32) / jnp.sqrt(2.0)

    # Combine real and imaginary parts into complex numbers
    return real_part + 1j * imag_part


def simulate_coh_stack(
    time: jnp.ndarray,
    gamma0: jnp.ndarray,
    gamma_inf: jnp.ndarray,
    Tau0: jnp.ndarray,
    signal: jnp.ndarray | None = None,
    seasonal_mask: jnp.ndarray | None = None,
    seasonal_A: jnp.ndarray | None = None,
    seasonal_B: jnp.ndarray | None = None,
) -> Array:
    """Create a coherence matrix at each pixel.

    Parameters
    ----------
    time : np.ndarray
        Array of time values, arbitrary units.
        shape (N,) where N is the number of acquisitions.
    gamma0 : np.ndarray
        Initial coherence value for each pixel.
    gamma_inf : np.ndarray
        Asymptotic coherence value for each pixel.
    Tau0 : np.ndarray
        Coherence decay time constant for each pixel.
    signal : np.ndarray
        Simulated signal phase for each pixel.
    seasonal_mask : np.ndarray, optional
        Boolean mask to indicate, for each 2D pixel, whether to
        use a seasonally decorelating model:
        gamma = (A + B cos(2pi * t / 365))
    seasonal_A : np.ndarray, optional
        A coefficient for pixels whose `seasonal_mask=True`
    seasonal_B : np.ndarray, optional
        B coefficient for pixels whose `seasonal_mask=True`

    Returns
    -------
    np.ndarray
        The simulated coherence matrix for each pixel.

    """
    num_time = time.shape[0]
    temp_baselines = jnp.abs(time[None, :] - time[:, None])
    temp_baselines = temp_baselines[None, None, :, :]
    gamma0 = jnp.atleast_2d(gamma0)[:, :, None, None]
    gamma_inf = jnp.atleast_2d(gamma_inf)[:, :, None, None]
    Tau0 = jnp.atleast_2d(Tau0)[:, :, None, None]

    gamma = (gamma0 - gamma_inf) * jnp.exp(-temp_baselines / Tau0) + gamma_inf
    if signal is not None:
        phase_diff = signal[:, None] - signal[None, :]
        phase_term = jnp.exp(1j * phase_diff)
    else:
        phase_term = jnp.exp(1j * 0)

    if seasonal_mask is not None:
        if seasonal_A is None or seasonal_B is None:
            raise ValueError(
                "Must provide `seasonal_A` and `seasonal_B` if passing `seasonal_mask`"
            )
        if seasonal_mask.dtype != bool:
            raise ValueError("`seasonal_mask` must be a boolean array")
        A = jnp.atleast_2d(seasonal_A)[:, :, None, None]
        B = jnp.atleast_2d(seasonal_B)[:, :, None, None]
        mask = jnp.atleast_2d(seasonal_mask)[:, :]
        # A, B = seasonal_coeffs.A, seasonal_coeffs.B
        # assert A.ndim == B.ndim == 4
        seasonal_factor = (A + B * jnp.cos(2 * jnp.pi * temp_baselines / 365.25)) ** 2
        # Ensure it is a valid coherence multiplier
        seasonal_factor = jnp.clip(seasonal_factor, 0, 1)
        # Where
        seasonal_factor = seasonal_factor.at[~mask, :, :].set(1.0)
        gamma = gamma * seasonal_factor

    C = gamma * phase_term

    rl, cl = jnp.tril_indices(num_time, k=-1)

    C = C.at[..., rl, cl].set(
        jnp.conj(jnp.transpose(C, axes=(0, 1, 3, 2))[..., rl, cl])
    )

    # Reset the diagonals of each pixel to 1
    rs, cs = jnp.diag_indices(num_time)
    C = C.at[:, :, rs, cs].set(1)

    return C


@partial(jit, static_argnums=(3, 4))
def _sample(C_tiled, defo_stack, key, num_pixels: int, num_time: int) -> Array:
    stack_i = jnp.exp(1j * defo_stack[:, None, :, :])
    stack_j = jnp.exp(1j * defo_stack[None, :, :, :])
    diff_stack = stack_i * stack_j.conj()
    signal_cov = diff_stack.transpose(2, 3, 0, 1)
    C_tiled_with_signal = C_tiled * signal_cov
    C_unstacked = C_tiled_with_signal.reshape(num_pixels, num_time, num_time)

    # noise = ccg_noise(num_time * num_pixels)
    noise = ccg_noise_jax(key, num_time * num_pixels)
    noise_unstacked = noise.reshape(num_pixels, num_time, 1)

    L_unstacked = jnp.linalg.cholesky(C_unstacked)
    return L_unstacked @ noise_unstacked


def make_noisy_samples_jax(
    key,
    C: Array,
    defo_stack: Array,
    amplitudes: Array | None = None,
) -> np.ndarray:
    """Create noisy deformation samples given a covariance matrix and deformation stack.

    Parameters
    ----------
    key : jax.random.KeyArray
        JAX pseudo random key.
        https://jax.readthedocs.io/en/latest/_autosummary/jax.random.key.htm
    C : Array,
        Covariance matrix of shape (num_time, num_time) , or you can pass
        on matrix per pixel as shape (rows, cols, num_time, num_time).
    defo_stack : Array,
        Deformation stack of shape (num_time, rows, cols).
    amplitudes: 2D Array, optional
        If provided, set the amplitudes of the output pixels.
        Default is to use all ones.

    Returns
    -------
    samps3d: np.ndarray
        Noisy deformation samples of shape (num_time, rows, cols).

    """
    if amplitudes is not None and amplitudes.ndim != 2:
        raise ValueError("`amplitudes` must be 2D, or None")

    num_time, rows, cols = defo_stack.shape
    shape2d = (rows, cols)
    num_pixels = jnp.prod(jnp.array(shape2d))

    C_tiled = jnp.tile(C, (*shape2d, 1, 1)) if C.squeeze().ndim == 2 else jnp.asarray(C)

    if C_tiled.shape != (rows, cols, num_time, num_time):
        raise ValueError(f"{C_tiled.shape=}, but {defo_stack.shape=}")
    # Reset the diagonals of each pixel to 1
    rs, cs = jnp.diag_indices(num_time)
    C_tiled = C_tiled.at[:, :, rs, cs].set(1.0 + 0.0j)

    samps = _sample(C_tiled, defo_stack, key, int(num_pixels), int(num_time))

    samps3d = jnp.moveaxis(samps.reshape(*shape2d, num_time), -1, 0)
    if amplitudes is None:
        return samps3d

    return samps3d * amplitudes[None, :, :]


@partial(jit, static_argnums=(1,))
def compute_crlb_batch(
    C_arrays: NDArray[np.complex64], num_looks: int, reference_idx: int = 0
) -> Array:
    """Compute CRLB for batch of covariance matrices."""
    rows, cols, n, _ = C_arrays.shape
    Gamma = jnp.abs(C_arrays)

    # Identity used for regularization and for solving
    Id = jnp.eye(n, dtype=Gamma.dtype)
    # repeat the identity matrix for each pixel
    Id = jnp.tile(Id, (rows, cols, 1, 1))

    # Attempt to invert Gamma
    cho, is_lower = cho_factor(Gamma)

    # Check: If it fails the cholesky factor, it's close to singular and
    # we should just fall back to EVD
    # Use the already- factored |Gamma|^-1, solving Ax = I gives the inverse
    Gamma_inv = cho_solve((cho, is_lower), Id)
    # Compute Fisher Information Matrix
    X = 2 * num_looks * (Gamma * Gamma_inv - Id.astype("float32"))

    # Compute the CRLB standard deviation
    # # Normally, we construct the Theta partial derivative matrix like this
    # Theta = np.zeros((N, N - 1))
    # First row is 0 (using day 0 as reference)
    # Theta[1:, :] = np.eye(N - 1)  # Last N-1 rows are identity
    # More efficient computation of Theta.T @ X @ Theta
    # Instead of explicit matrix multiplication, directly extract relevant elements
    # We want all elements except the reference row/column
    row_idx = jnp.concatenate(
        [jnp.arange(reference_idx), jnp.arange(reference_idx + 1, n)]
    )
    projected_fim = X[..., row_idx[:, None], row_idx]

    # Invert each (n-1, n-1) matrix in the batch
    # Use cholesky repeat the (n-1, n-1) identity matrix for each pixel
    Id = jnp.tile(jnp.eye(n - 1, dtype=projected_fim.dtype), (rows, cols, 1, 1))
    cho, is_lower = cho_factor(projected_fim)
    crlb = cho_solve((cho, is_lower), Id)  # Shape: (rows, cols, n-1, n-1)

    # Extract standard deviations from the diagonal of each CRLB matrix
    # Shape: (rows, cols, n-1)
    crlb_std_dev = jnp.sqrt(jnp.diagonal(crlb, axis1=-2, axis2=-1))
    # Insert zeros at reference_idx to match evd_estimate shape (rows, cols, n)
    crlb_std_dev = jnp.insert(crlb_std_dev, reference_idx, 0, axis=-1)
    # Now move the n (time) dimension to be first
    return jnp.moveaxis(crlb_std_dev, -1, 0)
