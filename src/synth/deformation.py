"""Module for simple synthetic deformations, not following any real physical model."""

import numpy as np


def gaussian(
    shape: tuple[int, int],
    sigma: float | tuple[float, float],
    row: int | None = None,
    col: int | None = None,
    normalize: bool = False,
    amp: float | None = None,
    noise_sigma: float = 0.0,
) -> np.ndarray:
    """Create a 2D Gaussian of given shape and width.

    Parameters
    ----------
    shape : tuple[int, int]
        (rows, cols) of the output array
    sigma : float or tuple[float, float]
        Standard deviation of the Gaussian.
        If one float provided, makes an isotropic Gaussian.
        Otherwise, uses [sigma_row, sigma_col] to make elongated Gaussian.
    row : int, optional
        Center row of the Gaussian. Defaults to the middle of the array.
    col : int, optional
        Center column of the Gaussian. Defaults to the middle of the array.
    normalize : bool, optional
        Normalize the amplitude peak to 1. Defaults to False.
    amp : float, optional
        Peak height of Gaussian. If None, peak will be 1/(2*pi*sigma^2).
    noise_sigma : float, optional
        Std. dev of random Gaussian noise added to image. Defaults to 0.0.

    Returns
    -------
    np.ndarray
        2D array containing the Gaussian

    """
    rows, cols = shape
    if not isinstance(sigma, tuple):
        sigma = (sigma, sigma)
    sy, sx = sigma

    # Set default center if not provided
    if row is None:
        row = rows // 2
    if col is None:
        col = cols // 2

    # Create coordinate grids
    y, x = np.ogrid[:rows, :cols]

    # Calculate the 2D Gaussian
    g = np.exp(
        -((x - col) ** 2.0 / (2.0 * sx**2.0) + (y - row) ** 2.0 / (2.0 * sy**2.0))
    )
    normed = _normalize_gaussian(g, normalize=normalize, amp=amp)
    if noise_sigma > 0:
        normed += noise_sigma * np.random.standard_normal(shape)
    return normed


def ramp(
    shape: tuple[int, int], amplitude: float = 1, rotate_degrees: float = 0
) -> np.ndarray:
    """Create a synthetic ramp with optional rotation.

    Parameters
    ----------
    shape : tuple[int, int]
        (rows, cols) of the output array.
    amplitude : float
        The maximum amplitude the ramp reaches.
        Default = 1.
    rotate_degrees : float, optional
        Rotation of the ramp in degrees. 0 degrees is a left-to-right ramp.
        Positive values rotate counterclockwise. Defaults to 0.

    Returns
    -------
    np.ndarray
        2D array containing the synthetic ramp

    """
    rows, cols = shape

    # Create coordinate grids
    y, x = np.ogrid[:rows, :cols]

    # Normalize coordinates to [-1, 1] range
    x = (x - cols / 2) / (cols / 2)
    y = (y - rows / 2) / (rows / 2)

    # Convert rotation to radians
    theta = np.radians(rotate_degrees)

    # Apply rotation
    x_rot = x * np.cos(theta) + y * np.sin(theta)

    # Create ramp
    ramp = (x_rot + 1) / 2  # Normalize to [0, 1] range

    # Scale by amplitude
    ramp *= amplitude

    return ramp


def _normalize_gaussian(out, normalize=False, amp=None):
    """Normalize either to 1 max, or to `amp` max."""
    if normalize or amp is not None:
        out /= out.max()
    if amp is not None:
        out *= amp
    return out


def valley(shape, rotate_degrees=0):
    """Make a valley in image center (curvature only in 1 direction)."""
    from scipy.ndimage import rotate

    rows, cols = shape
    out = np.dot(np.ones((rows, 1)), (np.linspace(-1, 1, cols) ** 2).reshape((1, cols)))
    if rotate_degrees > 0:
        rotate(out, rotate_degrees, mode="edge", output=out)
    return out
