import datetime
import logging
import re
from collections.abc import Callable, Mapping
from concurrent.futures import Executor, Future
from functools import partial
from pathlib import Path

import h5py
import jax.numpy as jnp
import numpy as np
from jax import Array, jit, lax
from jax.typing import ArrayLike

from ._types import P, PathOrStr, T

logger = logging.getLogger("synth")

DATE_FORMAT = "%Y%m%d"
DATETIME_FORMAT = "%Y%m%dT%H%M%S"


ALL_LAYERS = slice(None)


def load_current_phase(
    files: Mapping[str, PathOrStr],
    rows: slice,
    cols: slice,
    idx: slice | int = ALL_LAYERS,
) -> np.ndarray:
    """Load and sum the phase data from multiple HDF5 files for a row/column block.

    Parameters
    ----------
    files : dict[str, Path]
        Dictionary of file paths for different phase components.
    rows : slice
        Row slice to extract.
    cols : slice
        Column slice to extract.
    idx : slice | int, optional
        Single index or slice of the 3D cube to load.
        Default is to load all depth layers.

    Returns
    -------
    np.ndarray: 3D Float32 array representing the summed true input phase data
    for the specified block.

    """
    summed_phase = None

    # logger.debug(f"Loading {files} at {idx=}, {rows=}, {cols=}")
    for _, file_path in files.items():
        with h5py.File(file_path, "r") as f:
            # Assume the main dataset is named 'data'. Adjust if necessary.
            dset: h5py.Dataset = f["data"]

            # Check if the dset is 3D
            if dset.ndim == 3:
                # For 3D datasets, load the full depth
                data = dset[idx, rows, cols]
            elif dset.ndim == 2:
                # For 2D datasets, add a depth dimension of 1
                data = dset[rows, cols][np.newaxis, :, :]
            else:
                raise ValueError(f"Unexpected dset shape in {file_path}: {dset.shape}")

            if summed_phase is None:
                summed_phase = data
            else:
                summed_phase += data

    if summed_phase is None:
        raise ValueError("No valid data found in the provided files.")

    return summed_phase


def _setup_logging(level=logging.INFO):
    if not logger.handlers:
        logger.setLevel(level)
        h = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        h.setFormatter(formatter)
        logger.addHandler(h)


def round_mantissa(z: np.ndarray, significant_bits=10, truncate: bool = False):
    """Zero out bits in mantissa of elements of array in place.

    Attempts to round the floating point numbers zeroing.

    Parameters
    ----------
    z : numpy.array
        Real or complex array whose mantissas are to be zeroed out
    significant_bits : int, optional
        Number of bits to preserve in mantissa. Defaults to 10.
        Lower numbers will truncate the mantissa more and enable
        more compression.
    truncate : bool, optional
        Instead of attempting to round, simply truncate the mantissa.
        Default = False

    """
    # recurse for complex data
    if np.iscomplexobj(z):
        round_mantissa(z.real, significant_bits)
        round_mantissa(z.imag, significant_bits)
        return

    if not issubclass(z.dtype.type, np.floating):
        err_str = "argument z is not complex float or float type"
        raise TypeError(err_str)

    mant_bits = np.finfo(z.dtype).nmant
    float_bytes = z.dtype.itemsize

    if significant_bits == mant_bits:
        return

    if not 0 < significant_bits <= mant_bits:
        err_str = f"Require 0 < {significant_bits=} <= {mant_bits}"
        raise ValueError(err_str)

    # create integer value whose binary representation is one for all bits in
    # the floating point type.
    allbits = (1 << (float_bytes * 8)) - 1

    # Construct bit mask by left shifting by nzero_bits and then truncate.
    # This works because IEEE 754 specifies that bit order is sign, then
    # exponent, then mantissa.  So we zero out the least significant mantissa
    # bits when we AND with this mask.
    nzero_bits = mant_bits - significant_bits
    bitmask = (allbits << nzero_bits) & allbits

    utype = np.dtype(f"u{float_bytes}")
    # view as uint type (can not mask against float)
    u = z.view(utype)

    if truncate is False:
        round_mask = 1 << (nzero_bits - 1)
        u += round_mask  # Add the rounding mask before applying the bitmask
    # bitwise-and in-place to mask
    u &= bitmask


class DummyProcessPoolExecutor(Executor):
    """Dummy ProcessPoolExecutor for to avoid forking for single_job purposes."""

    def __init__(self, max_workers: int | None = None, **kwargs):  # noqa: D107
        self._max_workers = max_workers

    def submit(  # noqa: D102
        self, fn: Callable[P, T], /, *args: P.args, **kwargs: P.kwargs
    ) -> Future[T]:
        future: Future = Future()
        result = fn(*args, **kwargs)
        future.set_result(result)
        return future

    def shutdown(self, wait: bool = True, cancel_futures: bool = True):  # noqa: D102
        pass


@partial(jit, static_argnums=(1, 2, 3))
def take_looks(
    image: ArrayLike, row_looks: int, col_looks: int, average: bool = True
) -> Array:
    """Downsample a numpy matrix by summing blocks of (row_looks, col_looks).

    Parameters
    ----------
    image : ArrayLike
        2D array of an image
    row_looks : int
        the reduction rate in row direction
    col_looks : int
        the reduction rate in col direction
    average : bool, optional
        whether to average or sum, by default True

    Returns
    -------
    Array
        The downsampled array, shape = ceil(rows / row_looks, cols / col_looks)

    Notes
    -----
    Cuts off values if the size isn't divisible by num looks.

    """
    # Ensure the image has a channel/batch dimension (assuming grayscale image)
    # Add a (batch, ..., channel) dimensions to make NHWC
    image = jnp.array(image)[jnp.newaxis, ..., jnp.newaxis]

    # Create a kernel filled with ones
    # Kernel shape: HWIO (height, width, input_channels, output_channels)
    kernel = jnp.ones((row_looks, col_looks, 1, 1), dtype=image.dtype)

    # With each window, we're jumping over by the same number of pixels
    strides = (row_looks, col_looks)
    result = lax.conv_general_dilated(
        image,
        kernel,
        window_strides=strides,
        padding="SAME",
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )

    # Average if required
    if average:
        result /= row_looks * col_looks

    return result.squeeze()


def _get_path_from_gdal_str(name: Path | str) -> Path:
    s = str(name)
    if s.upper().startswith("DERIVED_SUBDATASET"):
        # like DERIVED_SUBDATASET:AMPLITUDE:slc_filepath.tif
        p = s.split(":")[-1].strip('"').strip("'")
    elif ":" in s and (s.upper().startswith("NETCDF") or s.upper().startswith("HDF")):
        # like NETCDF:"slc_filepath.nc":subdataset
        p = s.split(":")[1].strip('"').strip("'")
    else:
        # Whole thing is the path
        p = str(name)
    return Path(p)


def get_dates(filename: Path | str, fmt: str = DATE_FORMAT) -> list[datetime.datetime]:
    """Search for dates in the stem of `filename` matching `fmt`.

    Excludes dates that are not in the stem of `filename` (in the directories).

    Parameters
    ----------
    filename : Path or str
        Path or string to search for dates.
    fmt : str, optional
        Format of date to search for. Default is %Y%m%d

    Returns
    -------
    list[datetime.datetime]
        list of dates found in the stem of `filename` matching `fmt`.

    Examples
    --------
    >>> get_dates("/path/to/20191231.slc.tif")
    [datetime.datetime(2019, 12, 31, 0, 0)]
    >>> get_dates("S1A_IW_SLC__1SDV_20191231T000000_20191231T000000_032123_03B8F1_1C1D.nc")
    [datetime.datetime(2019, 12, 31, 0, 0), datetime.datetime(2019, 12, 31, 0, 0)]
    >>> get_dates("/not/a/date_named_file.tif")
    []

    """  # noqa: E501
    path = _get_path_from_gdal_str(filename)
    pattern = _date_format_to_regex(fmt)
    date_list = re.findall(pattern, path.name)
    if not date_list:
        return []
    return [datetime.datetime.strptime(d, fmt) for d in date_list]


def _date_format_to_regex(date_format: str) -> re.Pattern:
    r"""Convert a python date format string to a regular expression.

    Parameters
    ----------
    date_format : str
        Date format string, e.g. DATE_FORMAT

    Returns
    -------
    re.Pattern
        Regular expression that matches the date format string.

    Examples
    --------
    >>> pat2 = _date_format_to_regex("%Y%m%d").pattern
    >>> pat2 == re.compile(r'\d{4}\d{2}\d{2}').pattern
    True
    >>> pat = _date_format_to_regex("%Y-%m-%d").pattern
    >>> pat == re.compile(r'\d{4}\-\d{2}\-\d{2}').pattern
    True

    """
    # Escape any special characters in the date format string
    date_format = re.escape(date_format)

    # Replace each format specifier with a regular expression that matches it
    date_format = date_format.replace("%Y", r"\d{4}")
    date_format = date_format.replace("%y", r"\d{2}")
    date_format = date_format.replace("%m", r"\d{2}")
    date_format = date_format.replace("%d", r"\d{2}")
    date_format = date_format.replace("%H", r"\d{2}")
    date_format = date_format.replace("%M", r"\d{2}")
    date_format = date_format.replace("%S", r"\d{2}")
    date_format = date_format.replace("%j", r"\d{3}")

    # Return the resulting regular expression
    return re.compile(date_format)
