import logging
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import rasterio
import rasterio as rio
import rasterio.windows
from jax import Array, random
from numpy.typing import NDArray
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from troposim import turbulence

from . import covariance, crlb, deformation, global_coherence
from ._blocks import iter_blocks
from ._types import Bbox, PathOrStr
from .config import SimulationInputs
from .utils import _setup_logging, load_current_phase, round_mantissa

SENTINEL_WAVELENGTH = 0.055465763  # meters
METERS_TO_PHASE = 4 * 3.14159 / SENTINEL_WAVELENGTH
HDF5_KWARGS: dict[str, tuple | str] = {"chunks": (5, 256, 256), "compression": "lzf"}

logger = logging.getLogger("synth")


def create_simulation_data(
    inps: SimulationInputs,
    seed: int = 0,
    verbose: bool = False,
):
    """Create realistic SLC simulated data.

    This function generates the necessary data layers for a simulation, including
    turbulence, deformation, and phase ramps. It loads the global coherence model
    coefficients and uses them to simulate correlated noise for each pixel.

    Parameters
    ----------
    inps : SimulationInputs
        Input parameters for the simulation, including the bounding box, resolution,
        number of dates, and flags for including various components.
    seed : int, optional
        Seed for the random number generator, by default 0.
    verbose : bool, optional
        Whether to print additional progress information, by default False.

    Returns
    -------
    np.ndarray
        The simulated noisy stack of SLC data.

    """
    _setup_logging(level=logging.DEBUG if verbose else logging.INFO)

    # Create list of SLC dates
    time = inps.datetimes
    x_arr = np.array(inps.days_since_start)

    outdir = inps.output_dir
    layers_dir = outdir / "input_layers"
    output_slc_dir = outdir / "slcs"
    layers_dir.mkdir(exist_ok=True, parents=True)
    output_slc_dir.mkdir(exist_ok=True, parents=True)

    using_global_coh = inps.custom_covariance is None

    if inps.include_decorrelation and using_global_coh:
        logger.info("Getting Rhos, Tau rasters")
        # The global coherence is at 90 meters:
        upsample_y = round(90 / inps.res_y)
        upsample_x = round(90 / inps.res_x)
        upsample = (upsample_y, upsample_x)
        logger.info(f"Upsampling by {upsample}")
        coherence_files = global_coherence.get_coherence_model_coeffs(
            bounds=inps.bounding_box,
            upsample=upsample,
            output_dir=layers_dir,
            rho_transform=inps.rho_transform,
        )
        logger.info(f"{coherence_files = }")
        with rio.open(coherence_files[0]) as src:
            shape2d = src.shape
            profile = src.profile
    else:
        profile = inps.create_profile()
        shape2d = profile["height"], profile["width"]

    shape3d = (inps.num_dates, shape2d[0], shape2d[1])
    logger.info(f"{profile=}")
    logger.info(f"{shape3d = }")

    files = {}
    if inps.include_turbulence:
        logger.info("Generating turbulence")
        files["turbulence"] = layers_dir / "turbulence.h5"
        create_turbulence(
            shape2d=shape2d,
            num_days=inps.num_dates,
            out_hdf5=files["turbulence"],
            max_amplitude=inps.max_turbulence_amplitude,
            resolution=(inps.res_y, inps.res_x),
        )

    if inps.include_deformation:
        logger.info("Generating deformation")
        files["deformation"] = layers_dir / "deformation.h5"
        create_defo_stack(
            shape=shape3d,
            sigma=shape2d[0] * inps.defo_width_fraction,
            max_amplitude=inps.max_defo_amplitude,
            out_hdf5=files["deformation"],
        )

    if inps.include_ramps:
        logger.info("Generating ramps")
        files["phase_ramps"] = layers_dir / "phase_ramps.h5"
        create_ramps(
            shape2d=shape2d,
            num_days=inps.num_dates,
            out_hdf5=files["phase_ramps"],
        )

    # Setup output tif files
    output_slc_filenames = [
        output_slc_dir / f"{date.strftime('%Y%m%d')}.slc.tif" for date in time
    ]
    slc_profile = profile.copy() | {"dtype": "complex64"}
    for filename in output_slc_filenames:
        with rio.open(filename, "w", **slc_profile) as dst:
            pass

    if inps.include_summed_truth:
        # Setup summed truth phase files
        truth_dir = layers_dir / "truth_unwrapped_diffs"
        truth_dir.mkdir(exist_ok=True)
        d0 = time[0].strftime("%Y%m%d")
        truth_filenames = [
            truth_dir / f"{d0}_{date.strftime('%Y%m%d')}.int.tif" for date in time[1:]
        ]
        truth_profile = slc_profile.copy() | {"dtype": "float32", "nbits": "16"}
        # truth_profile["transform"][0] /= inps.multilook_truth[1]  # x looks
        # truth_profile["transform"][4] /= inps.multilook_truth[0]  # y looks

        for filename in truth_filenames:
            with rio.open(filename, "w", **truth_profile) as dst:
                pass

    b_iter = list(
        iter_blocks(
            arr_shape=shape2d,
            block_shape=inps.block_shape,
        )
    )
    key = random.key(seed)

    def _save_crlb_std_devs(
        C: NDArray | Array, outdir: Path, time: Sequence[datetime], num_looks: int
    ):
        crlb_std_devs = crlb.compute_lower_bound_std(C, num_looks=num_looks)
        # TODO: what format to save this in?
        out_crlb_file = outdir / "crlb_std_devs.csv"
        if not out_crlb_file.exists():
            dates = [t.strftime("%Y-%m-%d") for t in time]  # or '%Y-%m-%dT%H:%M:%S'
            data = np.column_stack((dates, crlb_std_devs))
            np.savetxt(
                out_crlb_file,
                data,
                delimiter=",",
                fmt="%s",
                header="date,crlb_std_dev_radians",
                comments="",
            )

    if not using_global_coh:
        assert inps.custom_covariance is not None
        C_arrays = inps.custom_covariance.to_array(x_arr)
        _save_crlb_std_devs(C_arrays, outdir, time, inps.crlb_num_looks)
    else:
        # Each pixel has unique coherence matrix, so unique CRLBs
        output_crlb_filenames = [
            output_slc_dir / f"{date.strftime('%Y%m%d')}.crlb.tif" for date in time
        ]
        for filename in output_crlb_filenames:
            with rio.open(filename, "w", **(slc_profile | {"dtype": "float32"})) as dst:
                pass

    logger.info("Simulating correlated noise")
    for rows, cols in tqdm(b_iter):
        if using_global_coh:
            amps, rhos, taus, seasonal_A, seasonal_B, seasonal_mask = (
                load_coherence_files(coherence_files, rows, cols)
            )
        else:
            amps = None
        if verbose:
            tqdm.write(f"Simulating correlated noise for {rows}, {cols}")
        if files:
            propagation_phase = load_current_phase(files, rows, cols)
        else:  # all zeros
            propagation_phase = np.zeros(
                (len(x_arr), rows.stop - rows.start, cols.stop - cols.start),
                dtype=np.float32,
            )

        key, subkey = random.split(key)
        if using_global_coh:
            C_arrays = covariance.simulate_coh_stack(
                time=x_arr,
                gamma_inf=rhos,
                # the global coherence raster model assumes gamma0=1
                gamma0=0.99 * np.ones_like(rhos),
                Tau0=taus,
                seasonal_A=seasonal_A,
                seasonal_B=seasonal_B,
                seasonal_mask=seasonal_mask,
            )
            crlb_std_devs = covariance.compute_crlb_batch(
                C_arrays=C_arrays, num_looks=inps.crlb_num_looks
            )

        noisy_stack = covariance.make_noisy_samples_jax(
            subkey, C=C_arrays, defo_stack=propagation_phase, amplitudes=amps
        )

        window = rasterio.windows.Window.from_slices(rows, cols)
        for filename, layer in zip(output_slc_filenames, noisy_stack, strict=False):
            with rio.open(filename, "r+") as dst:
                dst.write(layer, 1, window=window)

        if using_global_coh:
            for filename, layer in zip(
                output_crlb_filenames, crlb_std_devs, strict=False
            ):
                with rio.open(filename, "r+") as dst:
                    dst.write(layer, 1, window=window)

        if inps.include_summed_truth:
            for filename, layer in zip(
                truth_filenames, propagation_phase[1:], strict=False
            ):
                with rio.open(filename, "r+") as dst:
                    # Sign convention for `phi = -4pi * r` flips this:
                    phase_diff = -1 * (layer - propagation_phase[0])
                    dst.write(phase_diff, 1, window=window)

    return output_slc_filenames


def load_coherence_files(
    coherence_files: list[Path], rows: slice, cols: slice
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load the coherence rasters for a block of pixels.

    Parameters
    ----------
    coherence_files : list[Path]
        List of paths to the coherence rasters.
    rows : slice
        Row slice to extract.
    cols : slice
        Column slice to extract.

    """
    # amp_file, rho_file, tau_file, seasonal_A_file, seasonal_B_file, seasonal_mask_file
    assert len(coherence_files) == 6
    out_arrays = []
    for f in coherence_files:
        with rio.open(f) as src:
            data: np.ndarray = src.read(
                1, window=rasterio.windows.Window.from_slices(rows, cols), masked=True
            )
            if data.dtype == np.uint8:
                data = data.filled(0).astype(bool)
            else:
                data = data.filled()
            out_arrays.append(data)
    return tuple(out_arrays)


def create_ramps(
    shape2d: tuple[int, int],
    num_days: int,
    out_hdf5: PathOrStr,
    amplitude: float = 1,
    overwrite: bool = False,
):
    """Create a stack of ramp phase data for a given shape and number of days.

    Parameters
    ----------
    shape2d : tuple[int, int]
        The 2D shape of the ramp data.
    num_days : int
        The number of days to generate ramp data for.
    out_hdf5 : PathOrStr
        The output HDF5 file path to save the ramp data.
    amplitude : float, optional
        The maximum amplitude of the ramp, by default 1.
    overwrite : bool, optional
        Whether to overwrite the output file if it already exists, by default False.

    Returns
    -------
    None
        The ramp data is saved to the specified HDF5 file.

    """
    shape3d = (num_days, *shape2d)
    rotations = np.random.randint(0, 360, size=(num_days,))
    if Path(out_hdf5).exists() and not overwrite:
        logger.info(f"Not overwriting {out_hdf5}")
        return

    with h5py.File(out_hdf5, "w") as hf:
        dset = hf.create_dataset("data", shape=shape3d, dtype="float32", **HDF5_KWARGS)
        for idx, r in enumerate(rotations):
            with logging_redirect_tqdm():
                logger.debug("Making ramp %s", idx)
                ramp_phase = deformation.ramp(
                    shape=shape2d, amplitude=amplitude, rotate_degrees=r
                )
                logger.debug("Saving....")
                round_mantissa(ramp_phase, significant_bits=8)
                dset[idx] = ramp_phase


def create_turbulence(
    shape2d: tuple[int, int],
    num_days: int,
    out_hdf5: PathOrStr,
    overwrite: bool = False,
    max_amplitude: float = 1.0,
    resolution: tuple[float, float] = (30, 30),
) -> None:
    """Create a stack of turbulent atmospheric noise.

    Parameters
    ----------
    shape2d : tuple[int, int]
        The 2D shape (rows, cols) of the output data.
    num_days : int
        The number of days to generate turbulence data for.
    out_hdf5 : PathOrStr
        The output HDF5 file path to save the turbulence data.
    overwrite : bool, optional
        Whether to overwrite the output file if it already exists, by default False.
    max_amplitude : float, optional
        The maximum amplitude of the turbulence, by default 1.0.
    resolution : tuple[float, float], optional
        The resolution (in meters) of the turbulence simulation, by default [30, 30]

    Returns
    -------
    None
        The turbulence data is saved to the specified HDF5 file.

    """
    if Path(out_hdf5).exists() and not overwrite:
        logger.info(f"Not overwriting {out_hdf5}")
        return
    shape3d = (num_days, *shape2d)
    max_amp_meters = max_amplitude / METERS_TO_PHASE
    # Since it's slow to simulate a huge turbulence field at 10 meters, and it's
    # generally not necessary to go that high res, we'll use 60 meters, then upsample
    # to full scale
    res_y, res_x = resolution
    with h5py.File(out_hdf5, "w") as hf:
        dset = hf.create_dataset("data", shape=shape3d, dtype="float32", **HDF5_KWARGS)
        for idx in tqdm(list(range(num_days))):
            with logging_redirect_tqdm():
                logger.debug("Making turbulence %s", idx)
                turb = turbulence.simulate(
                    shape=shape2d, resolution=res_y, max_amp=max_amp_meters
                )
                turb *= METERS_TO_PHASE

                round_mantissa(turb, significant_bits=8)
                logger.debug("Saving....")
                dset[idx] = turb
            # dset.write_direct(turb, dest_sel=idx) # needs to be contiguous


def create_defo_stack(
    shape: tuple[int, int, int],
    sigma: float,
    out_hdf5: PathOrStr,
    max_amplitude: float = 1,
    overwrite: bool = False,
) -> None:
    """Create the time series of deformation to add to each SAR date.

    Parameters
    ----------
    shape : tuple[int, int, int]
        Shape of the deformation stack (num_time_steps, rows, cols).
    sigma : float
        Standard deviation of the Gaussian deformation.
    max_amplitude : float, optional
        Maximum amplitude of the final deformation. Defaults to 1.
    out_hdf5 : PathOrStr | None, optional
        Path to output HDF5 file, by default None.
    overwrite : bool, optional
        Whether to overwrite the output file if it already exists, by default False.

    """
    from .deformation import gaussian

    if Path(out_hdf5).exists() and not overwrite:
        logger.info(f"Not overwriting {out_hdf5}")
        return

    num_time_steps, rows, cols = shape
    shape2d = (rows, cols)
    # Get shape of deformation in final form (normalized to 1 max)
    final_defo = gaussian(shape=shape2d, sigma=sigma).reshape((1, *shape2d))
    final_defo *= max_amplitude / np.max(final_defo)
    # Broadcast this shape with linear evolution
    time_evolution = np.linspace(0, 1, num=num_time_steps)

    with h5py.File(out_hdf5, "w") as hf:
        dset = hf.create_dataset("data", shape=shape, dtype="float32", **HDF5_KWARGS)
        for idx, t in tqdm(list(enumerate(time_evolution))):
            with logging_redirect_tqdm():
                logger.debug("Making deformation %s", idx)
                data = final_defo * t
                round_mantissa(data, significant_bits=8)
                logger.debug("Saving....")
                dset[idx] = data


def fetch_dem(bounds: Bbox, output_dir: Path, upsample_factor: tuple[int, int]):
    """Download and stitch a DEM (Digital Elevation Model) for the given bounding box.

    Parameters
    ----------
    bounds : Bbox
        The bounding box to download the DEM for.
    output_dir : Path
        The directory to save the DEM file to.
    upsample_factor : tuple[int, int]
        The factors to upsample the DEM in the y and x dimensions.

    """
    from sardem import cop_dem

    cop_dem.download_and_stitch(
        output_name=output_dir / "dem.tif",
        bbox=bounds,
        yrate=upsample_factor[0],
        xrate=upsample_factor[1],
        output_format="GTiff",
        output_type="float32",
    )
