import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import jax.numpy as jnp
from jax import Array
from numpy.typing import ArrayLike
from pydantic import BaseModel, Field

from ._types import Bbox, RhoOption


class CustomCoherence(BaseModel):
    """Class for one custom coherence matrix to use in simulations."""

    gamma_inf: float = Field(
        ..., ge=0.0, le=1.0, description="Asymptotic coherence value"
    )
    tau0: float = Field(
        ..., ge=0, description="Coherence decay time constant (in days)"
    )
    gamma0: float = Field(1.0, ge=0.0, le=1.0, description="Initial coherence value")
    seasonal_A: float | None = Field(
        None, ge=0.0, le=1.0, description="Seasonal A coefficient"
    )
    seasonal_B: float | None = Field(
        None, ge=0.0, le=1.0, description="Seasonal B coefficient"
    )

    def to_array(self, time: ArrayLike) -> Array:
        """Create the complex coherence matrix as a JAX Array."""
        time_arr = jnp.asarray(time)
        num_time = time_arr.shape[0]
        temp_baselines = jnp.abs(time_arr[None, :] - time_arr[:, None])
        gamma = (self.gamma0 - self.gamma_inf) * jnp.exp(
            -temp_baselines / self.tau0
        ) + self.gamma_inf
        phase_term = jnp.exp(1j * 0)

        if (A := self.seasonal_A) is not None and (B := self.seasonal_B) is not None:
            seasonal_factor = (
                A + B * jnp.cos(2 * jnp.pi * temp_baselines / 365.25)
            ) ** 2
            # Ensure it is a valid coherence multiplier
            seasonal_factor = jnp.clip(seasonal_factor, 0, 1)
            gamma = gamma * seasonal_factor

        C = gamma * phase_term
        rl, cl = jnp.tril_indices(num_time, k=-1)
        C = C.at[rl, cl].set(jnp.conj(C.T)[rl, cl])

        # Reset the diagonals of each pixel to 1
        rs, cs = jnp.diag_indices(num_time)
        C = C.at[rs, cs].set(1.0)

        return C


class SimulationInputs(BaseModel):
    """Parameters describing simulation data to generate."""

    output_dir: Path = Field(
        default_factory=Path, description="Directory where output files will be saved."
    )

    bounding_box: Bbox = Field(
        ...,
        description=(
            "(left, bottom, right, top) in degrees EPSG:4326, defining the geographic"
            " bounding box."
        ),
    )

    start_date: datetime = Field(
        default=datetime(2020, 1, 1),
        description="The starting date for the simulation.",
    )

    dt: int = Field(
        default=12, ge=1, le=365, description="Time step in days for the simulation."
    )

    num_dates: int = Field(
        default=20, ge=2, le=1000, description="Total number of dates to simulate."
    )

    res_y: float = Field(
        default=15, ge=1, description="Pixel spacing along Y direction (meters)."
    )
    res_x: float = Field(
        default=15, ge=1, description="Pixel spacing along X direction (meters)."
    )

    include_turbulence: bool = Field(
        default=True, description="Flag to include turbulence in the simulation."
    )
    max_turbulence_amplitude: float = Field(
        default=5, description="Maximum amplitude for turbulence effects."
    )

    include_deformation: bool = Field(
        default=True,
        description="Flag to include deformation effects in the simulation.",
    )

    max_defo_amplitude: float = Field(
        default=5, description="Maximum amplitude for deformation effects."
    )
    defo_width_fraction: float = Field(
        default=0.2,
        description=(
            "Fraction of the total image width that the deformation bowl spans."
        ),
    )

    include_ramps: bool = Field(
        default=True, description="Flag to include ramp effects in the simulation."
    )

    max_ramp_amplitude: float = Field(
        default=1.0, description="Maximum amplitude for ramp effects."
    )

    include_stratified: bool = Field(
        default=False,
        description="Flag to include stratified effects in the simulation.",
    )

    rho_transform: RhoOption = Field(
        default=RhoOption.SHRUNK, description="Method for rho transformation."
    )

    include_summed_truth: bool = Field(
        default=True, description="Flag to include the summed true simulation data."
    )

    block_shape: tuple[int, int] = Field(
        default=(128, 128),
        description=(
            "Size of (rows, cols) to process at one time when generating covariance"
            " matrices/samples. You must have enough memory for several matrices of"
            " shape `(*block_shape, inps.num_dates, inps.num_dates)`."
        ),
    )

    include_decorrelation: bool = Field(
        default=True,
        description=(
            "Flag to include decorrelation in the simulation. Can be from either a"
            " manual, `custom_covariance`, or using the real Sentinel-1 Global"
            " Coherence dataset based on the area of interest."
        ),
    )
    custom_covariance: CustomCoherence | None = Field(
        default=None,
        description=(
            "Custom covariance parameters to use if not using the global dataset."
        ),
    )
    crlb_num_looks: int = Field(
        default=1, ge=1, description="Number of looks to use for CRLB computation."
    )

    @property
    def datetimes(self) -> list[datetime]:
        """Return the time array."""
        return [
            self.start_date + idx * timedelta(days=self.dt)
            for idx in range(self.num_dates)
        ]

    @property
    def days_since_start(self) -> list[int]:
        """Return a list of integers for the number of days since start_date."""
        return [(t - self.start_date).days for t in self.datetimes]

    def create_profile(self) -> dict[str, Any]:
        """Create a rasterio profile based on SimulationInputs.

        Returns
        -------
        dict[str, Any]
            A dictionary containing the profile information.

        """
        from pyproj import Geod

        # Extract bounding box
        left, bottom, right, top = (
            self.bounding_box.left,
            self.bounding_box.bottom,
            self.bounding_box.right,
            self.bounding_box.top,
        )

        # Calculate width and height in meters
        geod = Geod(ellps="WGS84")
        _, _, width_m = geod.inv(left, (top + bottom) / 2, right, (top + bottom) / 2)
        _, _, height_m = geod.inv((left + right) / 2, bottom, (left + right) / 2, top)

        # Calculate width and height in pixels
        width = math.ceil(width_m / self.res_x)
        height = math.ceil(height_m / self.res_y)

        # Calculate actual resolution
        res_x = (right - left) / width
        res_y = (top - bottom) / height

        profile = {
            "height": height,
            "width": width,
            "count": 1,
            "compress": "lzw",
            "transform": [res_x, 0.0, left, 0.0, -res_y, top, 0.0, 0.0, 1.0],
            "crs": "EPSG:4326",
            "driver": "GTiff",
            "dtype": "complex64",
            "interleave": "band",
            "nodata": 0.0,
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
        }

        return profile

    def get_custom_covariance_array(self) -> Array:
        """Return the custom covariance matrix."""
        if self.custom_covariance is None:
            raise ValueError("No custom covariance matrix provided.")
        return self.custom_covariance.to_array(self.days_since_start)
