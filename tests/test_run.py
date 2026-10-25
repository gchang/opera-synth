from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import rasterio

from synth.config import SimulationInputs
from synth.core import create_simulation_data

DATA_FILE = Path(__file__).parent / "sample.json"


def test_run(tmp_path):
    with open(DATA_FILE) as f:
        inputs = SimulationInputs.model_validate_json(f.read())
        inputs.output_dir = tmp_path
        create_simulation_data(inputs)

    # Validate expected output structure exists
    output_dir = Path(inputs.output_dir)
    slcs_dir = output_dir / "slcs"
    layers_dir = output_dir / "input_layers"
    truth_dir = layers_dir / "truth_unwrapped_diffs"

    # Check directory structure
    assert slcs_dir.exists(), "SLCs directory should exist"
    assert layers_dir.exists(), "Input layers directory should exist"
    assert truth_dir.exists(), "Truth unwrapped diffs directory should exist"

    # Validate SLC files - should have 6 dates (num_dates from config)
    slc_files = list(slcs_dir.glob("*.slc.tif"))
    assert len(slc_files) == 6, f"Expected 6 SLC files, got {len(slc_files)}"

    # Check SLC file properties
    for slc_file in slc_files:
        with rasterio.open(slc_file) as src:
            assert (
                np.dtype(src.dtypes[0]) == np.complex64
            ), f"SLC file {slc_file} should be complex64"
            assert src.count == 1, f"SLC file {slc_file} should have 1 band"
            assert (
                src.width > 0 and src.height > 0
            ), f"SLC file {slc_file} should have valid dimensions"

    # Validate input layer H5 files
    expected_h5_files = ["turbulence.h5", "deformation.h5", "phase_ramps.h5"]
    for h5_filename in expected_h5_files:
        h5_file = layers_dir / h5_filename
        assert h5_file.exists(), f"Expected H5 file {h5_file} should exist"

        with h5py.File(h5_file, "r") as f:
            assert len(f.keys()) > 0, f"H5 file {h5_file} should contain data"

    # Validate truth unwrapped diff files - should have 5 files (num_dates - 1)
    truth_files = list(truth_dir.glob("*.int.tif"))
    assert len(truth_files) == 5, f"Expected 5 truth files, got {len(truth_files)}"

    # Check truth file properties
    for truth_file in truth_files:
        with rasterio.open(truth_file) as src:
            assert (
                np.dtype(src.dtypes[0]) == np.float32
            ), f"Truth file {truth_file} should be float32"
            assert src.count == 1, f"Truth file {truth_file} should have 1 band"
            assert (
                src.width > 0 and src.height > 0
            ), f"Truth file {truth_file} should have valid dimensions"

    # Validate CRLB CSV file
    crlb_file = output_dir / "crlb_std_devs.csv"
    assert crlb_file.exists(), "CRLB CSV file should exist"

    # Check CRLB CSV content
    df = pd.read_csv(crlb_file)
    assert len(df) == 6, f"CRLB CSV should have 6 rows, got {len(df)}"
    assert "date" in df.columns, "CRLB CSV should have 'date' column"
    assert (
        "crlb_std_dev_radians" in df.columns
    ), "CRLB CSV should have 'crlb_std_dev_radians' column"
    assert df["crlb_std_dev_radians"][0] == 0, "CRLB first value should be 0"
    assert all(df["crlb_std_dev_radians"][1:] > 0), "CRLB values should be positive"
