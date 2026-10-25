import argparse
import logging
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import rasterio as rio
from rasterio.windows import Window
from tqdm.auto import tqdm

from ._blocks import iter_blocks
from ._types import PathOrStr
from .utils import _setup_logging

logger = logging.getLogger(__name__)

OUTPUT_PROFILE_DEFAULTS = {
    "driver": "GTiff",
    "dtype": "float32",
    "nodata": np.nan,
    "count": 1,
    "tiled": True,
    "compress": "lzw",
    "nbits": "16",
}
BLOCK_SHAPE = (512, 512)


def compare_phase(
    phase_files: Iterable[PathOrStr],
    truth_unwrapped_diffs_dir: PathOrStr,
    output_dir: Path = Path("differences"),
    is_wrapped: bool = False,
    flip_sign: bool = False,
    temporal_coherence_file: PathOrStr | None = None,
    temporal_coherence_threshold: float = 0.7,
    conncomp_files: Iterable[PathOrStr] | None = None,
    exclude_zero_conncomps: bool = True,
    block_shape: tuple[int, int] = BLOCK_SHAPE,
) -> list[Path]:
    """Compare unwrapped or wrapped interferograms to synthetic deformation.

    This function compares a set of interferograms to synthetic deformation
    data, optionally excluding areas with zero connected components or
    low temporal coherence, and outputs the differences as GeoTIFF files.

    Parameters
    ----------
    phase_files : Iterable[PathOrStr]
        An iterable of paths to wrapped, or unwrapped, interferogram files.
    truth_unwrapped_diffs_dir: PathOrStr
        Directory containing the summed propagation phase to use as truth comparison.
    output_dir : Path, optional
        Directory where the difference files will be saved.
        Default is Path("differences").
    is_wrapped : bool, optional
        Indicate if the input files are wrapped phase. Default is False.
    flip_sign : bool, optional
        Flag to flip the sign of `phase_files`. Used for comparing `timeseries/`,
        where the sign has been flipped from `unwrapped/`
    temporal_coherence_file: Optional[PathOrStr], optional
        Path to a temporal coherence file, used to ignore portions of the results with
        low coherence. Default is None.
    temporal_coherence_threshold : float, optional
        If `temporal_coherence_file` is passed, the threshold used to ignore
        low coherence areas.
        Default is 0.7.
    conncomp_files : Optional[Iterable[PathOrStr]], optional
        An iterable of paths to connected component files corresponding to the
        unwrapped interferograms, if comparing unwrapped results. Default is None.
    exclude_zero_conncomps : bool, optional
        If True, areas with connected component == 0 are excluded from the comparison.
        Default is True.
    block_shape : tuple[int, int]
        Size of blocks to load during comparison.
        Default is (512, 512)
    reference_idx : int
        Index of reference phase. Each layer will subtract this index.
        For single reference interferograms, referenced to first date, this is 0.

    Returns
    -------
    list[Path]
        A list of paths to the output difference files.

    Raises
    ------
    ValueError
        If the number of phase files doesn't match the number of truth phase layers.

    """
    phase_file_list = list(phase_files)

    with rio.open(phase_file_list[0]) as src:
        shape2d = src.shape
        profile = src.profile | OUTPUT_PROFILE_DEFAULTS

    truth_files = sorted(Path(truth_unwrapped_diffs_dir).glob("2*.tif"))
    strides = _get_downsample_factor(shape2d, truth_files[0])
    row_strides, col_strides = strides

    ref_vals = []

    # Check if we're passing interferograms, or the linked-phase rasters
    # one fewer interferogram than phase-linked slc
    if (
        str(phase_file_list[0]).endswith(".slc.tif")
        and len(phase_file_list) == len(truth_files) + 1
    ):
        phase_file_list = phase_file_list[1:]  # Skip the empty first date

    for est_file, truth in zip(phase_file_list, truth_files, strict=True):
        with rio.open(est_file) as src_est, rio.open(truth) as src_true:
            rows, cols = slice(10, 15), slice(10, 15)
            window = Window.from_slices(rows, cols)
            val1 = src_est.read(1, window=window, masked=True)

            full_window = _get_full_window(rows, cols, strides)
            val2 = src_true.read(1, window=full_window, out_shape=val1.shape)
            if is_wrapped:
                val1 = np.angle(val1) if np.iscomplexobj(val1) else val1
                val2 = np.angle(val2) if np.iscomplexobj(val2) else val2
                ref_vals.append(
                    np.nanmean(np.angle(np.exp(1j * val2) * np.exp(-1j * val1)))
                )
            else:
                ref_vals.append(np.nanmean(val2 - val1))

    # Setup all empty output rasters
    output_dir.mkdir(exist_ok=True, parents=True)
    output_files = [output_dir / f"difference_{Path(f).name}" for f in phase_file_list]

    for filename in output_files:
        with rio.open(filename, "w", **profile) as dst:
            pass

    b_iter = list(iter_blocks(arr_shape=shape2d, block_shape=block_shape))
    for rows, cols in tqdm(b_iter):
        window = Window.from_slices(rows, cols)
        full_window = _get_full_window(rows, cols, strides)

        # Load temporal coherence mask if provided
        if temporal_coherence_file:
            with rio.open(temporal_coherence_file) as src:
                temp_coh = src.read(1, window=window)
            coh_mask = temp_coh < temporal_coherence_threshold

        for cur_truth_file, compare_file, out_file, ref_val in zip(
            truth_files, phase_file_list, output_files, ref_vals, strict=True
        ):
            with rio.open(compare_file) as src:
                cur_phase = src.read(1, window=window)
            with rio.open(cur_truth_file) as src:
                cur_truth = src.read(1, window=full_window, out_shape=cur_phase.shape)

            k = -1 if flip_sign else 1.0
            if is_wrapped:
                if np.iscomplexobj(cur_phase):
                    cur_phase = np.angle(cur_phase * np.exp(-1j * ref_val))
                else:
                    cur_phase = cur_phase - ref_val
                # cur_truth is already float
                assert not np.iscomplexobj(cur_truth)
                difference = np.angle(
                    np.exp(1j * k * cur_phase) * np.exp(-1j * cur_truth)
                )
            else:
                difference = k * (cur_phase - ref_val) - cur_truth
                difference -= difference.mean()

            # Apply masks
            if temporal_coherence_file:
                difference[coh_mask] = np.nan

            if conncomp_files and exclude_zero_conncomps:
                cc_f = next(iter(conncomp_files))
                with rio.open(cc_f) as src:
                    mask = src.read(1, window=window) == 0
                difference[mask] = np.nan

            with rio.open(out_file, "r+") as dst:
                dst.write(difference, indexes=1, window=window)

    return output_files


def _get_downsample_factor(downsampled_shape, full_res_file):
    with rio.open(full_res_file) as src:
        full_shape = src.shape

    row_strides = round(full_shape[-2] / downsampled_shape[0])
    col_strides = round(full_shape[-1] / downsampled_shape[1])
    return row_strides, col_strides


def _get_full_window(rows: slice, cols: slice, strides: tuple[int, int]):
    row_strides, col_strides = strides
    full_rows = slice(rows.start * row_strides, rows.stop * row_strides, row_strides)
    full_cols = slice(cols.start * col_strides, cols.stop * col_strides, col_strides)
    return Window.from_slices(full_rows, full_cols)


def _get_cli_args():
    parser = argparse.ArgumentParser(
        description="Compare phase files to synthetic deformation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-t",
        "--truth",
        "--truth-unwrapped-diffs-dir",
        type=Path,
        required=True,
        help=(
            "Path to directory containing the summed propagation phase created during"
            " simulation."
        ),
    )
    parser.add_argument(
        "--phase-files",
        nargs="+",
        type=Path,
        required=True,
        help="Paths to phase files (wrapped or unwrapped) to compare to truth.",
    )
    parser.add_argument(
        "--wrapped",
        action="store_true",
        help="Indicate if the input files are wrapped phase.",
    )
    parser.add_argument(
        "--flip-sign",
        action="store_true",
        help=(
            "Flag to flip the sign of `phase_files`. Used for comparing `timeseries/`,"
            " where the sign has been flipped from `unwrapped/`"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("differences"),
        help="Directory where the difference files will be saved",
    )
    parser.add_argument(
        "--conncomp-files",
        nargs="+",
        type=Path,
        help="Paths to connected component files (for unwrapped phase only)",
    )
    parser.add_argument(
        "--temporal-coherence-file",
        type=Path,
        help="Path to temporal coherence file",
    )
    parser.add_argument(
        "--temporal-coherence-threshold",
        type=float,
        default=0.7,
        help="Threshold for temporal coherence masking",
    )
    parser.add_argument(
        "--include-zero-conncomps",
        action="store_false",
        dest="exclude_zero_conncomps",
        help="Include areas with zero connected components in the comparison",
    )
    return parser.parse_args()


def main():
    """Run the comparison analysis on a set of phase files."""
    _setup_logging()

    args = _get_cli_args()

    output_files = compare_phase(
        phase_files=args.phase_files,
        truth_unwrapped_diffs_dir=args.truth,
        output_dir=args.output_dir,
        is_wrapped=args.wrapped,
        flip_sign=args.flip_sign,
        temporal_coherence_file=args.temporal_coherence_file,
        temporal_coherence_threshold=args.temporal_coherence_threshold,
        conncomp_files=args.conncomp_files if not args.wrapped else None,
        exclude_zero_conncomps=args.exclude_zero_conncomps,
    )

    logger.info(f"Comparison completed. {len(output_files)} output files processed.")
    logger.debug("Comparison completed.")
    for file in output_files:
        logger.debug(f"  {file}")


if __name__ == "__main__":
    main()
