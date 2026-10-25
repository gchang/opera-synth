from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio as rio

from .utils import get_dates


def process_coherence_data(
    directory: str | Path,
    by_date: bool = False,
    subsample: int = 1,
    difference_dir: str | Path = "differences",
    trim: int = 20,
) -> pd.DataFrame:
    """Process InSAR coherence data from a directory and return analysis dataframe.

    Parameters
    ----------
    directory : str or Path
        Path to the directory containing the InSAR data with the following structure:
        - directory/
            - differences/*.tif
            - interferograms/temporal_coherence*.tif
            - interferograms/similarity*.tif
    by_date : bool, optional
        If True, return a dataframe with one row per date, by default False
    subsample : int, optional
        Subsample the data by this factor, by default 1
    difference_dir : str or Path, optional
        Directory containing the difference files, by default "differences"

    Returns
    -------
    pd.DataFrame
        DataFrame containing columns:
        - temporal_coherence: Temporal coherence values for each pixel
        - similarity: Phase similarity values for each pixel
        - rmse: Root mean square error for each pixel

    Notes
    -----
    The function:
    1. Reads the difference files as a time series
    2. Calculates RMSE across time for each pixel
    3. Combines with temporal coherence and similarity metrics

    The returned DataFrame has one row per pixel.

    """
    main_dir = Path(directory)
    diff_dir = main_dir / difference_dir

    # Read data
    # reader = io.RasterStackReader.from_file_list(sorted(diff_dir.glob("*tif")))
    file_list = sorted(diff_dir.glob("*tif"))

    temp_coh = _load_gdal(
        next(iter(main_dir.glob("interferograms/temporal_coherence_[0-9]*.tif"))),
        subsample_factor=subsample,
    )
    sim = _load_gdal(
        next(iter(main_dir.glob("interferograms/similarity_[0-9]*.tif"))),
        subsample_factor=subsample,
    )
    # sim = _load_gdal(main_dir / "interferograms/similarity.tif")
    # sim = _load_gdal(next(sorted(Path(main_dir / "linked_phase")
    #                                .rglob("similarity_*.tif"))))
    # AVERAGE sim... since this is average temp coh?
    # Or should i just do a single one...
    # print(sorted(Path(main_dir / "linked_phase").rglob("similarity_*.tif")))

    # Process differences
    pixels = np.stack(
        [_load_gdal(f, subsample_factor=subsample) for f in file_list], axis=0
    )
    pixels = pixels[..., trim:-trim, trim:-trim]
    pixels = pixels.reshape(pixels.shape[0], -1)

    if not by_date:
        rmse_by_pixel = np.sqrt(np.mean(pixels * pixels.conj(), axis=0))
        return pd.DataFrame(
            {
                "temporal_coherence": temp_coh.ravel(),
                "similarity": sim.ravel(),
                "rmse": rmse_by_pixel,
            }
        )
    else:
        # Save the rmse by pixel for each date
        rmse_by_by_date = np.sqrt(np.mean(pixels * pixels.conj(), axis=1))

        date_pairs = [Path(p).stem for p in file_list]
        dates = [get_dates(f)[1] for f in date_pairs]
        return pd.DataFrame(
            {
                "temporal_coherence": np.mean(temp_coh),
                "similarity": np.mean(sim),
                "rmse": rmse_by_by_date,
                "date_pair": date_pairs,
            },
            index=dates,
        )


def plot_coherence_analysis(
    df: pd.DataFrame,
    col: Literal["temporal_coherence", "similarity"] = "temporal_coherence",
    xlim: tuple[float | None, float | None] | None = (0, 1),
    ax: plt.Axes | None = None,
    add_colorbar: bool = False,
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    """Create an improved visualization of temporal coherence versus RMSE relationships.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns col and "rmse"
    col : Literal["temporal_coherence", "similarity"]
        Column to plot
    xlim : tuple[float | None, float | None] | None, optional
        X-axis limits, by default (0, 1)
    ax : plt.Axes | None, optional
        Axes object to plot on, by default None
    add_colorbar : bool, optional
        Whether to add a colorbar, by default False

    Returns
    -------
    tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]
        Figure and axes objects for further customization

    """
    from scipy.stats import gaussian_kde

    # Filter out problematic values
    mask = df[col] != 0
    df_filtered = df[mask].copy().dropna()
    label = " ".join(col.split("_")).title()
    if xlim is None:
        xlim = (0, 1) if col == "temporal_coherence" else (-1, 1)

    # Create figure
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    # Scatter plot with density-dependent alpha
    x = df_filtered[col]
    y = df_filtered["rmse"]

    # Calculate point density for color mapping
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)

    # Sort points by density for better visualization
    idx = z.argsort()
    x, y, z = x.iloc[idx], y.iloc[idx], z[idx]

    scatter = ax.scatter(x, y, c=z, s=1, alpha=0.5, cmap="viridis")
    ax.set_xlabel(f"{label}")
    ax.set_ylabel("RMSE")
    ax.set_xlim(xlim)

    if add_colorbar:
        fig.colorbar(scatter, ax=ax, label="Point Density")

    return fig, ax


def plot_quality_density(
    df: pd.DataFrame,
    bins: int = 100,
    col: Literal["temporal_coherence", "similarity"] = "temporal_coherence",
    y_col: Literal["rmse", "temporal_coherence", "similarity"] = "rmse",
    ax: plt.Axes | None = None,
    cmap="blues",
    add_colorbar: bool = False,
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    """Plot temporal coherence versus RMSE.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns 'temporal_coherence' and 'rmse'
    bins : int, optional
        Number of bins for the 2D histogram, by default 100
    col : Literal["temporal_coherence", "similarity"]
        Column to plot
    y_col : Literal["rmse", "temporal_coherence", "similarity"]
        Column to plot
    ax : plt.Axes | None, optional
        Axes object to plot on, by default None
    cmap : str, optional
        Colormap to use, by default "blues"
    add_colorbar : bool, optional
        Whether to add a colorbar, by default False

    Returns
    -------
    tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]
        Figure and axes objects

    """
    from matplotlib.colors import LogNorm

    label = " ".join(col.split("_")).title()
    ylabel = y_col.upper() if y_col == "rmse" else " ".join(y_col.split("_")).title()
    # Filter out problematic values
    mask = df[col] != 0
    df_filtered = df[mask].dropna()

    # Create figure
    if ax is None:
        fig, ax = plt.subplots(figsize=(15, 6))
    else:
        fig = ax.figure
    # 2D histogram plot
    hist, xedges, yedges = np.histogram2d(
        df_filtered[col], df_filtered[y_col], bins=bins, density=True
    )

    # Calculate bin centers
    xcenters = (xedges[:-1] + xedges[1:]) / 2
    ycenters = (yedges[:-1] + yedges[1:]) / 2

    # Plot as a heatmap with log scale for better visibility
    pcm = ax.pcolormesh(
        xedges,
        yedges,
        hist.T,  # Transpose because numpy and matplotlib handle dimensions differently
        norm=LogNorm(vmin=np.percentile(hist[hist > 0], 5), vmax=hist.max(), clip=True),
        cmap=cmap,
        skip_autolev=True,  # proplot only
    )
    if add_colorbar:
        fig.colorbar(pcm, ax=ax, label="Point Density (log scale)")

    # Add contour lines for additional clarity
    X, Y = np.meshgrid(xcenters, ycenters)
    ax.contour(X, Y, hist.T, levels=5, colors="white", alpha=0.3, linewidths=0.5)

    ax.set_xlabel(label)
    ax.set_ylabel(ylabel)
    # ax.set_xlim(0, 1)
    return fig, ax


def plot_boxplot(
    df: pd.DataFrame,
    min_coherence: float = 0.01,
    min_rmse: float = 1e-6,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Create a boxplot showing temporal coherence bins vs RMSE.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns 'temporal_coherence' and 'rmse'
    min_coherence : float, optional
        Minimum coherence value to include, by default 0.01
    min_rmse : float, optional
        Minimum RMSE value to include, by default 1e-6
    ax : plt.Axes | None, optional
        Axes object to plot on, by default None

    Returns
    -------
    tuple[plt.Figure, plt.Axes]
        Figure and axes objects

    """
    # Filter out problematic values
    mask = (df["temporal_coherence"] >= min_coherence) & (df["rmse"] >= min_rmse)
    df_filtered = df[mask]
    # Box plot using efficient binned statistics
    coherence_bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    bin_labels = ["0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]

    # Calculate statistics for each bin
    bin_stats = []
    for i in range(len(coherence_bins) - 1):
        mask = (df_filtered["temporal_coherence"] >= coherence_bins[i]) & (
            df_filtered["temporal_coherence"] < coherence_bins[i + 1]
        )
        bin_data = df_filtered.loc[mask, "rmse"]

        # Calculate key statistics
        q1, median, q3 = np.percentile(bin_data, [25, 50, 75])
        bin_stats.append(
            {
                "bin": bin_labels[i],
                "median": median,
                "q1": q1,
                "q3": q3,
                "whislo": q1 - 1.5 * (q3 - q1),
                "whishi": q3 + 1.5 * (q3 - q1),
            }
        )
    if ax is None:
        fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    else:
        fig = ax.figure
    # Plot box plot from statistics
    box_data = pd.DataFrame(bin_stats)
    bp = ax.boxplot(
        [np.array([]) for _ in range(len(bin_labels))],  # Empty data
        positions=range(len(bin_labels)),
        medianprops={"color": "red"},
        showfliers=False,
    )

    # Set the precomputed statistics
    for i in range(len(bin_labels)):
        bp["medians"][i].set_ydata([box_data.iloc[i]["median"]] * 2)
        bp["boxes"][i].set_ydata(
            [
                box_data.iloc[i]["q1"],
                box_data.iloc[i]["q1"],
                box_data.iloc[i]["q3"],
                box_data.iloc[i]["q3"],
                box_data.iloc[i]["q1"],
            ]
        )
        bp["whiskers"][i * 2].set_ydata(
            [box_data.iloc[i]["whislo"], box_data.iloc[i]["q1"]]
        )
        bp["whiskers"][i * 2 + 1].set_ydata(
            [box_data.iloc[i]["q3"], box_data.iloc[i]["whishi"]]
        )
        bp["caps"][i * 2].set_ydata([box_data.iloc[i]["whislo"]] * 2)
        bp["caps"][i * 2 + 1].set_ydata([box_data.iloc[i]["whishi"]] * 2)

    ax.set_xticklabels(bin_labels)
    ax.set_xlabel("Temporal Coherence Bin")
    ax.set_ylabel("RMSE")
    ax.set_title("RMSE Distribution by Coherence Bin")

    plt.tight_layout()
    return fig, ax


def _load_gdal(filename: Path | str, subsample_factor: int = 1) -> np.ndarray:
    with rio.open(filename) as src:
        data = src.read(
            1,
            out_shape=(
                1,
                src.height // subsample_factor,
                src.width // subsample_factor,
            ),
        )
    return data


def plot_differences(directories):  # noqa: D103
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(directories)))

    diffs = []
    errors = []
    dfs = []
    for directory, color in zip(directories, colors, strict=False):
        # Set up paths
        main_dir = Path(directory)
        diff_dir = main_dir / "differences"

        # Read data
        file_list = sorted(diff_dir.glob("*tif"))
        dates = [get_dates(f)[1] for f in file_list]
        temp_coh = _load_gdal(main_dir / "interferograms/temporal_coherence.tif")
        sim = _load_gdal(main_dir / "interferograms/similarity.tif")

        # TODO: process in patches?
        pixels = np.stack([_load_gdal(f) for f in file_list], axis=0)
        pixels = pixels.reshape(pixels.shape[0], -1)

        # Calculate product
        # cur_pixel_diffs = np.angle(np.exp(1j * (pixels - p0[:, None])))
        cur_pixel_diffs = pixels

        # Plot on first subplot
        # ax1.plot(dates, cur_pixel_diffs, lw=0.5, alpha=0.2, color=color)

        # Plot on second subplot
        rmse_by_date = rmse(cur_pixel_diffs, axis=1)
        ax2.plot(dates, rmse_by_date, color=color, label=Path(directory).name)

        rmse_by_pixel = rmse(cur_pixel_diffs, axis=0)
        df = pd.DataFrame(
            data={
                "temporal_coherence": temp_coh.ravel(),
                "similarity": sim.ravel(),
                "rmse": rmse_by_pixel,
            }
        )

        errors.append(rmse_by_date)
        diffs.append(cur_pixel_diffs)
        dfs.append(df)

    # Configure first subplot
    ax1.set_title("Errors by date")
    ax1.set_ylabel("Wrapped Error [radians]")

    # Configure second subplot
    ax2.set_ylabel("RMSE [radians]")
    ax2.legend()

    plt.tight_layout()
    plt.show()
    # return errors, diffs
    return errors, dfs


def rmse(arr, axis=1):
    """Calculate the Root Mean Square Error (RMSE) along a specified axis."""
    return np.sqrt(np.mean(arr * arr.conj(), axis=axis))


def plot_temporal_coherence_vs_rmse(df):
    """Visualize temporal coherence versus RMSE relationships.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing at least two columns:
        - 'temporal_coherence': temporal coherence values
        - 'rmse': RMSE values

    Notes
    -----
    Creates a figure with four subplots:
    1. Scatter plot of temporal coherence vs RMSE with color gradient
    2. Box plot showing RMSE distribution across temporal coherence bins
    3. Violin plot showing RMSE distribution across temporal coherence bins
    4. Heatmap showing density of temporal coherence vs RMSE values

    The temporal coherence values are binned into 5 categories:
    0-0.2, 0.2-0.4, 0.4-0.6, 0.6-0.8, and 0.8-1.0

    The visualization uses seaborn for enhanced statistical plotting.

    """
    import seaborn as sns

    # Create bins for temporal coherence
    df["coherence_bin"] = pd.cut(
        df["temporal_coherence"],
        bins=5,
        labels=["0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"],
    )

    # Set up the matplotlib figure
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle("Temporal Coherence vs RMSE Analysis", fontsize=16)

    # 1. Scatter plot with color gradient
    sns.scatterplot(
        data=df,
        x="temporal_coherence",
        y="rmse",
        hue="temporal_coherence",
        palette="viridis",
        ax=axes[0, 0],
    )
    axes[0, 0].set_title("Temporal Coherence vs RMSE")

    # 2. Box plot for each bin
    sns.boxplot(data=df, x="coherence_bin", y="rmse", ax=axes[0, 1])
    axes[0, 1].set_title("RMSE for Temporal Coherence Bins")
    axes[0, 1].set_xlabel("Temporal Coherence Bin")

    # 3. Violin plot for each bin
    sns.violinplot(data=df, x="coherence_bin", y="rmse", ax=axes[1, 0])
    axes[1, 0].set_title("RMSE Distribution for Temporal Coherence Bins")
    axes[1, 0].set_xlabel("Temporal Coherence Bin")

    # 4. Heatmap of average RMSE for coherence-rmse grid
    heatmap_data = df.pivot_table(
        values="rmse",
        index=pd.cut(df["temporal_coherence"], bins=10),
        columns=pd.cut(df["rmse"], bins=10),
        aggfunc="count",
    )
    sns.heatmap(heatmap_data, ax=axes[1, 1], cmap="YlOrRd")
    axes[1, 1].set_title("Density of Temporal Coherence vs RMSE")
    axes[1, 1].set_xlabel("RMSE Bins")
    axes[1, 1].set_ylabel("Temporal Coherence Bins")

    plt.tight_layout()
    plt.show()


def similarity_vs_temporal_coherence(df):
    """Create hexbin plots comparing similarity and temporal coherence against RMSE.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing three columns:
        - 'temporal_coherence': temporal coherence values
        - 'similarity': cosine similarity values
        - 'rmse': RMSE values

    Notes
    -----
    Creates a figure with two hexbin plots:
    1. Temporal coherence vs RMSE
    2. Similarity vs RMSE

    The hexbin plots use logarithmic binning and a viridis colormap to show
    density of points. The similarity plot is constrained to the range [-0.1, 1.1]
    for better visualization.

    """
    # Set up the matplotlib figure
    fig, axes = plt.subplots(1, 2, figsize=(9, 6), squeeze=False)
    fig.suptitle("Similarity vs Temporal Coherence - RMSE Analysis", fontsize=16)

    # Scatter plots with color gradient
    ax = axes[0, 0]
    grid_kw = {
        "gridsize": [20, 20],
        "cmap": "viridis",
        "mincnt": 10,
        "bins": "log",
    }
    ax.hexbin(df["temporal_coherence"], df["rmse"], **grid_kw)
    ax.set_xlabel("temporal_coherence")
    ax.set_ylabel("rmse")

    ax = axes[0, 1]
    grid_kw["gridsize"][0] *= 2
    ax.hexbin(df["similarity"], df["rmse"], **grid_kw)
    ax.set_xlabel("Cosine similarity")
    ax.set_ylabel("RMSE")
    ax.set_xlim(-0.1, 1.1)
    fig.tight_layout()
