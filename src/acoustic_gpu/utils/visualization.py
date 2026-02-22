"""
Plotting utilities for acoustic ray tracing and TFM imaging.

All functions return (Figure, Axes) and accept an optional `ax` argument
for embedding into multi-panel figures.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from typing import Optional


def plot_surface(
    surface,
    emitters: Optional[np.ndarray] = None,
    refraction_points: Optional[np.ndarray] = None,
    focuses: Optional[np.ndarray] = None,
    n_points: int = 300,
    title: str = "Surface Geometry",
    figsize: tuple[float, float] = (10, 5),
    ax: Optional[Axes] = None,
) -> tuple[Figure, Axes]:
    """Plot a 2D surface profile with optional elements and ray paths.

    Parameters
    ----------
    surface : SurfaceInterface
        Surface object with get_points() method.
    emitters : np.ndarray, optional
        Emitter positions, shape (N, 2).
    refraction_points : np.ndarray, optional
        Refraction point coordinates, shape (M, 2).
    focuses : np.ndarray, optional
        Focus point coordinates, shape (K, 2).
    n_points : int
        Number of sample points for surface discretization.
    title : str
        Plot title.
    figsize : tuple
        Figure size.
    ax : Axes, optional

    Returns
    -------
    fig, ax
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.get_figure()

    xS, zS = surface.get_points(n_points)
    ax.plot(xS * 1e3, zS * 1e3, "b-", linewidth=2, label="Surface")

    if emitters is not None:
        emitters = np.asarray(emitters)
        ax.scatter(emitters[:, 0] * 1e3, emitters[:, 1] * 1e3,
                   color="red", s=20, zorder=5, label="Elements")

    if refraction_points is not None:
        refraction_points = np.asarray(refraction_points)
        ax.scatter(refraction_points[:, 0] * 1e3, refraction_points[:, 1] * 1e3,
                   color="orange", s=10, alpha=0.6, zorder=4, label="Refraction pts")

    if focuses is not None:
        focuses = np.asarray(focuses)
        ax.scatter(focuses[:, 0] * 1e3, focuses[:, 1] * 1e3,
                   color="green", s=15, alpha=0.5, zorder=3, label="Focuses")

    ax.set_xlabel("x (mm)")
    ax.set_ylabel("z (mm)")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()

    return fig, ax


def plot_tof_table(
    tof: np.ndarray,
    extent: Optional[tuple[float, float, float, float]] = None,
    title: str = "Time-of-Flight Table",
    figsize: tuple[float, float] = (10, 4),
    element_idx: Optional[int] = None,
    ax: Optional[Axes] = None,
    cmap: str = "viridis",
) -> tuple[Figure, Axes]:
    """Plot a TOF table as a heatmap.

    Parameters
    ----------
    tof : np.ndarray
        TOF table. If 2D with shape (N_elements, N_pixels), can show
        a single element or all elements.
    extent : tuple, optional
        (x_min, x_max, z_min, z_max) in meters.
    title : str
        Plot title.
    element_idx : int, optional
        If provided, plot only this element's TOF row.
    ax : Axes, optional

    Returns
    -------
    fig, ax
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.get_figure()

    if element_idx is not None and tof.ndim >= 2:
        data = tof[element_idx]
    else:
        data = tof

    if extent is not None:
        ext_mm = (extent[0] * 1e3, extent[1] * 1e3, extent[2] * 1e3, extent[3] * 1e3)
    else:
        ext_mm = None

    if data.ndim == 1:
        ax.plot(data * 1e6, "b-")
        ax.set_xlabel("Pixel index")
        ax.set_ylabel("TOF (μs)")
    else:
        im = ax.imshow(
            data * 1e6,
            aspect="auto", cmap=cmap,
            extent=ext_mm,
        )
        plt.colorbar(im, ax=ax, label="TOF (μs)")
        ax.set_xlabel("x (mm)" if ext_mm is None else "Focus index")
        ax.set_ylabel("Element index" if ext_mm is None else "z (mm)")

    ax.set_title(title)

    return fig, ax


def plot_tfm_image(
    result,
    db_range: float = -40.0,
    title: str = "TFM Image",
    figsize: tuple[float, float] = (8, 6),
    cmap: str = "inferno",
    ax: Optional[Axes] = None,
) -> tuple[Figure, Axes]:
    """Plot a TFM reconstructed image in dB scale.

    Parameters
    ----------
    result : TFMResult
        TFM reconstruction result.
    db_range : float
        Dynamic range in dB (negative value, e.g., -40).
    title : str
        Plot title.
    cmap : str
        Colormap name.
    ax : Axes, optional

    Returns
    -------
    fig, ax
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.get_figure()

    ext_mm = (
        result.extent[0] * 1e3,
        result.extent[1] * 1e3,
        result.extent[2] * 1e3,
        result.extent[3] * 1e3,
    )

    image_db = result.image_db.copy()
    image_db[image_db < db_range] = db_range

    im = ax.imshow(
        image_db,
        extent=ext_mm,
        cmap=cmap,
        vmin=db_range,
        vmax=0,
        aspect="auto",
    )
    plt.colorbar(im, ax=ax, label="Amplitude (dB)")

    ax.set_xlabel("x (mm)")
    ax.set_ylabel("z (mm)")
    ax.set_title(title)

    return fig, ax


def plot_bscan(
    fmc_data: np.ndarray,
    element_idx: int,
    fs: float,
    title: Optional[str] = None,
    figsize: tuple[float, float] = (10, 4),
    cmap: str = "seismic",
    ax: Optional[Axes] = None,
) -> tuple[Figure, Axes]:
    """Plot a B-scan from FMC data for a given transmitter element.

    Parameters
    ----------
    fmc_data : np.ndarray, shape (N_tx, N_rx, N_samples)
        FMC data.
    element_idx : int
        Transmitter element index.
    fs : float
        Sampling frequency.
    title : str, optional
    ax : Axes, optional

    Returns
    -------
    fig, ax
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.get_figure()

    bscan = fmc_data[element_idx, :, :]  # (N_rx, N_samples)
    n_rx, n_samples = bscan.shape
    t_axis = np.arange(n_samples) / fs * 1e6  # μs

    vmax = np.max(np.abs(bscan)) * 0.8

    ax.imshow(
        bscan,
        aspect="auto",
        cmap=cmap,
        extent=[t_axis[0], t_axis[-1], n_rx - 0.5, -0.5],
        vmin=-vmax,
        vmax=vmax,
    )

    ax.set_xlabel("Time (μs)")
    ax.set_ylabel("Receiver element")
    ax.set_title(title or f"B-scan (TX element {element_idx})")

    return fig, ax


def plot_ascan(
    signal: np.ndarray,
    fs: float,
    title: str = "A-scan",
    figsize: tuple[float, float] = (10, 3),
    ax: Optional[Axes] = None,
) -> tuple[Figure, Axes]:
    """Plot a single A-scan signal.

    Parameters
    ----------
    signal : np.ndarray
        1D time-domain signal.
    fs : float
        Sampling frequency.
    title : str
    ax : Axes, optional

    Returns
    -------
    fig, ax
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.get_figure()

    t = np.arange(len(signal)) / fs * 1e6
    ax.plot(t, signal, "b-", linewidth=0.5)
    ax.set_xlabel("Time (μs)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    return fig, ax
