"""
Surface estimation from FMC (Full Matrix Capture) data.

Provides algorithms for estimating the specimen surface geometry
from ultrasonic signal data, including:
  - Peak detection of surface echoes
  - Parametric fitting (circle for tubular surfaces)
  - Threshold-based surface extraction
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Literal

from acoustic_gpu.config import Transducer, Material
from acoustic_gpu.surfaces.flat import FlatSurface
from acoustic_gpu.surfaces.tubular import TubularSurface
from acoustic_gpu.surfaces.irregular import IrregularSurface
from acoustic_gpu.surfaces.base import SurfaceInterface


def detect_surface_echo(
    ascan: np.ndarray,
    fs: float,
    t_min: float = 0.0,
    threshold: float = 0.3,
    method: Literal["peak", "threshold", "first_arrival"] = "threshold",
) -> float:
    """Detect surface echo time in a single A-scan signal.

    Parameters
    ----------
    ascan : np.ndarray
        Time-domain signal (1D array).
    fs : float
        Sampling frequency in Hz.
    t_min : float
        Minimum expected time of arrival (skip initial pulse).
    threshold : float
        Detection threshold as fraction of maximum amplitude.
    method : str
        Detection method: 'peak', 'threshold', or 'first_arrival'.

    Returns
    -------
    t_echo : float
        Time of the surface echo in seconds.
    """
    # Skip samples before t_min
    i_start = int(t_min * fs)
    signal = np.abs(ascan[i_start:])

    if len(signal) == 0:
        return 0.0

    if method == "peak":
        # Find the highest peak
        i_peak = np.argmax(signal)
        return (i_start + i_peak) / fs

    elif method == "threshold":
        # Find first sample exceeding threshold
        max_vol = np.max(signal)
        if max_vol < 1e-30:
            return 0.0
        above = np.where(signal > threshold * max_vol)[0]
        if len(above) == 0:
            return 0.0
        return (i_start + above[0]) / fs

    elif method == "first_arrival":
        # Find first sample above noise level (mean + 3σ)
        noise_level = np.mean(signal[:max(10, len(signal) // 10)])
        noise_std = np.std(signal[:max(10, len(signal) // 10)])
        level = noise_level + 3.0 * noise_std + 1e-30
        above = np.where(signal > level)[0]
        if len(above) == 0:
            return 0.0
        return (i_start + above[0]) / fs

    return 0.0


def estimate_surface_from_fmc(
    fmc_data: np.ndarray,
    transducer: Transducer,
    c_water: float,
    fs: float,
    t_min: float = 0.0,
    threshold: float = 0.3,
    method: Literal["peak", "threshold", "first_arrival"] = "threshold",
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate surface point cloud from FMC data.

    Uses the pulse-echo (diagonal) signals from the FMC matrix to
    detect surface echoes and convert them to spatial coordinates.

    Parameters
    ----------
    fmc_data : np.ndarray, shape (N_tx, N_rx, N_samples)
        Full Matrix Capture data.
    transducer : Transducer
        Transducer configuration.
    c_water : float
        Speed of sound in the coupling medium (m/s).
    fs : float
        Sampling frequency (Hz).
    t_min : float
        Minimum expected echo time (skip initial artifact).
    threshold : float
        Detection threshold.
    method : str
        Echo detection method.

    Returns
    -------
    x_surface : np.ndarray, shape (N_elements,)
        x-coordinates of detected surface points.
    z_surface : np.ndarray, shape (N_elements,)
        z-coordinates (depth) of detected surface points.
    """
    n_elements = transducer.n_elements
    elem_pos = transducer.element_positions()

    x_surface = elem_pos[:, 0].copy()
    z_surface = np.zeros(n_elements)

    for i in range(n_elements):
        # Use pulse-echo (diagonal) signal: tx == rx
        ascan = fmc_data[i, i, :]
        t_echo = detect_surface_echo(ascan, fs, t_min, threshold, method)

        # Convert round-trip time to one-way distance
        distance = t_echo * c_water / 2.0
        z_surface[i] = elem_pos[i, 1] + distance

    return x_surface, z_surface


def fit_surface(
    x_points: np.ndarray,
    z_points: np.ndarray,
    geometry: Literal["flat", "tubular", "irregular"] = "irregular",
    **kwargs,
) -> SurfaceInterface:
    """Fit a parametric surface model to detected points.

    Parameters
    ----------
    x_points, z_points : np.ndarray
        Detected surface point cloud.
    geometry : str
        Geometry type to fit: 'flat', 'tubular', or 'irregular'.
    **kwargs :
        Extra arguments passed to the surface constructor or fit method.

    Returns
    -------
    SurfaceInterface
        Fitted surface object.
    """
    if geometry == "flat":
        # Fit a linear model: z = a*x + b
        coeffs = np.polyfit(x_points, z_points, deg=1)
        slope, intercept = coeffs
        tilt_angle = np.arctan(slope)
        return FlatSurface(
            z_offset=intercept,
            tilt_angle=tilt_angle,
            x_extent=max(abs(x_points.max()), abs(x_points.min())) * 1.2,
        )

    elif geometry == "tubular":
        from scipy.optimize import least_squares

        # Fit a circle: (x - xc)^2 + (z - zc)^2 = r^2
        def residuals(params: np.ndarray) -> np.ndarray:
            xc, zc, r = params
            return np.sqrt((x_points - xc) ** 2 + (z_points - zc) ** 2) - r

        xc0 = np.mean(x_points)
        zc0 = np.max(z_points) - 0.01
        r0 = (np.max(z_points) - np.min(z_points)) * 2.0 + 0.01

        result = least_squares(residuals, [xc0, zc0, r0])
        xc, zc, r = result.x

        outer = kwargs.get("outer", True)
        return TubularSurface(
            radius=abs(r),
            center_x=xc,
            center_z=zc,
            outer=outer,
        )

    else:
        # Irregular — direct interpolation
        method = kwargs.get("method", "linear")
        return IrregularSurface(x_points, z_points, method=method)
