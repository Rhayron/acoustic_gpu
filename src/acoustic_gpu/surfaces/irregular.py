"""
Irregular (interpolated) surface geometry.

Models an arbitrary surface from a set of discrete measured points,
using piecewise linear or cubic spline interpolation. The piecewise linear
representation is directly compatible with GPU kernels that operate on
discrete surface arrays.
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Literal

from acoustic_gpu.surfaces.base import SurfaceInterface


class IrregularSurface(SurfaceInterface):
    """Interpolated surface from discrete point cloud.

    Parameters
    ----------
    x_points : np.ndarray
        x-coordinates of surface points, must be sorted ascending.
    z_points : np.ndarray
        z-coordinates of surface points.
    method : str
        Interpolation method: 'linear' or 'cubic'.
        'linear' is recommended for GPU kernels (piecewise linear).
        'cubic' uses scipy CubicSpline for smoother representation.
    """

    def __init__(
        self,
        x_points: np.ndarray,
        z_points: np.ndarray,
        method: Literal["linear", "cubic"] = "linear",
    ) -> None:
        self.x_points = np.asarray(x_points, dtype=np.float64)
        self.z_points = np.asarray(z_points, dtype=np.float64)
        self._method = method

        if len(self.x_points) != len(self.z_points):
            raise ValueError("x_points and z_points must have the same length")
        if len(self.x_points) < 2:
            raise ValueError("At least 2 points are required")

        # Sort ascending
        sort_idx = np.argsort(self.x_points)
        self.x_points = self.x_points[sort_idx]
        self.z_points = self.z_points[sort_idx]

        # Build interpolators
        if method == "cubic":
            from scipy.interpolate import CubicSpline
            self._interp = CubicSpline(self.x_points, self.z_points)
            self._deriv = self._interp.derivative()
        else:
            self._interp = None
            self._deriv = None

    @property
    def method(self) -> str:
        return self._method

    def evaluate(self, x: np.ndarray | float) -> np.ndarray | float:
        """Evaluate surface height via interpolation."""
        x = np.asarray(x, dtype=np.float64)
        if self._method == "cubic" and self._interp is not None:
            return self._interp(x)
        else:
            return np.interp(x, self.x_points, self.z_points)

    def slope(self, x: np.ndarray | float) -> np.ndarray | float:
        """Evaluate surface slope dz/dx."""
        x = np.asarray(x, dtype=np.float64)
        if self._method == "cubic" and self._deriv is not None:
            return self._deriv(x)
        # Finite difference approximation for linear interpolation
        delta = (self.x_points[-1] - self.x_points[0]) / (10 * len(self.x_points))
        z_plus  = np.interp(x + delta, self.x_points, self.z_points)
        z_minus = np.interp(x - delta, self.x_points, self.z_points)
        return (z_plus - z_minus) / (2.0 * delta)

    def get_bounds(self) -> tuple[float, float]:
        return (float(self.x_points[0]), float(self.x_points[-1]))

    @staticmethod
    def from_function(
        func: callable,
        x_min: float,
        x_max: float,
        n_points: int = 500,
        method: Literal["linear", "cubic"] = "linear",
    ) -> "IrregularSurface":
        """Create an IrregularSurface from an analytical function.

        Parameters
        ----------
        func : callable
            Function f(x) -> z.
        x_min, x_max : float
            Domain bounds.
        n_points : int
            Number of sample points.
        method : str
            Interpolation method.

        Returns
        -------
        IrregularSurface
        """
        x = np.linspace(x_min, x_max, n_points)
        z = func(x)
        return IrregularSurface(x, z, method=method)

    @staticmethod
    def sinusoidal(
        amplitude: float = 0.002,
        period: float = 0.02,
        z_offset: float = 0.02,
        x_min: float = -0.05,
        x_max: float = 0.05,
        n_points: int = 500,
    ) -> "IrregularSurface":
        """Create a sinusoidal surface for testing.

        f(x) = z_offset + amplitude·sin(2π·x / period)

        Parameters
        ----------
        amplitude : float
            Sinusoidal amplitude in meters.
        period : float
            Sinusoidal period in meters.
        z_offset : float
            Mean surface depth in meters.
        x_min, x_max : float
            Domain bounds.
        n_points : int
            Number of sample points.

        Returns
        -------
        IrregularSurface
        """
        omega = 2.0 * np.pi / period
        return IrregularSurface.from_function(
            func=lambda x: z_offset + amplitude * np.sin(omega * x),
            x_min=x_min,
            x_max=x_max,
            n_points=n_points,
            method="linear",
        )

    def resample(self, n_points: int) -> "IrregularSurface":
        """Create a new surface with different number of sample points.

        Parameters
        ----------
        n_points : int
            New number of points.

        Returns
        -------
        IrregularSurface
        """
        x_new = np.linspace(self.x_points[0], self.x_points[-1], n_points)
        z_new = self.evaluate(x_new)
        return IrregularSurface(x_new, z_new, method=self.method)

    @property
    def n_points(self) -> int:
        return len(self.x_points)

    @property
    def spacing(self) -> float:
        """Average spacing between points."""
        return (self.x_points[-1] - self.x_points[0]) / (self.n_points - 1)

    def __repr__(self) -> str:
        return (
            f"IrregularSurface(n_points={self.n_points}, "
            f"x=[{self.x_points[0] * 1e3:.1f}, {self.x_points[-1] * 1e3:.1f}]mm, "
            f"method='{self.method}')"
        )
