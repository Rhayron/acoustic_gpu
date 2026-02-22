"""
Abstract base class for surface geometry representations.

All surface classes must implement the SurfaceInterface protocol,
providing methods for evaluating the surface height, slope, and
discretization into point arrays suitable for GPU ray tracing kernels.
"""

from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional


class SurfaceInterface(ABC):
    """Abstract interface for surface geometries.

    A surface is a 2D curve f(x) = z describing the interface between
    the coupling medium (e.g., water) and the test specimen. The surface
    is assumed to be single-valued in x (no overhangs).
    """

    @abstractmethod
    def evaluate(self, x: np.ndarray | float) -> np.ndarray | float:
        """Evaluate surface height z = f(x).

        Parameters
        ----------
        x : array-like or float
            Horizontal coordinate(s) in meters.

        Returns
        -------
        z : array-like or float
            Surface height(s) at the given x coordinate(s).

        ...
        """

    @abstractmethod
    def slope(self, x: np.ndarray | float) -> np.ndarray | float:
        """Evaluate surface slope dz/dx at position x.

        Parameters
        ----------
        x : array-like or float
            Horizontal coordinate(s) in meters.

        Returns
        -------
        dzdx : array-like or float
            Surface slope(s) at the given x coordinate(s).

        ...
        """

    @abstractmethod
    def get_bounds(self) -> tuple[float, float]:
        """Get the valid x-range for this surface.

        Returns
        -------
        x_min, x_max : float
            Horizontal limits where the surface is defined.

        ...
        """

    def get_points(self, n: int, x_min: Optional[float] = None,
                   x_max: Optional[float] = None) -> tuple[np.ndarray, np.ndarray]:
        """Discretize the surface into n evenly-spaced points.

        Parameters
        ----------
        n : int
            Number of points.
        x_min, x_max : float, optional
            Override the default bounds.

        Returns
        -------
        xS, zS : np.ndarray
            Arrays of shape (n,) with x and z coordinates of surface points.
        """
        if x_min is None or x_max is None:
            bounds = self.get_bounds()
            x_min = x_min if x_min is not None else np.clip(bounds[0], None, bounds[0])
            x_max = x_max if x_max is not None else np.clip(bounds[1], None, bounds[1])

        xS = np.linspace(x_min, x_max, n, dtype=np.float64)
        zS = np.asarray(self.evaluate(xS), dtype=np.float64)
        return xS, zS

    def normal(self, x: float) -> tuple[np.ndarray, np.ndarray]:
        """Compute the outward unit normal at position x.

        Assumes the coupling medium is above (smaller z) and the specimen below.

        Parameters
        ----------
        x : array-like or float

        Returns
        -------
        nx, nz : array-like or float
            Components of the unit normal vector (pointing into coupling medium).
        """
        m = self.slope(x)
        # normal = (-dz/dx, 1) / |(-dz/dx, 1)| — pointing upward (into water)
        norm = np.sqrt(1.0 + m ** 2)
        nx = -m / norm
        nz = 1.0 / norm
        return nx, nz
