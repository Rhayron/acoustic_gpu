"""
Tubular (cylindrical) surface geometry.

Models a cylindrical interface — either the outer or inner surface of a
tube/pipe. The surface is described as a circular arc:

    f(x) = z_c ± sqrt(R² - (x - x_c)²)

where the sign depends on whether we're looking at the top (convex
towards transducer) or bottom (concave) of the cylinder.
"""

from __future__ import annotations

import numpy as np
from typing import Optional

from acoustic_gpu.surfaces.base import SurfaceInterface


class TubularSurface(SurfaceInterface):
    """Cylindrical (tubular) surface.

    Parameters
    ----------
    radius : float
        Cylinder radius in meters.
    center_x : float
        x-coordinate of the cylinder center in meters.
    center_z : float
        z-coordinate of the cylinder center in meters.
    outer : bool
        If True, models the top surface (convex toward transducer):
            f(x) = z_c - sqrt(R² - (x - x_c)²)
        If False, models the inner (concave) surface:
            f(x) = z_c + sqrt(R² - (x - x_c)²)
    angular_extent : float
        Angular extent of the arc in radians (centered at top/bottom).
        Default is π/2 (quarter-circle on each side of the apex).
    """

    def __init__(
        self,
        radius: float,
        center_x: float = 0.0,
        center_z: float = 0.05,
        outer: bool = True,
        angular_extent: float = np.pi / 2,
    ) -> None:
        self.radius = radius
        self.center_x = center_x
        self.center_z = center_z
        self.outer = outer
        self.angular_extent = angular_extent
        self._sign = -1.0 if outer else 1.0

        # Compute x-bounds from angular extent
        self._x_half = radius * np.sin(min(angular_extent, np.pi))

    def evaluate(self, x: np.ndarray | float) -> np.ndarray | float:
        """Evaluate surface z = z_c ± sqrt(R² - (x - x_c)²)."""
        x = np.asarray(x, dtype=np.float64)
        dx = x - self.center_x
        r2 = self.radius ** 2
        dx2 = dx ** 2

        # Clamp to avoid sqrt of negative (outside the arc)
        arg = np.maximum(r2 - dx2, 0.0)
        return self.center_z + self._sign * np.sqrt(arg)

    def slope(self, x: np.ndarray | float) -> np.ndarray | float:
        """Compute dz/dx = ∓(x - x_c) / sqrt(R² - (x - x_c)²)."""
        x = np.asarray(x, dtype=np.float64)
        dx = x - self.center_x
        r2 = self.radius ** 2
        dx2 = dx ** 2

        arg = np.maximum(r2 - dx2, 1e-30)  # Avoid division by zero
        return self._sign * dx / np.sqrt(arg)

    def get_bounds(self) -> tuple[float, float]:
        return (self.center_x - self._x_half, self.center_x + self._x_half)

    def apex(self) -> tuple[float, float]:
        """Return the apex point (top/bottom) of the cylinder surface.

        Returns
        -------
        x_apex, z_apex : float
        """
        z_apex = self.center_z + self._sign * self.radius
        return self.center_x, z_apex

    def fit_from_tof(
        self,
        x_elements: np.ndarray,
        tof_surface: np.ndarray,
        c_water: float,
        z_elements: float = 0.0,
    ) -> "TubularSurface":
        """Fit a cylindrical surface from measured surface-echo TOF values.

        Uses least squares circle fitting to estimate radius and center
        from the surface points derived from TOF measurements.

        Parameters
        ----------
        x_elements : np.ndarray
            x positions of transducer elements.
        tof_surface : np.ndarray
            Time-of-flight to the surface echo for each element (round-trip).
        c_water : float
            Speed of sound in the coupling medium.
        z_elements : float
            z-position of the transducer elements.

        Returns
        -------
        TubularSurface
            Fitted cylindrical surface.
        """
        from scipy.optimize import least_squares

        # Convert TOF to one-way distance
        distances = tof_surface * c_water / 2.0
        z_surface = z_elements + distances
        x_surface = x_elements

        # Least-squares circle fit: minimize ||(xi - xc)² + (zi - zc)² - R²||
        def residuals(params: np.ndarray) -> np.ndarray:
            xc, zc, r = params
            return np.sqrt((x_surface - xc) ** 2 + (z_surface - zc) ** 2) - r

        # Initial guess
        xc0 = np.mean(x_surface)
        zc0 = np.mean(z_surface) + 0.01  # Guess center is below surface
        r0 = np.std(z_surface) * 2 + 0.01

        result = least_squares(residuals, [xc0, zc0, r0])
        xc, zc, r = result.x

        return TubularSurface(
            radius=abs(r),
            center_x=xc,
            center_z=zc,
            outer=self.outer,
            angular_extent=self.angular_extent,
        )

    def __repr__(self) -> str:
        kind = "outer" if self.outer else "inner"
        return (
            f"TubularSurface(R={self.radius * 1e3:.1f}mm, "
            f"center=({self.center_x * 1e3:.1f}, {self.center_z * 1e3:.1f})mm, "
            f"{kind})"
        )
