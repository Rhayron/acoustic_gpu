"""
Flat (planar) surface geometry.

Models a flat interface f(x) = z0 + x*tan(θ), where z0 is the vertical
offset and θ is an optional tilt angle. This is the simplest geometry
and admits an analytical Snell's law solution for validation.
"""

from __future__ import annotations

import numpy as np
from typing import Optional

from acoustic_gpu.surfaces.base import SurfaceInterface


class FlatSurface(SurfaceInterface):
    """Flat planar surface with optional tilt.

    Parameters
    ----------
    z_offset : float
        Vertical offset of the surface in meters (depth below transducer).
    tilt_angle : float
        Tilt angle in radians. Default is 0 (horizontal).
        Positive angle tilts the surface clockwise.
    x_extent : float
        Half-width of the surface definition domain in meters.
        Surface is defined on [-x_extent, +x_extent].
    """

    def __init__(self, z_offset: float, tilt_angle: float = 0.0,
                 x_extent: float = 0.05) -> None:
        self.z_offset = z_offset
        self.tilt_angle = tilt_angle
        self._slope_val = np.tan(tilt_angle)
        self.x_extent = x_extent

    def evaluate(self, x: np.ndarray | float) -> np.ndarray | float:
        """z = z0 + x*tan(θ)"""
        return self.z_offset + np.asarray(x) * self._slope_val

    def slope(self, x: np.ndarray | float) -> np.ndarray | float:
        """dz/dx = tan(θ)"""
        return np.full_like(np.asarray(x, dtype=np.float64), self._slope_val)

    def get_bounds(self) -> tuple[float, float]:
        return (-self.x_extent, self.x_extent)

    def snell_analytical(
        self,
        x_a: float, z_a: float,
        x_f: float, z_f: float,
        c1: float, c2: float,
    ) -> tuple[float, float, float]:
        """Analytical Snell's law solution for a flat horizontal surface.

        Only valid when tilt_angle == 0. Uses Newton-Raphson on the
        Snell condition directly for fast high-precision reference.

        Parameters
        ----------
        x_a, z_a : float
            Emitter position (in coupling medium).
        x_f, z_f : float
            Focus position (in specimen).
        c1, c2 : float
            Wave speeds in medium 1 and medium 2.

        Returns
        -------
        x_s : float
            x coordinate of the refraction point.
        z_s : float
            z coordinate of the refraction point (= z_offset).
        tof : float
            Total time of flight in seconds.

        Raises
        ------
        ValueError
            If total internal reflection occurs (no solution).
        """
        z_s = self.z_offset
        h1 = abs(z_s - z_a)  # Height in medium 1
        h2 = abs(z_f - z_s)  # Height in medium 2

        # Newton-Raphson to solve Snell's law: sin(θ1)/c1 = sin(θ2)/c2
        # Let xs be the x-coordinate of the refraction point.
        # f(xs) = (xs - xa)/[c1*d1(xs)] - (xf - xs)/[c2*d2(xs)] = 0
        # where d1 = sqrt((xs-xa)^2 + h1^2), d2 = sqrt((xf-xs)^2 + h2^2)

        xs = (x_a + x_f) / 2.0  # Initial guess: midpoint

        # Bounds for clamping
        x_lo = min(x_a, x_f) - abs(x_f - x_a)
        x_hi = max(x_a, x_f) + abs(x_f - x_a)

        for _ in range(100):
            dx1 = xs - x_a
            dx2 = x_f - xs
            d1 = np.sqrt(dx1 ** 2 + h1 ** 2)
            d2 = np.sqrt(dx2 ** 2 + h2 ** 2)

            # Snell residual
            f_val = dx1 / (c1 * d1) - dx2 / (c2 * d2)

            # Derivative of f w.r.t. xs
            df_val = (h1 ** 2) / (c1 * d1 ** 3) + (h2 ** 2) / (c2 * d2 ** 3)

            if abs(df_val) < 1e-30:
                break

            xs_new = xs - f_val / df_val

            # Clamp to bounds
            xs_new = max(x_lo, min(x_hi, xs_new))

            if abs(xs_new - xs) < 1e-12:
                break
            xs = xs_new

        # Verify no total internal reflection
        d1 = np.sqrt((xs - x_a) ** 2 + h1 ** 2)
        sin_theta1 = abs(xs - x_a) / d1
        if sin_theta1 >= c1 / c2 and c1 < c2:
            raise ValueError(
                f"Total internal reflection: sin(θ1)={sin_theta1:.4f} ≥ c1/c2={c1/c2:.4f}"
            )

        d2 = np.sqrt((x_f - xs) ** 2 + h2 ** 2)
        tof = d1 / c1 + d2 / c2

        return xs, z_s, tof

    def __repr__(self) -> str:
        return (
            f"FlatSurface(z_offset={self.z_offset:.4f}, "
            f"tilt_angle={np.degrees(self.tilt_angle):.2f}°)"
        )
