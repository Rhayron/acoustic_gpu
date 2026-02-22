"""
Tests for flat surface geometry and ray-tracing.

Validates the Newton-Raphson refraction algorithm against the
analytical Snell's law solution for flat interfaces.
"""

import numpy as np
import pytest

from acoustic_gpu.config import Transducer, WATER, STEEL_3020
from acoustic_gpu.surfaces.flat import FlatSurface
from acoustic_gpu.raytracing.cpu_ref import (
    find_refraction_point,
    find_refraction_points_cpu,
)


class TestFlatSurface:
    """Test FlatSurface geometry."""

    def test_evaluate_horizontal(self):
        surface = FlatSurface(z_offset=0.020)
        x = np.linspace(-0.05, 0.05, 100)
        z  = surface.evaluate(x)
        np.testing.assert_allclose(z, 0.020, atol=1e-15)

    def test_evaluate_tilted(self):
        angle = np.radians(10)
        surface = FlatSurface(z_offset=0.020, tilt_angle=angle)
        x = np.array([0.0, 0.01, -0.01])
        z  = surface.evaluate(x)
        expected = 0.020 + x * np.tan(angle)
        np.testing.assert_allclose(z, expected, atol=1e-15)

    def test_slope_horizontal(self):
        surface = FlatSurface(z_offset=0.020)
        x = np.array([0.0, 0.01, 0.05])
        m = surface.slope(x)
        np.testing.assert_allclose(m, 0.0, atol=1e-15)

    def test_slope_tilted(self):
        angle = np.radians(10)
        surface = FlatSurface(z_offset=0.020, tilt_angle=angle)
        x = np.array([0.0, 0.01])
        m = surface.slope(x)
        np.testing.assert_allclose(m, np.tan(angle), atol=1e-15)

    def test_get_points(self):
        surface  = FlatSurface(z_offset=0.020)
        xS, zS = surface.get_points(100)
        assert len(xS) == 100
        assert len(zS) == 100
        np.testing.assert_allclose(zS, 0.020, atol=1e-15)

    def test_bounds(self):
        surface = FlatSurface(z_offset=0.020, x_extent=0.03)
        bounds  = surface.get_bounds()
        assert bounds == (-0.03, 0.03)


class TestFlatAnalytical:
    """Validates Newton-Raphson vs analytical Snell solution on flat surface."""

    def setup_method(self):
        self.surface = FlatSurface(z_offset=0.020)
        self.c1 = 1480.0  # water
        self.c2 = 5900.0  # steel
        self.xS, self.zS = self.surface.get_points(500)

    def test_normal_incidence(self):
        """Ray directly below element — refraction point == element x."""
        x_a, z_a = 0.0, 0.0
        x_f, z_f = 0.0, 0.030

        # Analytical
        xs_ana, _, tof_ana = self.surface.snell_analytical(
            x_a, z_a, x_f, z_f, self.c1, self.c2
        )

        # Newton-Raphson
        result = find_refraction_point(
            x_a, z_a, x_f, z_f, self.xS, self.zS, self.c1, self.c2
        )

        assert result.converged
        np.testing.assert_allclose(result.x_s, xs_ana, atol=1e-5)
        np.testing.assert_allclose(result.tof, tof_ana, rtol=1e-6)

    def test_oblique_incidence(self):
        """Oblique ray — general case."""
        x_a, z_a = -0.008, 0.0
        x_f, z_f = 0.005, 0.030

        xs_ana, _, tof_ana = self.surface.snell_analytical(
            x_a, z_a, x_f, z_f, self.c1, self.c2
        )

        result = find_refraction_point(
            x_a, z_a, x_f, z_f, self.xS, self.zS, self.c1, self.c2
        )

        assert result.converged
        np.testing.assert_allclose(result.tof, tof_ana, rtol=1e-6)

    def test_high_angle(self):
        """High incidence angle near critical angle."""
        x_a, z_a = -0.005, 0.0
        x_f, z_f =  0.003, 0.030

        xs_ana, _, tof_ana = self.surface.snell_analytical(
            x_a, z_a, x_f, z_f, self.c1, self.c2
        )

        result = find_refraction_point(
            x_a, z_a, x_f, z_f, self.xS, self.zS, self.c1, self.c2
        )

        assert result.converged
        np.testing.assert_allclose(result.tof, tof_ana, rtol=1e-5)

    def test_symmetric_rays(self):
        """TOF should be symmetric for mirrored geometries."""
        x_a, z_a =  0.010, 0.0
        x_f, z_f =  0.010, 0.030

        result1 = find_refraction_point(
            x_a, z_a, x_f, z_f, self.xS, self.zS, self.c1, self.c2
        )
        result2 = find_refraction_point(
            -x_a, z_a, -x_f, z_f, self.xS, self.zS, self.c1, self.c2
        )

        np.testing.assert_allclose(result1.tof, result2.tof, rtol=1e-6)

    def test_batch_cpu(self):
        """Test batch CPU computation against analytical solutions."""
        transducer = Transducer(n_elements=16, pitch=0.6e-3, frequency=5e6)
        emitters   = transducer.element_positions()

        # Single focus below center
        focuses = np.array([[0.0, 0.030]])

        k_result, tof_result = find_refraction_points_cpu(
            emitters, focuses, self.xS, self.zS, self.c1, self.c2
        )

        assert k_result.shape   == (16, 1)
        assert tof_result.shape == (16, 1)

        # Compare each element with analytical
        for i in range(16):
            xs_ana, _, tof_ana = self.surface.snell_analytical(
                emitters[i, 0], emitters[i, 1],
                0.0, 0.030,
                self.c1, self.c2,
            )
            np.testing.assert_allclose(
                tof_result[i, 0], tof_ana, rtol=1e-5,
                err_msg=f"Element {i} TOF mismatch",
            )
