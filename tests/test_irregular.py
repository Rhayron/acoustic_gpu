"""
Tests for irregular (interpolated) surface geometry and ray-tracing.

Uses a sinusoidal surface as test case and validates GPU vs CPU
reference implementations.
"""

import numpy as np
import pytest

from acoustic_gpu.surfaces.irregular import IrregularSurface
from acoustic_gpu.raytracing.cpu_ref import (
    find_refraction_point,
    find_refraction_points_cpu,
)


class TestIrregularSurface:
    """Test IrregularSurface geometry."""

    def test_linear_interpolation(self):
        """Linear interpolation should pass through given points."""
        x = np.array([0.0, 0.01, 0.02, 0.03, 0.04])
        z = np.array([0.020, 0.025, 0.019, 0.022, 0.020])
        surface = IrregularSurface(x, z, method="linear")

        z_eval = surface.evaluate(x)
        np.testing.assert_allclose(z_eval, z, atol=1e-15)

    def test_midpoint_linear(self):
        """Linear interpolation at midpoints should be average."""
        x = np.array([0.0, 0.01, 0.02])
        z = np.array([0.020, 0.030, 0.019])
        surface = IrregularSurface(x, z, method="linear")

        z_mid     = surface.evaluate(0.005)
        expected  = (0.020 + 0.030) / 2.0
        np.testing.assert_allclose(z_mid, expected, atol=1e-10)

    def test_bounds(self):
        x = np.array([0.01, 0.02, 0.03, 0.04])
        z = np.array([0.021, 0.024, 0.019, 0.020])
        surface = IrregularSurface(x, z)
        assert surface.get_bounds() == (0.01, 0.04)

    def test_sinusoidal_factory(self):
        """Test sinusoidal surface factory method."""
        surface = IrregularSurface.sinusoidal(
            amplitude=0.002,
            period=0.02,
            z_offset=0.020,
            x_min=-0.05,
            n_points=100,
        )

        assert surface.n_points == 100

        # At x=0, sin(0)=0, so z should equal z_offset
        z_center = surface.evaluate(0.0)
        np.testing.assert_allclose(z_center, 0.020, atol=1e-4)

    def test_from_function(self):
        """Test creation from arbitrary function."""
        surface = IrregularSurface.from_function(
            func=lambda x: 0.020 + 0.001 * x,
            x_min=-0.05,
            x_max=0.05,
            n_points=200,
        )

        assert surface.n_points  == 200
        z_center = surface.evaluate(0.0)
        np.testing.assert_allclose(z_center, 0.020, atol=1e-5)

    def test_resample(self):
        """Resampling should produce a new surface with new point count."""
        surface  = IrregularSurface.sinusoidal(n_points=100)
        resampled = surface.resample(200)
        assert resampled.n_points == 200

    def test_sorting(self):
        """Surface should sort unsorted inputs."""
        x = np.array([0.03, 0.01, 0.02])
        z = np.array([0.020, 0.025, 0.021])
        surface = IrregularSurface(x, z)
        np.testing.assert_array_equal(surface.x_points, [0.01, 0.02, 0.03])


class TestIrregularRaytracing:
    """Test ray-tracing on sinusoidal irregular surface."""

    def setup_method(self):
        self.surface = IrregularSurface.sinusoidal(
            amplitude=0.002,
            period=0.02,
            z_offset=0.020,
            x_min=-0.05,
            n_points=500,
        )
        self.c1 = 1480.0
        self.c2 = 5900.0
        self.xS = self.surface.x_points
        self.zS = self.surface.z_points

    def test_convergence(self):
        """Newton-Raphson should converge on smooth sinusoidal surface."""
        x_a, z_a = 0.0, 0.0
        x_f, z_f = 0.005, 0.035

        result = find_refraction_point(
            x_a, z_a, x_f, z_f, self.xS, self.zS, self.c1, self.c2
        )

        assert result.converged
        assert result.tof > 0

    def test_refraction_on_surface(self):
        """Refraction point should lie on the surface."""
        x_a, z_a = -0.005, 0.0
        x_f, z_f =  0.005, 0.035

        result = find_refraction_point(
            x_a, z_a, x_f, z_f, self.xS, self.zS, self.c1, self.c2
        )

        # Check that (x_s, z_s) is on the surface
        z_expected = self.surface.evaluate(result.x_s)
        np.testing.assert_allclose(result.z_s, z_expected, atol=1e-4)

    def test_tof_positive(self):
        """TOF should always be positive."""
        emitters = np.array([[-0.010, 0.0], [0.0, 0.0], [0.010, 0.0]])
        focuses  = np.array([[-0.010, 0.030], [0.0, 0.030], [0.005, 0.030]])

        _, tof = find_refraction_points_cpu(
            emitters, focuses, self.xS, self.zS, self.c1, self.c2
        )

        assert np.all(tof > 0)

    def test_tof_continuity(self):
        """TOF should vary smoothly across adjacent focus points."""
        x_a, z_a  = 0.0, 0.0
        focuses_x = np.linspace(-0.010, 0.010, 50)
        focuses   = np.column_stack([focuses_x, np.full(50, 0.035)])
        emitters  = np.array([[x_a, z_a]])

        _, tof = find_refraction_points_cpu(
            emitters, focuses, self.xS, self.zS, self.c1, self.c2
        )

        # TOF gradient should be smooth (no large jumps)
        tof_flat = tof[0, :]
        dtof     = np.diff(tof_flat)
        max_jump = np.max(np.abs(np.diff(dtof)))

        # The second derivative of TOF should be small
        assert max_jump < 1e-7, f"TOF discontinuity detected: max_jump={max_jump}"
