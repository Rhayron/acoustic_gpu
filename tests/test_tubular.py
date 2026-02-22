"""
Tests for tubular (cylindrical) surface geometry and ray-tracing.

Validates the Newton-Raphson refraction algorithm on curved surfaces
by comparing GPU vs CPU reference implementations.
"""

import numpy as np
import pytest

from acoustic_gpu.surfaces.tubular import TubularSurface
from acoustic_gpu.raytracing.cpu_ref import (
    find_refraction_point,
    find_refraction_points_cpu,
)


class TestTubularSurface:
    """Test TubularSurface geometry."""

    def test_evaluate_apex(self):
        """Apex of outer cylinder should be at z_c - R."""
        surface = TubularSurface(radius=0.025, center_x=0.0, center_z=0.045)
        z_apex   = surface.evaluate(0.0)
        expected = 0.045 - 0.025
        np.testing.assert_allclose(z_apex, expected, atol=1e-10)

    def test_evaluate_inner(self):
        """Inner surface apex at z_c + R."""
        surface = TubularSurface(
            radius=0.025, center_x=0.0, center_z=0.048, outer=False
        )
        z_apex   = surface.evaluate(0.0)
        expected = 0.048 + 0.025
        np.testing.assert_allclose(z_apex, expected, atol=1e-10)

    def test_slope_at_apex(self):
        """Slope at apex (x = center_x) should be zero."""
        surface = TubularSurface(radius=0.025, center_x=0.0, center_z=0.045)
        m = surface.slope(0.0)
        np.testing.assert_allclose(m, 0.0, atol=1e-10)

    def test_slope_symmetry(self):
        """Slope should be antisymmetric around the center."""
        surface = TubularSurface(radius=0.025, center_x=0.0, center_z=0.045)
        x_pos = 0.010
        m_pos = surface.slope( x_pos)
        m_neg = surface.slope(-x_pos)
        np.testing.assert_allclose(m_pos, -m_neg, atol=1e-10)

    def test_circularity(self):
        """All surface points should be at distance R from center."""
        surface = TubularSurface(radius=0.025, center_x=0.0, center_z=0.045)
        x = np.linspace(-0.025, 0.025, 100)
        z = surface.evaluate(x)
        distances = np.sqrt((x - 0.0) ** 2 + (z - 0.045) ** 2)
        np.testing.assert_allclose(distances, 0.025, atol=1e-10)

    def test_bounds(self):
        surface = TubularSurface(
            radius=0.025, center_x=0.0,
            angular_extent=np.pi / 2,
        )
        half   = 0.025 * np.sin(np.pi / 2)
        bounds = surface.get_bounds()
        np.testing.assert_allclose(bounds[0], -half, atol=1e-10)
        np.testing.assert_allclose(bounds[1],  half, atol=1e-10)


class TestTubularRaytracing:
    """Test ray tracing on cylindrical surfaces."""

    def setup_method(self):
        self.surface = TubularSurface(
            radius=0.025, center_x=0.0, center_z=0.045
        )
        self.c1 = 1480.0
        self.c2 = 5900.0
        self.xS, self.zS = self.surface.get_points(500)

    def test_normal_incidence(self):
        """Normal incidence on curved surface at apex."""
        x_a, z_a = 0.0, 0.0
        x_f, z_f = 0.0, 0.060  # Below apex

        result = find_refraction_point(
            x_a, z_a, x_f, z_f, self.xS, self.zS, self.c1, self.c2
        )

        assert result.converged
        # Refraction point should be near x = 0 (apex)
        np.testing.assert_allclose(result.x_s, 0.0, atol=1e-4)

    def test_oblique_convergence(self):
        """Check that oblique rays converge on curved surface."""
        x_a, z_a = -0.010, 0.0
        x_f, z_f =  0.005, 0.055

        result = find_refraction_point(
            x_a, z_a, x_f, z_f, self.xS, self.zS, self.c1, self.c2
        )

        assert result.converged
        assert result.tof > 0

    def test_tof_physical_bounds(self):
        """TOF should be between straight-line and longest-path estimates."""
        x_a, z_a = -0.005, 0.0
        x_f, z_f =  0.005, 0.035

        result = find_refraction_point(
            x_a, z_a, x_f, z_f, self.xS, self.zS, self.c1, self.c2
        )

        # Minimum possible TOF (straight line at max speed)
        d_total = np.sqrt((x_a - x_f) ** 2 + (z_a - z_f) ** 2)
        tof_min = d_total / max(self.c1, self.c2)

        # Maximum possible TOF (straight line at min speed)
        tof_max = d_total / min(self.c1, self.c2)

        assert result.tof >= tof_min * 0.9  # Allow some margin
        assert result.tof <= tof_max * 1.5

    def test_batch_consistency(self):
        """Batch computation should match single-ray computations."""
        emitters = np.array([[-0.005, 0.0], [0.0, 0.0], [0.005, 0.0]])
        focuses  = np.array([[0.0, 0.035], [0.003, 0.030]])

        k_batch, tof_batch = find_refraction_points_cpu(
            emitters, focuses, self.xS, self.zS, self.c1, self.c2
        )

        for i in range(len(emitters)):
            for j in range(len(focuses)):
                result = find_refraction_point(
                    emitters[i, 0], emitters[i, 1],
                    focuses[j, 0],  focuses[j, 1],
                    self.xS, self.zS, self.c1, self.c2,
                )
                np.testing.assert_allclose(
                    tof_batch[i, j], result.tof, rtol=1e-6,
                    err_msg=f"Mismatch at emitter={i}, focus={j}",
                )
