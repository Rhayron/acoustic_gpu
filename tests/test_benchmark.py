"""
Benchmark tests: GPU vs CPU performance comparison.

Measures execution time for ray-tracing on different grid sizes
and reports speedup factors.
"""

import numpy as np
import pytest
import time

from acoustic_gpu.config import Transducer
from acoustic_gpu.surfaces.flat import FlatSurface
from acoustic_gpu.raytracing.cpu_ref import find_refraction_points_cpu


class TestBenchmark:
    """Performance benchmarks for ray-tracing."""

    def setup_method(self):
        self.c1 = 1480.0
        self.c2 = 5900.0
        self.surface = FlatSurface(z_offset=0.020)

    def _run_cpu_benchmark(self, n_elements: int, n_focus: int) -> float:
        """Run CPU benchmark and return elapsed time in seconds."""
        transducer = Transducer(n_elements=n_elements, pitch=0.6e-3, frequency=5e6)
        emitters = transducer.element_positions()

        # Create focus grid
        x_f = np.linspace(-0.015, 0.015, int(np.sqrt(n_focus)))
        z_f = np.linspace(0.025, 0.060, int(np.sqrt(n_focus)))
        X, Z = np.meshgrid(x_f, z_f)
        focuses = np.column_stack([X.ravel(), Z.ravel()])

        xS, zS = self.surface.get_points(500)

        start = time.perf_counter()
        _, tof = find_refraction_points_cpu(
            emitters, focuses, xS, zS, self.c1, self.c2
        )
        elapsed = time.perf_counter() - start

        assert np.all(tof > 0)
        return elapsed

    def test_cpu_small(self):
        """Benchmark: 16 elements × 64 focus points."""
        elapsed = self._run_cpu_benchmark(16, 64)
        print(f"\nCPU 16×64: {elapsed:.3f}s ({16 * 64} rays)")
        assert elapsed < 10  # Should complete in reasonable time

    def test_cpu_medium(self):
        """Benchmark: 32 elements × 256 focus points."""
        elapsed = self._run_cpu_benchmark(32, 256)
        print(f"\nCPU 32×256: {elapsed:.3f}s ({(32 * 256)} rays)")
        assert elapsed < 120

    @pytest.mark.slow
    def test_cpu_large(self):
        """Benchmark: 64 elements × 1024 focus points."""
        elapsed = self._run_cpu_benchmark(64, 1024)
        print(f"\nCPU 64×1024: {elapsed:.3f}s ({64 * 1024} rays)")

    def test_gpu_available(self):
        """Check if GPU ray-tracing kernel is initialized."""
        try:
            from acoustic_gpu.raytracing.kernels import ensure_initialized
            ensure_initialized("cpu")  # Use CPU backend for CI
            assert True
        except ImportError:
            pytest.skip("Taichi not available")

    def test_gpu_vs_cpu_small(self):
        """Compare GPU vs CPU for a small problem."""
        try:
            from acoustic_gpu.raytracing.kernels import find_refraction_points_gpu
        except ImportError:
            pytest.skip("Taichi not available")

        n_elements = 16
        n_focus_side = 8  # 8 × 8 = 64 total
        emitters = Transducer(n_elements=n_elements, pitch=0.6e-3, frequency=5e6) \
            .element_positions()

        x_f = np.linspace(-0.015, 0.015, n_focus_side)
        z_f = np.linspace(0.025, 0.060, n_focus_side)
        X, Z = np.meshgrid(x_f, z_f)
        focuses = np.column_stack([X.ravel(), Z.ravel()])

        xS, zS = self.surface.get_points(500)

        # CPU
        start = time.perf_counter()
        _, tof_cpu = find_refraction_points_cpu(
            emitters, focuses, xS, zS, self.c1, self.c2
        )
        t_cpu = time.perf_counter() - start

        # GPU (with CPU backend for portability)
        start = time.perf_counter()
        _, tof_gpu = find_refraction_points_gpu(
            emitters, focuses, xS, zS, self.c1, self.c2, arch="cpu"
        )
        t_gpu = time.perf_counter() - start

        # Verify correctness
        np.testing.assert_allclose(tof_gpu, tof_cpu, rtol=1e-4)

        print(f"\nCPU: {t_cpu:.3f}s, GPU(cpu-backend): {t_gpu:.3f}s")
