"""
Ray-tracing algorithms for refracted acoustic paths.

Provides GPU-accelerated (Taichi) and CPU reference implementations
of Newton-Raphson based refraction point search and TOF calculation.
"""

from acoustic_gpu.raytracing.kernels import find_refraction_points_gpu, compute_tof_table
from acoustic_gpu.raytracing.cpu_ref import find_refraction_points_cpu

__all__ = [
    "find_refraction_points_gpu",
    "compute_tof_table",
    "find_refraction_points_cpu",
]
