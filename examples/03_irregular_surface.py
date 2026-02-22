"""
Example 03: Irregular (Sinusoidal) Surface Ray-Tracing

Demonstrates ray-tracing through an arbitrary surface shape,
using a sinusoidal profile as a test case. Validates that
Newton-Raphson converges on non-trivial geometries.
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import matplotlib.pyplot as plt

from acoustic_gpu.config import Transducer, WATER, STEEL_1020, ROI
from acoustic_gpu.surfaces.irregular import IrregularSurface
from acoustic_gpu.raytracing.cpu_ref import (
    find_refraction_point,
    find_refraction_points_cpu,
)
from acoustic_gpu.utils.visualization import plot_surface, plot_tof_table


def main():
    # — Configuration ————————————————————————————————————————————————————————
    c1 = WATER.c_longitudinal       # ≈ 1480 m/s
    c2 = STEEL_1020.c_longitudinal  # ≈ 5900 m/s

    transducer = Transducer(n_elements=32, pitch=0.6e-3, frequency=5e6)
    emitters   = transducer.element_positions()

    # Sinusoidal surface: z = 20mm + 3mm·sin(2π·x / 20mm)
    surface = IrregularSurface.sinusoidal(
        amplitude=0.003,    # 3 mm amplitude
        period=0.020,       # 20 mm period
        z_offset=0.020,     # 20 mm mean depth
        x_min=-0.018,
        x_max=0.018,
        n_points=500,
    )

    xS = surface.x_points
    zS = surface.z_points

    print(f"Surface: sinusoidal, A=3mm, T=20mm, z_mean=20mm")
    print(f"   {surface.n_points} points, spacing={surface.spacing*1e3:.3f} mm")

    # Focus grid below the surface
    roi = ROI(
        x_min=-0.018, x_max=0.018,
        z_min=0.025,  z_max=0.045,
        nx=60, nz=60,
    )
    focuses = roi.pixel_coordinates()

    print(f"Computing {len(emitters)} × {len(focuses)} = "
          f"{len(emitters) * len(focuses)} rays...")

    # — CPU Ray-Tracing ———————————————————————————————————————————————————————
    k_result, tof_result = find_refraction_points_cpu(
        emitters, focuses, xS, zS, c1, c2
    )

    print(f"TOF range: [{tof_result.min()*1e6:.2f}, {tof_result.max()*1e6:.2f}] µs")

    # — Check convergence quality —————————————————————————————————————————————
    # Verify all refraction points lie on the surface
    max_surface_error = 0.0
    elem_idx = transducer.n_elements // 2

    for j in range(0, len(focuses), 10):
        result = find_refraction_point(
            emitters[elem_idx, 0], emitters[elem_idx, 1],
            focuses[j, 0],         focuses[j, 1],
            xS, zS, c1, c2,
        )
        z_expected        = surface.evaluate(result.x_s)
        err               = abs(result.z_s - z_expected)
        max_surface_error = max(max_surface_error, err)

    print(f"Max surface fitting error: {max_surface_error*1e6:.2f} µm")

    # — Visualization —————————————————————————————————————————————————————————
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Sample rays
    n_rays        = 12
    focus_indices = np.linspace(0, len(focuses) - 1, n_rays, dtype=int)

    rays = []
    for fi in focus_indices:
        result = find_refraction_point(
            emitters[elem_idx, 0], emitters[elem_idx, 1],
            focuses[fi, 0],        focuses[fi, 1],
            xS, zS, c1, c2,
        )
        if result.converged:
            rays.append((
                emitters[elem_idx],
                np.array([result.x_s, result.z_s]),
                focuses[fi],
            ))

    plot_surface(
        surface, n_points=300, emitters=emitters,
        rays=rays,
        title="Irregular (Sinusoidal) Surface — Refracted Rays",
        ax=axes[0],
    )

    # TOF table
    tof_2d = tof_result[elem_idx].reshape(roi.nz, roi.nx)
    plot_tof_table(
        tof_2d, extent=roi.extent,
        title=f"TOF Table (Element {elem_idx})",
        ax=axes[1],
    )

    plt.tight_layout()
    plt.savefig("irregular_surface_result.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\nSaved: irregular_surface_result.png")


if __name__ == "__main__":
    main()
