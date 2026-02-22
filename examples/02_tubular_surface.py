"""
Example 02: Tubular (Cylindrical) Surface Ray-Tracing

Demonstrates ray-tracing through a cylindrical water-steel interface,
simulating inspection of a pipe/tube element. Visualizes the curved
surface, refracted rays, and TOF table.
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import matplotlib.pyplot as plt

from acoustic_gpu.config import Transducer, WATER, STEEL_1020, ROI
from acoustic_gpu.surfaces.tubular import TubularSurface
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

    # Cylinder with R=25 mm, centred below the transducer
    surface = TubularSurface(
        radius=0.025,
        center_x=0.0,
        center_z=0.045,
        outer=True,
    )

    xs, zs         = surface.get_points(500)
    apex_x, apex_z = surface.apex()

    print(f"Cylinder: R={surface.radius*1e3:.1f} mm, "
          f"center=({surface.center_x*1e3:.1f}, {surface.center_z*1e3:.1f}) mm")
    print(f"Apex at z={apex_z*1e3:.1f} mm")

    # Focus grid inside the tube wall
    roi = ROI(
        x_min=-0.010, x_max=0.010,
        z_min=apex_z + 0.002, z_max=apex_z + 0.020,
        nx=60, nz=60,
    )
    focuses = roi.pixel_coordinates()

    print(f"Computing {len(emitters)} × {len(focuses)} = "
          f"{len(emitters) * len(focuses)} rays...")

    # — CPU Ray-Tracing ———————————————————————————————————————————————————————
    k_result, tof_result = find_refraction_points_cpu(
        emitters, focuses, xs, zs, c1, c2
    )

    print(f"TOF range: [{tof_result.min()*1e6:.2f}, {tof_result.max()*1e6:.2f}] µs")

    # — Visualization —————————————————————————————————————————————————————————
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Sample rays from centre element to various focus points
    elem_idx      = transducer.n_elements // 2
    n_rays        = 10
    focus_indices = np.linspace(0, len(focuses) - 1, n_rays, dtype=int)

    rays = []
    for fi in focus_indices:
        result = find_refraction_point(
            emitters[elem_idx, 0], emitters[elem_idx, 1],
            focuses[fi, 0],        focuses[fi, 1],
            xs, zs, c1, c2,
        )
        if result.converged:
            rays.append((
                emitters[elem_idx],
                np.array([result.x_s, result.z_s]),
                focuses[fi],
            ))

    plot_surface(
        surface, n_points=300, emitters=emitters,
        rays=rays, focuses=focuses[:170],
        title="Tubular Surface — Refracted Rays",
        ax=axes[0],
    )

    # TOF table for centre element
    tof_2d = tof_result[elem_idx].reshape(roi.nz, roi.nx)
    plot_tof_table(
        tof_2d, extent=roi.extent,
        title=f"TOF Table (Element {elem_idx})",
        ax=axes[1],
    )

    plt.tight_layout()
    plt.savefig("tubular_surface_result.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\nSaved: tubular_surface_result.png")


if __name__ == "__main__":
    main()
