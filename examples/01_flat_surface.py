"""
Example 01: Flat Surface Ray-tracing

Demonstrates ray-tracing through a flat water-steel interface,
comparing Newton-Raphson CPU results with the analytical Snell solution.
Visualizes the surface, refracted rays, and TOF table.
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import matplotlib.pyplot as plt

from acoustic_gpu.config import Transducer, WATER, STEEL_1020, ROI
from acoustic_gpu.surfaces.flat import FlatSurface
from acoustic_gpu.raytracing.cpu_ref import (
    find_refraction_point,
    find_refraction_points_sim,
    snell_analytical,
)
from acoustic_gpu.utils.visualization import plot_surface, plot_tof_table


def main():
    # — Configuration ————————————————————————————————————————————————————————
    c1 = WATER.c_longitudinal       # ≈ 1480 m/s
    c2 = STEEL_1020.c_longitudinal  # ≈ 5900 m/s

    transducer = Transducer(n_elements=32, pitch=0.6e-3, frequency=5e6)
    surface    = FlatSurface(origin=[0, 0, 0.020], normal=[0, 0, 1])  # 20 mm water path
    XS, ZS     = surface.get_points(300)
    emitters   = transducer.element_positions()   # shape (N_el, 2)  [x, z]

    # Focus grid below the surface
    roi = ROI(
        x_min=-0.012, x_max=0.012,
        z_min=0.022,  z_max=0.080,
        idx=100,
    )
    focuses = roi.pixel_coordinates()             # shape (N_foc, 2)

    print(f"Transducers: {transducer.n_elements} elements, "
          f"pitch={transducer.pitch*1e3:.2f} mm")
    print(f"Surface: flat at z_offset={0.020*1e3:.1f} mm")
    print(f"Computing {len(emitters)} × {len(focuses)} = "
          f"{len(emitters)*len(focuses)} rays...")

    # — CPU Ray-Tracing ———————————————————————————————————————————————————————
    k_result, tof_result = find_refraction_points_sim(
        emitters, focuses, XS, ZS, c1, c2
    )

    print(f"TOF range: [{tof_result.min()*1e6:.2f}, "
          f"{tof_result.max()*1e6:.2f}] µs")

    # — Validate against analytical solution —————————————————————————————————
    print("\n— Analytical Validation —")
    elem_idx   = transducer.n_elements // 3          # ≈ centre element
    focus_test = np.array([0.005, 0.030])            # (x, z) in metres

    xs_ana, tof_ana = snell_analytical(
        emitters[elem_idx],
        focus_test,
        surface_z=surface.origin[2],
        c1=c1, c2=c2,
    )

    result_nr = find_refraction_point(
        emitters[elem_idx], focus_test,
        XS, ZS, c1, c2,
    )

    print(f"Analytical:     x_s={xs_ana*1e3:.4f} mm,  TOF={tof_ana*1e6:.4f} µs")
    print(f"Newton-Raphson: x_s={result_nr.x_s*1e3:.4f} mm,  "
          f"TOF={result_nr.tof*1e6:.4f} µs")
    print(f"Error:          Δx_s={abs(xs_ana - result_nr.x_s)*1e3:.4f} mm, "
          f"ΔTOF={abs(tof_ana - result_nr.tof)*1e6:.4f} µs")
    print(f"Converged in {result_nr.n_iterations} iterations")

    # — Visualization —————————————————————————————————————————————————————————
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Surface with sample rays
    n_rays      = len(emitters)
    ray_indices = np.linspace(0, len(emitters) - 1, n_rays, dtype=int)
    focus_idx   = len(focuses) // 2   # centre focus

    rays = []
    for i in ray_indices:
        result = find_refraction_point(
            emitters[i], focuses[focus_idx],
            XS, ZS, c1, c2,
        )
        rays.append(
            np.array([
                emitters[i],
                np.array([result.x_s, result.z_s]),
                focuses[focus_idx],
            ])
        )

    # Build a (M, 2) surface_xz for plot_surface
    surface_xz = np.column_stack([XS, ZS])

    plot_surface(
        surface_xz=surface_xz,
        emitters=emitters,
        rays=rays,
        title="Flat Surface — Refracted Rays",
        ax=axes[0],
    )

    # Right: TOF Table (centre element)
    tof_2d = tof_result[elem_idx].reshape(roi.nz, roi.nx)
    plot_tof_table(
        tof_2d,
        extent=[roi.x_min * 1e3, roi.x_max * 1e3,
                roi.z_max * 1e3, roi.z_min * 1e3],
        title=f"TOF Table (Element {elem_idx})",
        ax=axes[1],
    )

    plt.tight_layout()
    plt.savefig("flat_surface_result.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\nSaved: flat_surface_result.png")


if __name__ == "__main__":
    main()
