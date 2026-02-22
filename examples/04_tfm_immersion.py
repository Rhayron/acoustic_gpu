"""
Example 04: Full TFM Pipeline for Immersion Testing

End-to-end demonstration of the acoustic GPU module:
1. Configure transducer, materials, and surface
2. Generate synthetic FMC data with point defects
3. Compute TOF tables via refracted ray-tracing
4. Reconstruct TFM image
5. Visualize results (B-scan, surface, rays, TFM image)
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import matplotlib.pyplot as plt
import time

from acoustic_gpu.config import Transducer, Material, ROI, WATER, STEEL_1020
from acoustic_gpu.surfaces.flat import FlatSurface
from acoustic_gpu.raytracing.cpu_ref import (
    find_refraction_point,
    compute_tof_table_cpu,
)
from acoustic_gpu.imaging.tfm import tfm_reconstruct_cpu, TFMResult
from acoustic_gpu.utils.synthetic import generate_fmc_data, Defect
from acoustic_gpu.utils.visualization import (
    plot_surface,
    plot_tof_table,
    plot_tfm_image,
    plot_bscan,
)


def main():
    print("=" * 60)
    print("  TFM Immersion Testing — Full Pipeline Demo")
    print("=" * 60)

    # ——————————————————————————————————————————————————————————————
    # 1. Configuration
    # ——————————————————————————————————————————————————————————————
    c1 = WATER.c_longitudinal       # ≈ 1480 m/s
    c2 = STEEL_1020.c_longitudinal  # ≈ 5900 m/s
    fs = 100e6                      # 100 MHz sampling
    n_samples = 4096

    transducer = Transducer(n_elements=32, pitch=0.6e-3, frequency=5e6)
    surface    = FlatSurface(z_offset=0.030)  # 30 mm offset

    # Two point defects at different depths
    defects = [
        Defect(x=-0.005, z=0.038, amplitude=1.0, name="Defect A"),
        Defect(x= 0.004, z=0.035, amplitude=0.7, name="Defect B"),
    ]

    # ROI below the surface
    roi = ROI(
        x_min=-0.012, x_max=0.012,
        z_min=0.032,  z_max=0.045,
        nx=100, nz=80,
    )

    print(f"\nTransducer: {transducer.n_elements} elements, "
          f"pitch={transducer.pitch*1e3:.1f} mm, f={transducer.frequency*1e-6:.1f} MHz")
    print(f"Surface:  flat at z={surface.z_offset*1e3:.1f} mm")
    print(f"Defects: {len(defects)}")
    for d in defects:
        print(f"   {d.name}: ({d.x*1e3:.1f}, {d.z*1e3:.1f}) mm, A={d.amplitude}")
    print(f"ROI:  x=[{roi.x_min*1e3:.1f}, {roi.x_max*1e3:.1f}] mm, "
          f"z=[{roi.z_min*1e3:.1f}, {roi.z_max*1e3:.1f}] mm")

    # ——————————————————————————————————————————————————————————————
    # 2. Generate Synthetic FMC Data
    # ——————————————————————————————————————————————————————————————
    print(f"\n[1/4] Generating synthetic FMC data...")
    t0 = time.perf_counter()

    fmc_data = generate_fmc_data(
        transducer,
        surface,
        defects,
        c1=c1, c2=c2,
        fs=fs,
        n_samples=n_samples,
        include_surface_echo=True,
        n_cycles=3,
    )

    t_fmc = time.perf_counter() - t0
    print(f"   FMC shape: {fmc_data.shape}, time: {t_fmc:.1f}s")

    # ——————————————————————————————————————————————————————————————
    # 3. Compute TOF Tables
    # ——————————————————————————————————————————————————————————————
    print(f"\n[2/4] Computing TOF tables (CPU)...")
    t0 = time.perf_counter()

    elem_pos = transducer.element_positions()
    pixels   = roi.pixel_coordinates()
    XS, ZS   = surface.get_points(300)

    tof_table = compute_tof_table_cpu(
        elem_pos, pixels, XS, ZS, c1, c2
    )

    t_tof  = time.perf_counter() - t0
    n_rays = len(elem_pos) * len(pixels)
    print(f"   {n_rays} rays computed in {t_tof:.1f}s "
          f"({n_rays / t_tof:.0f} rays/s)")

    # ——————————————————————————————————————————————————————————————
    # 4. TFM Reconstruction
    # ——————————————————————————————————————————————————————————————
    print(f"\n[3/4] TFM reconstruction (CPU)...")
    t0 = time.perf_counter()

    tfm_result = tfm_reconstruct_cpu(
        fmc_data, tof_table, fs,
        roi.nx, roi.nz, roi.extent,
    )

    t_tfm = time.perf_counter() - t0
    print(f"   Reconstruction time: {t_tfm:.1f}s")

    # Find peak locations
    peak_idx = np.unravel_index(
        np.argmax(tfm_result.envelope), tfm_result.envelope.shape
    )
    x_coords = np.linspace(roi.x_min, roi.x_max, roi.nx)
    z_coords = np.linspace(roi.z_min, roi.z_max, roi.nz)
    print(f"   Peak at ({x_coords[peak_idx[0]]*1e3:.2f}, "
          f"{z_coords[peak_idx[1]]*1e3:.2f}) mm")

    # ——————————————————————————————————————————————————————————————
    # 5. Visualization
    # ——————————————————————————————————————————————————————————————
    print(f"\n[4/4] Generating plots...")

    fig = plt.figure(figsize=(16, 10))

    # Panel 1: B-scan
    ax1 = fig.add_subplot(2, 2, 1)
    plot_bscan(fmc_data, transducer.n_elements // 2, fs, ax=ax1)

    # Panel 2: Surface + rays
    ax2      = fig.add_subplot(2, 2, 2)
    elem_idx     = transducer.n_elements // 2
    focus_center = np.array([0.0, 0.030])
    n_rays       = 8
    ray_indices  = np.linspace(0, transducer.n_elements - 1, n_rays, dtype=int)
    rays = []
    for i in ray_indices:
        result = find_refraction_point(
            elem_pos[i, 0], elem_pos[i, 1],
            focus_center[0], focus_center[1],
            XS, ZS, c1, c2,
        )
        rays.append((
            elem_pos[i],
            np.array([result.x_s, result.z_s]),
            focus_center,
        ))
    plot_surface(
        surface, emitters=elem_pos, rays=rays,
        title="Surface + Refracted Rays", ax=ax2,
    )
    # Mark defects
    for d in defects:
        ax2.plot(d.x * 1e3, d.z * 1e3, "+", markersize=12)

    # Panel 3: TOF table
    ax3    = fig.add_subplot(2, 2, 3)
    tof_2d = tof_table[elem_idx].reshape(roi.nz, roi.nx)
    plot_tof_table(tof_2d, extent=roi.extent,
                   title=f"TOF (Element {elem_idx})", ax=ax3)

    # Panel 4: TFM image
    ax4 = fig.add_subplot(2, 2, 4)
    plot_tfm_image(tfm_result, db_range=40, title="TFM Image", ax=ax4)
    # Mark defect positions
    for d in defects:
        ax4.plot(d.x * 1e3, d.z * 1e3, "w+", markersize=10, markeredgewidth=2)

    plt.suptitle("Ultrasonic Immersion Testing — Full TFM Pipeline", fontsize=14)
    plt.tight_layout()
    plt.savefig("tfm_immersion_result.png", dpi=150, bbox_inches="tight")
    plt.show()

    print(f"\n" + "=" * 60)
    print(f"   Total time: {t_fmc + t_tof + t_tfm:.1f}s")
    print(f"   Saved: tfm_immersion_result.png")
    print("=" * 60)


if __name__ == "__main__":
    main()
