"""
Benchmark: Bristol University TFM Dataset vs acoustic_gpu Pipeline

Dataset: 5 MHz, 18-element linear array on 50mm mild steel
Defect:  Side-Drilled Hole (SDH) at ~25mm depth
Velocity: 5850 m/s (longitudinal, contact mode => c1 = c2)

Steps:
  1. Load .mat data and parse FMC
  2. Run Bristol reference TFM (interpolation)
  3. Run acoustic_gpu TFM pipeline (contact: flat surface at z=0)
  4. Compare images, peak locations, and timing
  5. Generate comprehensive analysis plots
"""

import sys
import pathlib
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.io import loadmat
from scipy.signal import butter, lfilter, hilbert
from scipy.interpolate import interp1d

from acoustic_gpu.config import Transducer, ROI, Material
from acoustic_gpu.surfaces.flat import FlatSurface
from acoustic_gpu.raytracing.cpu_ref import compute_tof_table_cpu
from acoustic_gpu.imaging.tfm import tfm_reconstruct_cpu, TFMResult

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
DATASET_PATH = pathlib.Path(__file__).resolve().parent / "datasets" / "bristol_tfm" / "sdata-5mhz-els18-steel5850.mat"
OUTPUT_DIR = pathlib.Path(__file__).resolve().parent / "benchmark_results"
OUTPUT_DIR.mkdir(exist_ok=True)

# Known ground truth
DEFECT_Z_TRUE = 25.0  # mm depth of SDH
PH_VELOCITY = 5850.0  # m/s longitudinal velocity in mild steel
FC = 5e6              # 5 MHz center frequency

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load Bristol Dataset
# ─────────────────────────────────────────────────────────────────────────────
def load_bristol_dataset(mat_path):
    """Load and parse the Bristol .mat FMC dataset."""
    print("=" * 70)
    print("  Loading Bristol University FMC Dataset")
    print("=" * 70)

    data = loadmat(str(mat_path))
    exp_data = data['exp_data']

    # Time axis
    timx = exp_data['time'][0, 0][:, 0]
    # Raw time-domain data (all FMC pairs flattened)
    time_data = exp_data['time_data'][0, 0]
    # Tx/Rx indices (MATLAB 1-indexed → Python 0-indexed)
    tx = exp_data['tx'][0, 0][0, :] - 1
    rx = exp_data['rx'][0, 0][0, :] - 1
    # Element positions
    el_xc = exp_data['array'][0, 0]['el_xc'][0, 0][0, :].astype(float)
    el_yc = exp_data['array'][0, 0]['el_yc'][0, 0][0, :].astype(float)
    el_zc = exp_data['array'][0, 0]['el_zc'][0, 0][0, :].astype(float)

    # Derived
    fs = 1.0 / (timx[1] - timx[0])
    n_elements = len(el_xc)
    n_samples = len(timx)
    n_pairs = len(tx)

    print(f"\n  Transducer:    {n_elements} elements")
    print(f"  Frequency:     {FC * 1e-6:.0f} MHz")
    print(f"  Sampling rate: {fs * 1e-6:.1f} MHz")
    print(f"  Samples/scan:  {n_samples}")
    print(f"  FMC pairs:     {n_pairs} (expected {n_elements**2})")
    print(f"  Velocity:      {PH_VELOCITY} m/s")
    print(f"  Element X range: [{el_xc.min()*1e3:.2f}, {el_xc.max()*1e3:.2f}] mm")

    # Compute pitch from element positions
    pitch = np.diff(el_xc).mean()
    print(f"  Pitch:         {pitch*1e3:.3f} mm")

    # Reshape to FMC 3D array (N_tx, N_rx, N_samples)
    fmc_3d = np.zeros((n_elements, n_elements, n_samples))
    for i in range(n_pairs):
        fmc_3d[int(tx[i]), int(rx[i]), :] = time_data[:, i]

    return {
        'fmc_3d': fmc_3d,
        'time_data': time_data,
        'timx': timx,
        'tx': tx.astype(int),
        'rx': rx.astype(int),
        'el_xc': el_xc,
        'el_yc': el_yc,
        'el_zc': el_zc,
        'fs': fs,
        'n_elements': n_elements,
        'n_samples': n_samples,
        'pitch': pitch,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. Bandpass Filter
# ─────────────────────────────────────────────────────────────────────────────
def bandpass_filter(time_data, fs, fc=5e6, bw_pct=50, order=5):
    """Apply Butterworth bandpass filter."""
    lowcut = fc * (1 - bw_pct / 200)
    highcut = fc * (1 + bw_pct / 200)
    b, a = butter(order, [lowcut, highcut], fs=fs, btype='bandpass')
    return lfilter(b, a, time_data, axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Bristol Reference TFM
# ─────────────────────────────────────────────────────────────────────────────
def run_bristol_tfm(dataset, filtered_data, roi_params):
    """Run TFM using Bristol's interpolation method."""
    print("\n[Bristol TFM] Running reference implementation...")

    x = np.linspace(roi_params['x_min'], roi_params['x_max'], roi_params['nx'])
    z = np.linspace(roi_params['z_min'], roi_params['z_max'], roi_params['nz'])

    el_xc = dataset['el_xc']
    timx = dataset['timx']
    tx = dataset['tx']
    rx = dataset['rx']
    n_elem = dataset['n_elements']

    # Compute delay maps for each element
    x_mg, z_mg, t_mg = np.meshgrid(x, z, el_xc)
    delay = np.sqrt((t_mg - x_mg)**2 + z_mg**2) / PH_VELOCITY

    # TFM reconstruction
    t0 = time.perf_counter()
    II = np.zeros([len(z), len(x)], dtype=complex)
    for ii in range(filtered_data.shape[1]):
        itp = interp1d(timx, hilbert(filtered_data[:, ii]),
                       kind='linear', fill_value=0, bounds_error=False)
        II += itp(delay[:, :, tx[ii]] + delay[:, :, rx[ii]])

    t_elapsed = time.perf_counter() - t0

    II_abs = np.abs(II)
    II_db = 20 * np.log10(II_abs / np.max(II_abs) + 1e-30)

    # Find peak (defect location)
    peak_idx = np.unravel_index(np.argmax(II_abs), II_abs.shape)
    peak_z = z[peak_idx[0]] * 1e3  # mm
    peak_x = x[peak_idx[1]] * 1e3  # mm

    print(f"  Time: {t_elapsed:.2f}s")
    print(f"  Peak at: ({peak_x:.2f}, {peak_z:.2f}) mm")
    print(f"  Peak z error vs ground truth: {abs(peak_z - DEFECT_Z_TRUE):.2f} mm")

    return {
        'image': II_abs,
        'image_db': II_db,
        'x': x,
        'z': z,
        'peak_x': peak_x,
        'peak_z': peak_z,
        'time': t_elapsed,
        'extent': [x[0]*1e3, x[-1]*1e3, z[-1]*1e3, z[0]*1e3],
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. acoustic_gpu Pipeline TFM
# ─────────────────────────────────────────────────────────────────────────────
def run_acoustic_gpu_tfm(dataset, roi_params):
    """Run TFM using our acoustic_gpu pipeline.

    Contact mode: surface at z=0, c1 = c2 = 5850 m/s
    """
    print("\n[acoustic_gpu TFM] Running our pipeline...")

    n_elem = dataset['n_elements']
    pitch = dataset['pitch']
    fs = dataset['fs']
    fmc_3d = dataset['fmc_3d']

    # Configure transducer
    transducer = Transducer(n_elements=n_elem, pitch=pitch, frequency=FC)

    # Override element positions to match the Bristol dataset positions exactly
    el_xc = dataset['el_xc']

    # For contact mode: surface at z=0, c1 = c2
    c1 = PH_VELOCITY
    c2 = PH_VELOCITY
    surface = FlatSurface(z_offset=0.0)  # Contact: surface at z=0

    # ROI
    roi = ROI(
        x_min=roi_params['x_min'],
        x_max=roi_params['x_max'],
        z_min=roi_params['z_min'],
        z_max=roi_params['z_max'],
        nx=roi_params['nx'],
        nz=roi_params['nz'],
    )

    # Element positions as (x, z) with z=0
    elem_pos = np.column_stack([el_xc, np.zeros(n_elem)])
    pixels = roi.pixel_coordinates()
    xS, zS = surface.get_points(300)

    # Compute TOF table
    t0 = time.perf_counter()
    tof_table = compute_tof_table_cpu(elem_pos, pixels, xS, zS, c1, c2)
    t_tof = time.perf_counter() - t0
    print(f"  TOF computation: {t_tof:.2f}s ({n_elem * roi.n_pixels} rays)")

    # TFM reconstruction
    t0 = time.perf_counter()
    tfm_result = tfm_reconstruct_cpu(
        fmc_3d, tof_table, fs,
        roi.nx, roi.nz, roi.extent,
    )
    t_tfm = time.perf_counter() - t0
    print(f"  TFM reconstruction: {t_tfm:.2f}s")

    # Find peak
    peak_idx = np.unravel_index(
        np.argmax(tfm_result.envelope), tfm_result.envelope.shape
    )
    x_coords = np.linspace(roi.x_min, roi.x_max, roi.nx)
    z_coords = np.linspace(roi.z_min, roi.z_max, roi.nz)
    peak_x = x_coords[peak_idx[1]] * 1e3
    peak_z = z_coords[peak_idx[0]] * 1e3

    print(f"  Peak at: ({peak_x:.2f}, {peak_z:.2f}) mm")
    print(f"  Peak z error vs ground truth: {abs(peak_z - DEFECT_Z_TRUE):.2f} mm")

    return {
        'tfm_result': tfm_result,
        'peak_x': peak_x,
        'peak_z': peak_z,
        'time_tof': t_tof,
        'time_tfm': t_tfm,
        'time_total': t_tof + t_tfm,
        'tof_table': tof_table,
        'elem_pos': elem_pos,
        'roi': roi,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. Multi-resolution benchmark
# ─────────────────────────────────────────────────────────────────────────────
def run_resolution_benchmark(dataset):
    """Run TFM at multiple resolutions to measure scaling."""
    print("\n" + "=" * 70)
    print("  Resolution Scaling Benchmark")
    print("=" * 70)

    resolutions = [
        (25, 30),
        (50, 60),
        (75, 90),
        (100, 120),
        (150, 180),
    ]

    results = []

    for nx, nz in resolutions:
        roi_params = {
            'x_min': -25e-3, 'x_max': 25e-3,
            'z_min': 0.0, 'z_max': 50e-3,
            'nx': nx, 'nz': nz,
        }

        n_elem = dataset['n_elements']
        el_xc = dataset['el_xc']
        fs = dataset['fs']
        fmc_3d = dataset['fmc_3d']

        c1 = PH_VELOCITY
        c2 = PH_VELOCITY
        surface = FlatSurface(z_offset=0.0)

        roi = ROI(**roi_params)
        elem_pos = np.column_stack([el_xc, np.zeros(n_elem)])
        pixels = roi.pixel_coordinates()
        xS, zS = surface.get_points(300)

        # TOF
        t0 = time.perf_counter()
        tof_table = compute_tof_table_cpu(elem_pos, pixels, xS, zS, c1, c2)
        t_tof = time.perf_counter() - t0

        # TFM
        t0 = time.perf_counter()
        tfm_result = tfm_reconstruct_cpu(
            fmc_3d, tof_table, fs, roi.nx, roi.nz, roi.extent,
        )
        t_tfm = time.perf_counter() - t0

        n_pixels = nx * nz
        n_rays = n_elem * n_pixels

        # Find peak
        peak_idx = np.unravel_index(
            np.argmax(tfm_result.envelope), tfm_result.envelope.shape
        )
        x_coords = np.linspace(roi.x_min, roi.x_max, roi.nx)
        z_coords = np.linspace(roi.z_min, roi.z_max, roi.nz)
        peak_z = z_coords[peak_idx[0]] * 1e3

        res = {
            'nx': nx, 'nz': nz,
            'n_pixels': n_pixels,
            'n_rays': n_rays,
            't_tof': t_tof,
            't_tfm': t_tfm,
            't_total': t_tof + t_tfm,
            'peak_z': peak_z,
            'z_error': abs(peak_z - DEFECT_Z_TRUE),
            'pixel_size_mm': (50.0 / nz),
        }
        results.append(res)

        print(f"  {nx}×{nz} ({n_pixels:,} px, {n_rays:,} rays): "
              f"TOF={t_tof:.2f}s, TFM={t_tfm:.2f}s, "
              f"Total={t_tof + t_tfm:.2f}s, "
              f"z_err={res['z_error']:.2f}mm")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 6. SNR Analysis
# ─────────────────────────────────────────────────────────────────────────────
def compute_snr(image_db, peak_idx, roi_size=5, noise_margin=20):
    """Compute SNR from TFM image.

    Signal region: around the peak.
    Noise region: everything far from the peak.
    """
    nz, nx = image_db.shape

    # Signal region
    r0 = max(0, peak_idx[0] - roi_size)
    r1 = min(nz, peak_idx[0] + roi_size + 1)
    c0 = max(0, peak_idx[1] - roi_size)
    c1_idx = min(nx, peak_idx[1] + roi_size + 1)
    signal_region = image_db[r0:r1, c0:c1_idx]
    signal_mean = np.mean(signal_region)

    # Noise: mask around peak
    mask = np.ones_like(image_db, dtype=bool)
    nr0 = max(0, peak_idx[0] - noise_margin)
    nr1 = min(nz, peak_idx[0] + noise_margin + 1)
    nc0 = max(0, peak_idx[1] - noise_margin)
    nc1 = min(nx, peak_idx[1] + noise_margin + 1)
    mask[nr0:nr1, nc0:nc1] = False
    noise_region = image_db[mask]
    noise_mean = np.mean(noise_region)

    return signal_mean - noise_mean  # Already in dB


# ─────────────────────────────────────────────────────────────────────────────
# 7. Generate Plots
# ─────────────────────────────────────────────────────────────────────────────
def generate_plots(dataset, bristol_res, gpu_res, resolution_results, filtered_data):
    """Generate comprehensive analysis plots."""
    print("\n" + "=" * 70)
    print("  Generating Analysis Plots")
    print("=" * 70)

    # ── Style configuration ──────────────────────────────────────────────
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 11,
        'font.family': 'sans-serif',
        'axes.titlesize': 13,
        'axes.labelsize': 11,
        'figure.facecolor': '#0d1117',
        'axes.facecolor': '#161b22',
        'text.color': '#c9d1d9',
        'axes.labelcolor': '#c9d1d9',
        'xtick.color': '#8b949e',
        'ytick.color': '#8b949e',
        'axes.edgecolor': '#30363d',
        'grid.color': '#21262d',
        'savefig.facecolor': '#0d1117',
        'savefig.edgecolor': '#0d1117',
    })

    # ══════════════════════════════════════════════════════════════════════
    # PLOT 1: TFM Comparison (Bristol vs acoustic_gpu)
    # ══════════════════════════════════════════════════════════════════════
    fig = plt.figure(figsize=(18, 8))
    gs = GridSpec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.3)

    # Bristol TFM
    ax1 = fig.add_subplot(gs[0])
    bristol_db = bristol_res['image_db'].copy()
    bristol_db[bristol_db < -40] = -40
    im1 = ax1.imshow(
        bristol_db,
        extent=bristol_res['extent'],
        cmap='inferno', vmin=-40, vmax=0,
        aspect='auto',
    )
    ax1.plot(bristol_res['peak_x'], bristol_res['peak_z'],
             'w+', markersize=15, markeredgewidth=2.5, label='Peak')
    ax1.axhline(y=DEFECT_Z_TRUE, color='#58a6ff', linestyle='--',
                alpha=0.7, label=f'Ground truth ({DEFECT_Z_TRUE}mm)')
    ax1.set_xlabel('x (mm)')
    ax1.set_ylabel('z (mm)')
    ax1.set_title('Bristol Reference TFM', fontweight='bold', color='#58a6ff')
    ax1.legend(loc='lower right', fontsize=9, facecolor='#21262d', edgecolor='#30363d')

    # acoustic_gpu TFM
    ax2 = fig.add_subplot(gs[1])
    gpu_extent = (
        gpu_res['tfm_result'].extent[0] * 1e3,
        gpu_res['tfm_result'].extent[1] * 1e3,
        gpu_res['tfm_result'].extent[2] * 1e3,
        gpu_res['tfm_result'].extent[3] * 1e3,
    )
    gpu_db = gpu_res['tfm_result'].image_db.copy()
    gpu_db[gpu_db < -40] = -40
    im2 = ax2.imshow(
        gpu_db,
        extent=gpu_extent,
        cmap='inferno', vmin=-40, vmax=0,
        aspect='auto',
    )
    ax2.plot(gpu_res['peak_x'], gpu_res['peak_z'],
             'w+', markersize=15, markeredgewidth=2.5, label='Peak')
    ax2.axhline(y=DEFECT_Z_TRUE, color='#58a6ff', linestyle='--',
                alpha=0.7, label=f'Ground truth ({DEFECT_Z_TRUE}mm)')
    ax2.set_xlabel('x (mm)')
    ax2.set_ylabel('z (mm)')
    ax2.set_title('acoustic_gpu Pipeline TFM', fontweight='bold', color='#3fb950')
    ax2.legend(loc='lower right', fontsize=9, facecolor='#21262d', edgecolor='#30363d')

    # Shared colorbar
    cax = fig.add_subplot(gs[2])
    cb = fig.colorbar(im2, cax=cax, label='Amplitude (dB)')
    cb.ax.yaxis.label.set_color('#c9d1d9')

    fig.suptitle('TFM Reconstruction Comparison — Bristol SDH Dataset',
                 fontsize=16, fontweight='bold', color='#f0f6fc', y=1.02)

    plt.savefig(OUTPUT_DIR / "01_tfm_comparison.png", dpi=200, bbox_inches='tight')
    print("  ✓ Saved 01_tfm_comparison.png")
    plt.close()

    # ══════════════════════════════════════════════════════════════════════
    # PLOT 2: B-scan + A-scan Analysis
    # ══════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    fmc_3d = dataset['fmc_3d']
    timx = dataset['timx']
    fs = dataset['fs']
    n_elem = dataset['n_elements']

    # B-scan (raw)
    ax = axes[0, 0]
    mid_elem = n_elem // 2
    bscan_raw = fmc_3d[mid_elem, :, :]
    t_us = timx * 1e6
    vmax = np.max(np.abs(bscan_raw)) * 0.7
    ax.imshow(
        bscan_raw, aspect='auto', cmap='seismic',
        extent=[t_us[0], t_us[-1], n_elem - 0.5, -0.5],
        vmin=-vmax, vmax=vmax,
    )
    ax.set_xlabel('Time (μs)')
    ax.set_ylabel('Receiver Element')
    ax.set_title(f'B-scan Raw (TX elem {mid_elem})', fontweight='bold')

    # B-scan (filtered)
    ax = axes[0, 1]
    # Reconstruct filtered B-scan from filtered_data
    bscan_filt = np.zeros((n_elem, dataset['n_samples']))
    for i in range(len(dataset['tx'])):
        if dataset['tx'][i] == mid_elem:
            bscan_filt[dataset['rx'][i], :] = filtered_data[:, i]
    vmax_f = np.max(np.abs(bscan_filt)) * 0.7
    ax.imshow(
        bscan_filt, aspect='auto', cmap='seismic',
        extent=[t_us[0], t_us[-1], n_elem - 0.5, -0.5],
        vmin=-vmax_f, vmax=vmax_f,
    )
    ax.set_xlabel('Time (μs)')
    ax.set_ylabel('Receiver Element')
    ax.set_title(f'B-scan Filtered (TX elem {mid_elem})', fontweight='bold')

    # A-scan (pulse-echo: tx=rx=mid)
    ax = axes[1, 0]
    ascan = fmc_3d[mid_elem, mid_elem, :]
    ax.plot(t_us, ascan, color='#58a6ff', linewidth=0.5, alpha=0.8)
    # Compute and overlay Hilbert envelope
    env = np.abs(hilbert(ascan))
    ax.plot(t_us, env, color='#f85149', linewidth=1.2, label='Envelope')
    ax.set_xlabel('Time (μs)')
    ax.set_ylabel('Amplitude')
    ax.set_title('A-scan Pulse-Echo (tx=rx=center)', fontweight='bold')
    ax.legend(facecolor='#21262d', edgecolor='#30363d')
    ax.grid(True, alpha=0.3)

    # Mark expected SDH time
    t_sdh = 2 * DEFECT_Z_TRUE * 1e-3 / PH_VELOCITY * 1e6
    ax.axvline(x=t_sdh, color='#3fb950', linestyle='--', alpha=0.7,
               label=f'SDH t≈{t_sdh:.1f}μs')
    ax.legend(facecolor='#21262d', edgecolor='#30363d')

    # Frequency spectrum
    ax = axes[1, 1]
    n_fft = int(2**np.ceil(np.log2(len(ascan))))
    freq = np.fft.rfftfreq(n_fft, d=1/fs) * 1e-6  # MHz
    spec = np.abs(np.fft.rfft(ascan, n=n_fft))
    spec_filt = np.abs(np.fft.rfft(bscan_filt[mid_elem], n=n_fft))
    ax.plot(freq, spec / spec.max(), color='#8b949e', linewidth=1, label='Raw', alpha=0.7)
    ax.plot(freq, spec_filt / spec_filt.max() if spec_filt.max() > 0 else spec_filt,
            color='#f85149', linewidth=1.2, label='Filtered')
    ax.axvline(x=FC * 1e-6, color='#3fb950', linestyle='--', alpha=0.7,
               label=f'fc={FC*1e-6:.0f} MHz')
    ax.set_xlim(0, 15)
    ax.set_xlabel('Frequency (MHz)')
    ax.set_ylabel('Normalized Amplitude')
    ax.set_title('Frequency Spectrum', fontweight='bold')
    ax.legend(facecolor='#21262d', edgecolor='#30363d')
    ax.grid(True, alpha=0.3)

    fig.suptitle('FMC Data Analysis — Bristol Dataset',
                 fontsize=16, fontweight='bold', color='#f0f6fc')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "02_fmc_analysis.png", dpi=200, bbox_inches='tight')
    print("  ✓ Saved 02_fmc_analysis.png")
    plt.close()

    # ══════════════════════════════════════════════════════════════════════
    # PLOT 3: Resolution Scaling
    # ══════════════════════════════════════════════════════════════════════
    if resolution_results:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

        n_pixels = [r['n_pixels'] for r in resolution_results]
        t_tof = [r['t_tof'] for r in resolution_results]
        t_tfm = [r['t_tfm'] for r in resolution_results]
        t_total = [r['t_total'] for r in resolution_results]
        z_errors = [r['z_error'] for r in resolution_results]
        pixel_sizes = [r['pixel_size_mm'] for r in resolution_results]
        labels = [f"{r['nx']}×{r['nz']}" for r in resolution_results]

        # Timing vs resolution
        ax = axes[0]
        x_pos = np.arange(len(labels))
        width = 0.35
        bars1 = ax.bar(x_pos - width/2, t_tof, width, label='Ray-tracing (TOF)',
                        color='#58a6ff', alpha=0.85, edgecolor='#30363d')
        bars2 = ax.bar(x_pos + width/2, t_tfm, width, label='TFM Reconstruct',
                        color='#3fb950', alpha=0.85, edgecolor='#30363d')
        ax.set_xlabel('Resolution (nx × nz)')
        ax.set_ylabel('Time (s)')
        ax.set_title('Computation Time vs Resolution', fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=30, ha='right')
        ax.legend(facecolor='#21262d', edgecolor='#30363d')
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar in bars1:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.05,
                    f'{h:.1f}s', ha='center', va='bottom', fontsize=8)
        for bar in bars2:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.05,
                    f'{h:.1f}s', ha='center', va='bottom', fontsize=8)

        # Scaling (log-log)
        ax = axes[1]
        ax.loglog(n_pixels, t_tof, 'o-', color='#58a6ff', linewidth=2,
                  markersize=8, label='TOF')
        ax.loglog(n_pixels, t_tfm, 's-', color='#3fb950', linewidth=2,
                  markersize=8, label='TFM')
        ax.loglog(n_pixels, t_total, 'D-', color='#f85149', linewidth=2,
                  markersize=8, label='Total')
        # Add O(N) reference line
        ref_x = np.array([n_pixels[0], n_pixels[-1]])
        ref_y = np.array([t_total[0], t_total[0] * (ref_x[1] / ref_x[0])])
        ax.loglog(ref_x, ref_y, '--', color='#8b949e', alpha=0.5, label='O(N) ref')
        ax.set_xlabel('Number of Pixels')
        ax.set_ylabel('Time (s)')
        ax.set_title('Scaling Behavior (log-log)', fontweight='bold')
        ax.legend(facecolor='#21262d', edgecolor='#30363d')
        ax.grid(True, alpha=0.3, which='both')

        # Localization error vs pixel size
        ax = axes[2]
        ax.plot(pixel_sizes, z_errors, 'o-', color='#d2a8ff', linewidth=2,
                markersize=10)
        for i, (ps, err) in enumerate(zip(pixel_sizes, z_errors)):
            ax.annotate(f'{err:.2f}mm', (ps, err),
                       textcoords="offset points", xytext=(5, 10),
                       fontsize=9, color='#d2a8ff')
        ax.set_xlabel('Pixel Size (mm)')
        ax.set_ylabel('z-Localization Error (mm)')
        ax.set_title('Defect Localization Accuracy', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.invert_xaxis()

        fig.suptitle('Resolution Scaling Analysis — acoustic_gpu Pipeline',
                     fontsize=16, fontweight='bold', color='#f0f6fc')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "03_resolution_scaling.png", dpi=200, bbox_inches='tight')
        print("  ✓ Saved 03_resolution_scaling.png")
        plt.close()

    # ══════════════════════════════════════════════════════════════════════
    # PLOT 4: Detailed TFM with Line Profiles
    # ══════════════════════════════════════════════════════════════════════
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 2, hspace=0.35, wspace=0.3)

    # TFM image (acoustic_gpu, zoomed)
    ax = fig.add_subplot(gs[0, 0])
    gpu_db_zoom = gpu_res['tfm_result'].image_db.copy()
    gpu_db_zoom[gpu_db_zoom < -40] = -40
    im = ax.imshow(
        gpu_db_zoom,
        extent=gpu_extent,
        cmap='inferno', vmin=-40, vmax=0,
        aspect='auto',
    )
    ax.plot(gpu_res['peak_x'], gpu_res['peak_z'],
            'w+', markersize=15, markeredgewidth=2.5)
    ax.axhline(y=DEFECT_Z_TRUE, color='#58a6ff', linestyle='--', alpha=0.7)

    # Add crosshair lines for profiles
    ax.axhline(y=gpu_res['peak_z'], color='#3fb950', linestyle=':', alpha=0.5)
    ax.axvline(x=gpu_res['peak_x'], color='#d2a8ff', linestyle=':', alpha=0.5)

    ax.set_xlabel('x (mm)')
    ax.set_ylabel('z (mm)')
    ax.set_title('TFM Image (acoustic_gpu)', fontweight='bold', color='#3fb950')
    fig.colorbar(im, ax=ax, label='dB', shrink=0.9)

    # Horizontal profile through peak (z = peak_z)
    ax = fig.add_subplot(gs[0, 1])
    roi = gpu_res['roi']
    envelope = gpu_res['tfm_result'].envelope
    env_db = gpu_res['tfm_result'].image_db

    peak_idx = np.unravel_index(np.argmax(envelope), envelope.shape)
    x_coords_mm = np.linspace(roi.x_min, roi.x_max, roi.nx) * 1e3

    h_profile = env_db[peak_idx[0], :]
    ax.plot(x_coords_mm, h_profile, color='#3fb950', linewidth=2)
    ax.axvline(x=gpu_res['peak_x'], color='#d2a8ff', linestyle='--', alpha=0.7,
               label=f"Peak x={gpu_res['peak_x']:.2f}mm")
    ax.axhline(y=-6, color='#f85149', linestyle=':', alpha=0.5, label='-6 dB')
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('Amplitude (dB)')
    ax.set_title(f'Horizontal Profile @ z={gpu_res["peak_z"]:.1f}mm',
                 fontweight='bold')
    ax.legend(facecolor='#21262d', edgecolor='#30363d')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-45, 5)

    # Vertical profile through peak (x = peak_x)
    ax = fig.add_subplot(gs[1, 0])
    z_coords_mm = np.linspace(roi.z_min, roi.z_max, roi.nz) * 1e3

    v_profile = env_db[:, peak_idx[1]]
    ax.plot(z_coords_mm, v_profile, color='#d2a8ff', linewidth=2)
    ax.axvline(x=gpu_res['peak_z'], color='#3fb950', linestyle='--', alpha=0.7,
               label=f"Peak z={gpu_res['peak_z']:.2f}mm")
    ax.axvline(x=DEFECT_Z_TRUE, color='#58a6ff', linestyle='--', alpha=0.7,
               label=f"Ground Truth z={DEFECT_Z_TRUE}mm")
    ax.axhline(y=-6, color='#f85149', linestyle=':', alpha=0.5, label='-6 dB')
    ax.set_xlabel('z (mm)')
    ax.set_ylabel('Amplitude (dB)')
    ax.set_title(f'Vertical Profile @ x={gpu_res["peak_x"]:.1f}mm',
                 fontweight='bold')
    ax.legend(facecolor='#21262d', edgecolor='#30363d', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-45, 5)

    # Summary statistics card
    ax = fig.add_subplot(gs[1, 1])
    ax.axis('off')

    # Compute -6dB width (lateral resolution)
    above_6db = h_profile >= -6
    if np.any(above_6db):
        x_above = x_coords_mm[above_6db]
        lateral_res = x_above[-1] - x_above[0]
    else:
        lateral_res = float('nan')

    # Compute -6dB height (axial resolution)
    above_6db_v = v_profile >= -6
    if np.any(above_6db_v):
        z_above = z_coords_mm[above_6db_v]
        axial_res = z_above[-1] - z_above[0]
    else:
        axial_res = float('nan')

    # SNR
    snr = compute_snr(env_db, peak_idx)

    stats_text = (
        "╔══════════════════════════════════════╗\n"
        "║      BENCHMARK RESULTS SUMMARY       ║\n"
        "╠══════════════════════════════════════╣\n"
        f"║  Dataset:  Bristol SDH (18 elem)     ║\n"
        f"║  Material: Mild Steel, 5850 m/s      ║\n"
        f"║  Defect:   SDH @ 25mm depth          ║\n"
        "╠══════════════════════════════════════╣\n"
        f"║  Peak (x,z):  ({gpu_res['peak_x']:+.2f}, {gpu_res['peak_z']:.2f}) mm  ║\n"
        f"║  z Error:  {abs(gpu_res['peak_z'] - DEFECT_Z_TRUE):.2f} mm              ║\n"
        f"║  Lateral Res (-6dB): {lateral_res:.2f} mm       ║\n"
        f"║  Axial Res (-6dB):   {axial_res:.2f} mm       ║\n"
        f"║  Image SNR:   {snr:.1f} dB              ║\n"
        "╠══════════════════════════════════════╣\n"
        "║  TIMING                              ║\n"
        f"║  Bristol ref: {bristol_res['time']:.2f}s              ║\n"
        f"║  GPU TOF:     {gpu_res['time_tof']:.2f}s              ║\n"
        f"║  GPU TFM:     {gpu_res['time_tfm']:.2f}s              ║\n"
        f"║  GPU Total:   {gpu_res['time_total']:.2f}s              ║\n"
        "╚══════════════════════════════════════╝"
    )

    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
            fontfamily='monospace', fontsize=10, color='#3fb950',
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='#0d1117',
                      edgecolor='#3fb950', alpha=0.9))

    fig.suptitle('Detailed TFM Analysis — SDH Defect Characterization',
                 fontsize=16, fontweight='bold', color='#f0f6fc')
    plt.savefig(OUTPUT_DIR / "04_detailed_analysis.png", dpi=200, bbox_inches='tight')
    print("  ✓ Saved 04_detailed_analysis.png")
    plt.close()

    # ══════════════════════════════════════════════════════════════════════
    # PLOT 5: TOF Table Visualization
    # ══════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    tof_table = gpu_res['tof_table']
    roi = gpu_res['roi']
    n_elem = dataset['n_elements']

    for idx, (elem_i, ax) in enumerate(zip([0, n_elem//2, n_elem-1], axes)):
        tof_2d = tof_table[elem_i].reshape(roi.nz, roi.nx)
        ext_mm = (roi.x_min*1e3, roi.x_max*1e3, roi.z_max*1e3, roi.z_min*1e3)
        im = ax.imshow(
            tof_2d * 1e6,
            extent=ext_mm,
            cmap='viridis',
            aspect='auto',
        )
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('z (mm)')
        ax.set_title(f'TOF — Element {elem_i}', fontweight='bold')
        fig.colorbar(im, ax=ax, label='TOF (μs)', shrink=0.9)

        # Mark element position
        ax.plot(gpu_res['elem_pos'][elem_i, 0]*1e3, 0, 'rv', markersize=10)

    fig.suptitle('Time-of-Flight Tables (acoustic_gpu)',
                 fontsize=16, fontweight='bold', color='#f0f6fc')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "05_tof_tables.png", dpi=200, bbox_inches='tight')
    print("  ✓ Saved 05_tof_tables.png")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "█" * 70)
    print("█  BENCHMARK: Bristol FMC Dataset × acoustic_gpu Pipeline")
    print("█" * 70)

    # 1. Load dataset
    dataset = load_bristol_dataset(DATASET_PATH)

    # 2. Filter
    filtered_data = bandpass_filter(dataset['time_data'], dataset['fs'])

    # Inject filtered data into FMC 3D for acoustic_gpu
    fmc_filtered_3d = np.zeros_like(dataset['fmc_3d'])
    for i in range(len(dataset['tx'])):
        fmc_filtered_3d[dataset['tx'][i], dataset['rx'][i], :] = filtered_data[:, i]
    dataset['fmc_3d'] = fmc_filtered_3d

    # 3. Define ROI
    roi_params = {
        'x_min': -25e-3, 'x_max': 25e-3,
        'z_min': 0.0, 'z_max': 50e-3,
        'nx': 100, 'nz': 120,
    }

    # 4. Run Bristol reference TFM
    bristol_res = run_bristol_tfm(dataset, filtered_data, roi_params)

    # 5. Run acoustic_gpu TFM
    gpu_res = run_acoustic_gpu_tfm(dataset, roi_params)

    # 6. Resolution scaling benchmark
    resolution_results = run_resolution_benchmark(dataset)

    # 7. Generate plots
    generate_plots(dataset, bristol_res, gpu_res, resolution_results, filtered_data)

    # 8. Final summary
    print("\n" + "=" * 70)
    print("  FINAL SUMMARY")
    print("=" * 70)
    print(f"\n  Bristol Reference:")
    print(f"    Peak: ({bristol_res['peak_x']:.2f}, {bristol_res['peak_z']:.2f}) mm")
    print(f"    Time: {bristol_res['time']:.2f}s")
    print(f"\n  acoustic_gpu Pipeline:")
    print(f"    Peak: ({gpu_res['peak_x']:.2f}, {gpu_res['peak_z']:.2f}) mm")
    print(f"    Time: {gpu_res['time_total']:.2f}s (TOF: {gpu_res['time_tof']:.2f}s + TFM: {gpu_res['time_tfm']:.2f}s)")
    print(f"\n  Ground Truth SDH depth: {DEFECT_Z_TRUE} mm")
    print(f"  Bristol z error: {abs(bristol_res['peak_z'] - DEFECT_Z_TRUE):.2f} mm")
    print(f"  GPU z error:     {abs(gpu_res['peak_z'] - DEFECT_Z_TRUE):.2f} mm")

    peak_dist = np.sqrt(
        (bristol_res['peak_x'] - gpu_res['peak_x'])**2 +
        (bristol_res['peak_z'] - gpu_res['peak_z'])**2
    )
    print(f"\n  Peak difference (Bristol vs GPU): {peak_dist:.2f} mm")
    print(f"\n  Plots saved to: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
