"""
TFM (Total Focusing Method) image reconstruction on GPU.

Implements the TFM algorithm using Taichi lang for massively parallel
pixel-wise reconstruction. The TFM image intensity at each pixel P is:

    I(P) = Σ_i Σ_j s_ij[ t_i,P + t_j,P ]

where s_ij is the FMC signal for transmitter i and receiver j, and
t_i,P, t_j,P are the travel times from element i/j to pixel P
(computed by the Surface module considering refraction).
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

try:
    import taichi as ti
    _TAICHI_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TAICHI_AVAILABLE = False
    ti = None  # type: ignore[assignment]

from ..raytracing.kernels import ensure_initialized


@dataclass
class TFMResult:
    """Result of TFM reconstruction.

    Attributes
    ----------
    image    : np.ndarray, shape (nz, nx)
               Reconstructed image intensity.
    envelope : np.ndarray, shape (nz, nx)
               Rectivelope (absolute value) of the image.
    image_db : np.ndarray, shape (nz, nx)
               image in dB scale (normalized to max).
    extent   : tuple
               (x_min, x_max, z_max, z_min) for matplotlib imshow.
    """

    image:    np.ndarray
    envelope: np.ndarray
    image_db: np.ndarray
    extent:   tuple


# ---------------------------------------------------------------------------
# Taichi GPU kernel (only compiled when Taichi is available)
# ---------------------------------------------------------------------------

if _TAICHI_AVAILABLE:

    @ti.kernel
    def _tfm_kernel(
        fmc:        ti.types.ndarray(),
        tof_tx:     ti.types.ndarray(),
        tof_rx:     ti.types.ndarray(),
        image:      ti.types.ndarray(),
        fs:         float,
        n_elements: int,
        n_samples:  int,
        n_pixels:   int,
    ):
        """GPU kernel for TFM reconstruction.

        Each thread processes one pixel. For that pixel, it sums contributions
        from all (tx, rx) pairs using the pre-computed TOF tables.

        Parameters
        ----------
        fmc : ndarray[n_elements * n_elements * n_samples]
            Flattened FMC data: fmc[tx * n_elements * n_samples + rx * n_samples + sample].
        tof_tx : ndarray[n_elements * n_pixels]
            TOF table for transmitters: tof_tx[elem * n_pixels + pixel_idx].
        tof_rx : ndarray[n_elements * n_pixels]
            TOF table for receivers (can be same as tof_tx for symmetric case).
        image : ndarray[n_pixels]
            Output image (flat array).
        fs : float
            Sampling frequency.
        n_elements, n_samples, n_pixels : int
            Dimensions.
        """
        for p in range(n_pixels):
            pixel_val = 0.0

            for tx in range(n_elements):
                t_tx = tof_tx[tx * n_pixels + p]

                for rx in range(n_elements):
                    t_rx = tof_rx[rx * n_pixels + p]

                    # Total round-trip time → sample index
                    t_total  = t_tx + t_rx
                    sample_f = t_total * fs

                    # Linear interpolation in the signal
                    sample_idx = int(sample_f)
                    frac       = sample_f - float(sample_idx)

                    if sample_idx >= 0 and sample_idx < n_samples - 1:
                        fmc_idx_0 = tx * n_elements * n_samples + rx * n_samples + sample_idx
                        fmc_idx_1 = fmc_idx_0 + 1
                        val = fmc[fmc_idx_0] * (1.0 - frac) + fmc[fmc_idx_1] * frac
                        pixel_val += val

            image[p] = pixel_val


# ---------------------------------------------------------------------------
# Shared dB helper
# ---------------------------------------------------------------------------

def _to_db(envelope: np.ndarray) -> np.ndarray:
    """Convert envelope to dB (normalized to peak = 0 dB)."""
    max_val = np.max(envelope)
    if max_val > 0:
        return 20.0 * np.log10(envelope / max_val + 1e-30)
    return np.zeros_like(envelope)


# ---------------------------------------------------------------------------
# GPU reconstruction (Taichi kernel)
# ---------------------------------------------------------------------------

def tfm_reconstruct(
    fmc_data:  np.ndarray,
    tof_table: np.ndarray,
    fs:        float,
    nx:        int,
    nz:        int,
    extent:    tuple,
    tof_rx:    Optional[np.ndarray] = None,
    arch:      Optional[str]        = None,
) -> TFMResult:
    """Reconstruct a TFM image from FMC data and travel-time tables.

    Parameters
    ----------
    fmc_data : np.ndarray, shape (N_elements, N_elements, N_samples)
        Full Matrix Capture data.
    tof_table : np.ndarray, shape (N_elements, N_pixels)
        Travel-time table from each element to each ROI pixel.
        N_pixels = nx * nz.
    fs : float
        Sampling frequency in Hz.
    nx, nz : int
        Image dimensions (pixels).
    extent : tuple
        (x_min, x_max, z_max, z_min) for visualization.
    tof_rx : np.ndarray, optional
        Separate TOF table for receivers. If None, uses tof_table
        (symmetric assumption).
    arch : str, optional
        Taichi architecture override.

    Returns
    -------
    TFMResult
        Reconstructed image with envelope and dB representations.
    """
    ensure_initialized(arch)

    n_elements = fmc_data.shape[0]
    n_samples  = fmc_data.shape[2]
    n_pixels   = nx * nz

    # Flatten arrays for GPU (contiguous float64)
    fmc_flat    = np.ascontiguousarray(fmc_data.ravel(),   dtype=np.float64)
    tof_tx_flat = np.ascontiguousarray(tof_table.ravel(),  dtype=np.float64)

    if tof_rx is not None:
        tof_rx_flat = np.ascontiguousarray(tof_rx.ravel(), dtype=np.float64)
    else:
        tof_rx_flat = tof_tx_flat.copy()

    image_flat = np.zeros(n_pixels, dtype=np.float64)

    # Launch TFM kernel
    _tfm_kernel(
        fmc_flat, tof_tx_flat, tof_rx_flat, image_flat,
        fs, n_elements, n_samples, n_pixels,
    )

    # Reshape to 2D image
    image    = image_flat.reshape(nz, nx)
    envelope = np.abs(image)
    image_db = _to_db(envelope)

    return TFMResult(
        image=image,
        envelope=envelope,
        image_db=image_db,
        extent=extent,
    )


# ---------------------------------------------------------------------------
# CPU reference reconstruction (pure NumPy)
# ---------------------------------------------------------------------------

def tfm_reconstruct_cpu(
    fmc_data:  np.ndarray,
    tof_table: np.ndarray,
    fs:        float,
    nx:        int,
    nz:        int,
    extent:    tuple,
    tof_rx:    Optional[np.ndarray] = None,
) -> TFMResult:
    """CPU reference implementation of TFM reconstruction.

    Same interface as tfm_reconstruct but runs on a CPU using NumPy.
    Useful for validation and debugging.
    """
    n_elements = fmc_data.shape[0]
    n_samples  = fmc_data.shape[2]
    n_pixels   = nx * nz

    if tof_rx is None:
        tof_rx = tof_table

    image_flat = np.zeros(n_pixels, dtype=np.float64)

    for p in range(n_pixels):
        pixel_val = 0.0
        for tx in range(n_elements):
            t_tx = tof_table[tx, p]
            for rx in range(n_elements):
                t_rx       = tof_rx[rx, p]
                t_total    = t_tx + t_rx
                sample_f   = t_total * fs
                sample_idx = int(sample_f)
                frac       = sample_f - float(sample_idx)
                if 0 <= sample_idx < n_samples - 1:
                    val = (
                        fmc_data[tx, rx, sample_idx]       * (1.0 - frac)
                        + fmc_data[tx, rx, sample_idx + 1] * frac
                    )
                    pixel_val += val
        image_flat[p] = pixel_val

    image    = image_flat.reshape(nz, nx)
    envelope = np.abs(image)
    image_db = _to_db(envelope)

    return TFMResult(
        image=image,
        envelope=envelope,
        image_db=image_db,
        extent=extent,
    )
