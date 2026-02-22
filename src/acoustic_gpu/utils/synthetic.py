"""
Synthetic FMC data generation for validation and testing.

Generates simulated Full Matrix Capture (FMC) ultrasonic data by
computing ray-tracing through a surface interface and placing synthetic
reflector pulses at the computed arrival times.

This module is used to validate the ray tracing and TFM algorithms
without requiring real experimental data.
"""

from __future__ import annotations

import numpy as np
from typing import Optional
from dataclasses import dataclass, field

from acoustic_gpu.config import Transducer, Material, ROI, SimulationConfig
from acoustic_gpu.surfaces.base import SurfaceInterface
from acoustic_gpu.raytracing.cpu_ref import find_refraction_point


@dataclass
class Defect:
    """Point reflector (defect) definition.

    Parameters
    ----------
    x : float
        x-coordinate in meters.
    z : float
        z-coordinate in meters (depth below surface).
    amplitude : float
        Reflectivity amplitude (0 to 1).
    name : str
        Optional label.
    """

    x: float
    z: float
    amplitude: float = 1.0
    name: str = ""


def _toneburst(
    t: np.ndarray,
    t_center: float,
    frequency: float,
    n_cycles: int = 5,
    amplitude: float = 1.0,
) -> np.ndarray:
    """Generate a Gaussian-windowed toneburst pulse.

    Parameters
    ----------
    t : np.ndarray
        Time axis.
    t_center : float
        Center time of the pulse.
    frequency : float
        Central frequency in Hz.
    n_cycles : int
        Number of cycles in the burst.
    amplitude : float
        Peak amplitude.

    Returns
    -------
    pulse : np.ndarray
        Toneburst signal.
    """
    # Gaussian envelope width
    sigma = n_cycles / (2.0 * frequency * 2.355)  # A FWHM relation

    envelope = np.exp(-0.5 * ((t - t_center) / sigma) ** 2)
    carrier = np.sin(2.0 * np.pi * frequency * (t - t_center))

    return amplitude * envelope * carrier


def generate_fmc_data(
    transducer: Transducer,
    surface: SurfaceInterface,
    defects: list[Defect],
    c1: float,
    c2: float,
    fs: float,
    n_samples: int,
    n_surface_points: int = 500,
    snr_db: float = 40.0,
    include_surface_echo: bool = True,
    n_cycles: int = 5,
) -> np.ndarray:
    """Generate synthetic FMC data with point reflectors.

    For each (tx, rx) pair:
    1. Compute surface echo TOF (water path only) for tx and rx
    2. For each defect, compute refracted TOF through the surface
    3. Place toneburst pulses at the computed arrival times
    4. Add Gaussian noise

    Parameters
    ----------
    transducer : Transducer
        Transducer configuration.
    surface : SurfaceInterface
        Surface geometry.
    defects : list[Defect]
        Point reflector definitions.
    c1, c2 : float
        Wave speeds in media 1 (coupling) and 2 (specimen).
    fs : float
        Sampling frequency in Hz.
    n_samples : int
        Number of time samples per A-scan.
    n_surface_points : int
        Number of discrete surface points for ray-tracing.
    snr_db : float
        Signal-to-noise ratio in dB.
    include_surface_echo : bool
        Whether to include the interface echo in the signal.
    n_cycles : int
        Number of cycles in the toneburst pulse.

    Returns
    -------
    fmc_data : np.ndarray, shape (N_tx, N_rx, N_samples)
        Simulated FMC data.
    """
    n_elements = transducer.n_elements
    elem_pos = transducer.element_positions()
    freq = transducer.frequency

    # Discretise surface
    xS, zS = surface.get_points(n_surface_points)

    # Time axis
    t = np.arange(n_samples) / fs

    # Output array
    fmc_data = np.zeros((n_elements, n_elements, n_samples), dtype=np.float64)

    for tx in range(n_elements):
        x_tx, z_tx = elem_pos[tx]

        for rx in range(n_elements):
            x_rx, z_rx = elem_pos[rx]
            signal = np.zeros(n_samples, dtype=np.float64)

            # Surface echo (pulse-echo in water)
            if include_surface_echo:
                # Approximate: nearest surface point to each element
                z_surf    = surface.evaluate(x_tx)
                z_surf_rx = surface.evaluate(x_rx)
                d_tx_surf = np.sqrt((x_tx - x_tx) ** 2 + (z_surf - z_tx) ** 2)
                d_rx_surf = np.sqrt((x_rx - x_rx) ** 2 + (z_surf_rx - z_rx) ** 2)
                t_surface = (d_tx_surf + d_rx_surf) / c1
                signal += _toneburst(t, t_surface, freq, n_cycles, amplitude=0.5)

            # Defect echoes
            for defect in defects:
                # Forward path: tx -> surface -> defect
                result_tx = find_refraction_point(
                    x_tx, z_tx,
                    defect.x, defect.z,
                    xS, zS, c1, c2,
                )
                tof_tx = result_tx.tof

                # Return path: defect -> surface -> rx
                result_rx = find_refraction_point(
                    x_rx, z_rx,
                    defect.x, defect.z,
                    xS, zS, c1, c2,
                )
                tof_rx = result_rx.tof

                t_defect = tof_tx + tof_rx
                signal += _toneburst(
                    t, t_defect, freq, n_cycles,
                    amplitude=defect.amplitude,
                )

            fmc_data[tx, rx, :] = signal

    # Add noise
    if snr_db < 100:
        signal_power = np.mean(fmc_data ** 2)
        noise_power = signal_power / (10.0 ** (snr_db / 10.0))
        noise = np.random.normal(0, np.sqrt(noise_power), fmc_data.shape)
        fmc_data += noise

    return fmc_data


def generate_simple_fmc(
    n_elements: int = 64,
    pitch: float = 0.6e-3,
    frequency: float = 5.0e6,
    z_surface: float = 0.020,
    defect_x: float = 0.0,
    defect_z: float = 0.030,
    c1: float = 1480.0,
    c2: float = 5800.0,
    fs: float = 100e6,
    n_samples: int = 4096,
    snr_db: float = 40.0,
) -> tuple[np.ndarray, Transducer, SurfaceInterface, list[Defect], SimulationConfig]:
    """Generate a simple FMC dataset with one defect below a flat surface.

    Convenience function for quick testing and examples.

    Returns
    -------
    fmc_data : np.ndarray
    transducer : Transducer
    surface : FlatSurface
    defects : list[Defect]
    config : SimulationConfig
    """
    from acoustic_gpu.surfaces.flat import FlatSurface

    transducer = Transducer(
        n_elements=n_elements,
        pitch=pitch,
        frequency=frequency,
    )

    surface = FlatSurface(z_offset=z_surface)

    defects = [Defect(x=defect_x, z=defect_z, amplitude=1.0, name="SDH")]

    water = Material(name="Water", c_longitudinal=c1)
    steel = Material(name="Steel", c_longitudinal=c2, c_transversal=3240.0)

    roi = ROI(
        x_min=-0.015, x_max=0.015,
        z_min=z_surface + 0.001, z_max=defect_z + 0.015,
        nx=128, nz=128,
    )

    config = SimulationConfig(
        medium1=water,
        medium2=steel,
        transducer=transducer,
        roi=roi,
        fs=fs,
        n_samples=n_samples,
    )

    fmc_data = generate_fmc_data(
        transducer=transducer,
        surface=surface,
        defects=defects,
        c1=c1,
        c2=c2,
        fs=fs,
        n_samples=n_samples,
        snr_db=snr_db,
    )

    return fmc_data, transducer, surface, defects, config
