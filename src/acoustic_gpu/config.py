"""
Configuration module for acoustic GPU computations.

Defines dataclasses for materials, transducers, and regions of interest (ROI)
used throughout the acoustic ray-tracing and imaging pipeline.

All units are SI (meters, seconds, m/s, kg/m³) unless noted otherwise.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Material:
    """IC material properties.

    Parameters
    ----------
    name : str
        Human-readable material name.
    c_longitudinal : float
        Longitudinal wave speed in m/s.
    c_transversal : float, optional
        Transversal (shear) wave speed in m/s. None for fluids.
    density : float, optional
        Material density in kg/m³.
    impedance : float, optional
        Acoustic impedance in 10⁶ kg/(m²·s). Computed from density and
        longitudinal speed if not provided.
    """

    name: str
    c_longitudinal: float
    c_transversal: Optional[float] = None
    density: Optional[float] = None
    impedance: Optional[float] = None

    def __post_init__(self) -> None:
        if self.impedance is None and self.density is not None:
            self.impedance = self.density * self.c_longitudinal

    @property
    def is_fluid(self) -> bool:
        """Whether this material is a fluid (no shear waves)."""
        return self.c_transversal is None


# — Pre-defined materials ————————————————————————————————————————————————————
WATER = Material(
    name="Water (20°C)",
    c_longitudinal=1480.0,
    density=998.0,
    impedance=1.48e6,
)

STEEL_3020 = Material(
    name="Steel 3020",
    c_longitudinal=5940.0,
    c_transversal=3240.0,
    density=7800.0,
    impedance=45.0e6,
)

ALUMINUM = Material(
    name="Aluminum",
    c_longitudinal=6320.0,
    c_transversal=3080.0,
    density=2700.0,
    impedance=17.0e6,
)

TITANIUM = Material(
    name="Titanium",
    c_longitudinal=6070.0,
    c_transversal=3110.0,
    density=4507.0,
    impedance=27.0e6,
)

LUCITE = Material(
    name="Lucite",
    c_longitudinal=2710.0,
    c_transversal=1490.0,
    density=1180.0,
    impedance=3.2e6,
)


@dataclass
class Transducer:
    """Linear phased-array transducer parameters.

    Parameters
    ----------
    n_elements : int
        Number of elements in the array.
    pitch : float
        Center-to-center element spacing in meters.
    frequency : float
        Central frequency in Hz.
    element_width : float, optional
        Active element width in meters. Defaults to 90% of pitch.
    elevation : float, optional
        Elevation aperture in meters.
    position : tuple[float, float], optional
        (x, z) position of array center in meters. Default is (0, 0).
    """

    n_elements: int
    pitch: float
    frequency: float
    element_width: Optional[float] = None
    elevation: Optional[float] = None
    position: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self) -> None:
        if self.element_width is None:
            self.element_width = 0.9 * self.pitch

    @property
    def aperture(self) -> float:
        """Total array aperture in meters."""
        return (self.n_elements - 1) * self.pitch

    @property
    def wavelength(self, medium: Material = WATER) -> float:
        """Wavelength in the coupling medium."""
        return medium.c_longitudinal / self.frequency

    def element_positions(self) -> np.ndarray:
        """Compute (x, z) positions of each element.

        Returns
        -------
        positions : np.ndarray, shape (n_elements, 2)
            Array of (x, z) coordinates for each element center.
        """
        x_center, z_center = self.position
        half_aperture = self.aperture / 2.0
        x_coords = np.linspace(
            x_center - half_aperture,
            x_center + half_aperture,
            self.n_elements,
        )
        z_coords = np.full(self.n_elements, z_center)
        return np.column_stack([x_coords, z_coords])


@dataclass
class ROI:
    """Region of interest for image reconstruction.

    Defines a 2D rectangular grid of pixels in the (x, z) plane.

    Parameters
    ----------
    x_min, x_max : float
        Horizontal extent in meters.
    z_min, z_max : float
        Vertical extent in meters (depth direction).
    nx, nz : int
        Number of pixels along each axis.
    """

    x_min: float
    x_max: float
    z_min: float
    z_max: float
    nx: int
    nz: int

    @property
    def n_pixels(self) -> int:
        return self.nx * self.nz

    @property
    def dx(self) -> float:
        """Pixel spacing in x direction."""
        return (self.x_max - self.x_min) / max(self.nx - 1, 1)

    @property
    def dz(self) -> float:
        """Pixel spacing in z direction."""
        return (self.z_max - self.z_min) / max(self.nz - 1, 1)

    def grid(self) -> tuple[np.ndarray, np.ndarray]:
        """Create meshgrid of pixel coordinates.

        Returns
        -------
        X, Z : np.ndarray
            2D arrays of shape (nz, nx) with x and z coordinates.
        """
        x = np.linspace(self.x_min, self.x_max, self.nx)
        z = np.linspace(self.z_min, self.z_max, self.nz)
        return np.meshgrid(x, z)

    def pixel_coordinates(self) -> np.ndarray:
        """Flat array of all pixel (x, z) coordinates.

        Returns
        -------
        coords : np.ndarray, shape (n_pixels, 2)
            Each row is (x, z) of a pixel.
        """
        X, Z = self.grid()
        return np.column_stack([X.ravel(), Z.ravel()])

    @property
    def extent(self) -> tuple[float, float, float, float]:
        """Extent tuple for matplotlib imshow: (x_min, x_max, z_max, z_min)."""
        return (self.x_min, self.x_max, self.z_max, self.z_min)


@dataclass
class SimulationConfig:
    """Complete simulation configuration.

    Parameters
    ----------
    medium1 : Material
        Coupling medium (e.g., water).
    medium2 : Material
        Test specimen material (e.g., steel).
    transducer : Transducer
        Phased-array transducer parameters.
    roi : ROI
        Region of interest for imaging.
    fs : float
        Sampling frequency in Hz.
    n_samples : int
        Number of time samples per A-scan.
    wave_type : str
        Wave type in medium2: 'longitudinal' or 'transversal'.
    """

    medium1: Material
    medium2: Material
    transducer: Transducer
    roi: ROI
    fs: float = 100e6
    n_samples: int = 4096
    wave_type: str = "longitudinal"

    @property
    def c1(self) -> float:
        """Speed in medium 1 (coupling)."""
        return self.medium1.c_longitudinal

    @property
    def c2(self) -> float:
        """Speed in medium 2 (specimen)."""
        if self.wave_type == "transversal" and self.medium2.c_transversal is not None:
            return self.medium2.c_transversal
        return self.medium2.c_longitudinal

    @property
    def critical_angle(self) -> float:
        """First critical angle in radians. Beyond this, total reflection occurs."""
        ratio = self.c1 / self.c2
        if ratio >= 1.0:
            return np.pi / 2.0  # No critical angle (c1 >= c2)
        return np.arcsin(ratio)
