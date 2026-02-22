"""
Surface geometry representations for ultrasonic immersion testing.

Provides classes for flat, tubular, and irregular surface geometries,
plus utilities for estimating surface geometry from FMC data.
"""

from acoustic_gpu.surfaces.base import SurfaceInterface
from acoustic_gpu.surfaces.flat import FlatSurface
from acoustic_gpu.surfaces.tubular import TubularSurface
from acoustic_gpu.surfaces.irregular import IrregularSurface

__all__ = [
    "SurfaceInterface",
    "FlatSurface",
    "TubularSurface",
    "IrregularSurface",
]
