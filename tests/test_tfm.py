"""
Tests for TFM (Total Focusing Method) image reconstruction.

Validates that a point defect at a known position appears at the
correct location in the reconstructed TFM image.
"""

import numpy as np
import pytest

from acoustic_gpu.config import Transducer, ROI
from acoustic_gpu.surfaces.flat import FlatSurface
from acoustic_gpu.raytracing.cpu_ref import compute_tof_table_cpu
from acoustic_gpu.imaging.tfm import tfm_reconstruct_cpu, TFMResult
from acoustic_gpu.utils.synthetic import generate_fmc_data, Defect


class TestTFMReconstruction:
    """Test TFM image reconstruction."""

    def setup_method(self):
        """Set up a simple test case: flat surface, one defect."""
        self.n_elements  = 16
        self.pitch       = 0.6e-3
        self.frequency   = 5e6
        self.c1          = 1480.0
        self.c2          = 5800.0
        self.z_surface   = 0.020
        self.fs          = 100e6
        self.n_samples   = 4096

        self.defect_x = 0.0
        self.defect_z = 0.038

        self.transducer = Transducer(
            n_elements=self.n_elements,
            pitch=self.pitch,
            frequency=self.frequency,
        )

        self.surface  = FlatSurface(z_offset=self.z_surface)
        self.defects  = [Defect(x=self.defect_x, z=self.defect_z)]

        self.roi = ROI(
            x_min=-0.015, x_max=0.015,
            z_min=self.z_surface + 0.002, z_max=self.defect_z + 0.010,
            nx=64, nz=64,
        )

    def _generate_and_reconstruct(self) -> TFMResult:
        """Generate synthetic FMC data and perform TFM reconstruction."""
        # Generate synthetic FMC
        fmc = generate_fmc_data(
            transducer=self.transducer,
            surface=self.surface,
            defects=self.defects,
            c1=self.c1,
            c2=self.c2,
            fs=self.fs,
            n_samples=self.n_samples,
            snr_db=40.0,  # high SNR for reliable detection
            include_surface_echo=False,
        )

        # Compute TOF table
        elem_pos = self.transducer.element_positions()
        pixels   = self.roi.pixel_coordinates()
        xS, zS   = self.surface.get_points(500)

        tof_table = compute_tof_table_cpu(
            elem_pos, pixels, xS, zS, self.c1, self.c2
        )

        # TFM reconstruction
        result = tfm_reconstruct_cpu(
            fmc, tof_table, self.fs,
            self.roi.nx, self.roi.nz,
            self.roi.extent,
        )

        return result

    def test_peak_location(self):
        """Peak of TFM image should coincide with defect position."""
        result = self._generate_and_reconstruct()

        peak_idx = np.unravel_index(np.argmax(result.envelope), result.envelope.shape)
        iz_peak, ix_peak = peak_idx

        # Convert pixel indices to coordinates
        x_coords = np.linspace(self.roi.x_min, self.roi.x_max, self.roi.nx)
        z_coords = np.linspace(self.roi.z_min, self.roi.z_max, self.roi.nz)

        x_peak = x_coords[ix_peak]
        z_peak = z_coords[iz_peak]

        # Check position within ±3 pixels (synthetic data has limited resolution)
        dx = self.roi.dx
        dz = self.roi.dz

        np.testing.assert_allclose(
            x_peak, self.defect_x, atol=3 * dx,
            err_msg=f"Peak x: {x_peak * 1e3:.2f}mm, expected {self.defect_x * 1e3:.2f}mm",
        )
        np.testing.assert_allclose(
            z_peak, self.defect_z, atol=3 * dz,
            err_msg=f"Peak z: {z_peak * 1e3:.2f}mm, expected {self.defect_z * 1e3:.2f}mm",
        )

    def test_image_nonzero(self):
        """Reconstructed image should not be all zeros."""
        result = self._generate_and_reconstruct()
        assert np.max(result.envelope) > 0

    def test_db_range(self):
        """`dB image should have max = 0 dB."""
        result = self._generate_and_reconstruct()
        np.testing.assert_allclose(np.max(result.image_db), 0.0, atol=0.1)

    def test_image_shape(self):
        """Output image shape should match ROI dimensions."""
        result = self._generate_and_reconstruct()
        assert result.image.shape    == (self.roi.nz, self.roi.nx)
        assert result.envelope.shape == (self.roi.nz, self.roi.nx)
        assert result.image_db.shape == (self.roi.nz, self.roi.nx)
