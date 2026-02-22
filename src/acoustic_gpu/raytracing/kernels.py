"""
Taichi GPU kernels for acoustic ray-tracing via refraction.

Implements the Newton-Raphson refraction point search algorithm
(Parrilla 2007) using Taichi lang for massively parallel execution
on GPU. Falls back to bisection when Newton-Raphson diverges.

Architecture
- @ti.func helpers: _lin_interp, _compute_slope, _compute_tof, _compute_V
- @ti.kernel find_refraction_kernel: 2D parallel loop over (emitter, focus) pairs
- Python wrappers: find_refraction_points_gpu() — converts NumPy arrays, launches kernel

All intermediate TOF computations use float64 for precision.
"""

import numpy as np
from typing import Optional

try:
    import taichi as ti
    _TAICHI_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TAICHI_AVAILABLE = False
    ti = None  # type: ignore[assignment]

# — Taichi initialization ————————————————————————————————————————————————————
# Lazily initialized; call ensure_initialized() before any kernel use.
_ti_initialized = False


def ensure_initialized(arch: Optional[str] = None) -> None:
    """Initialize Taichi runtime if not already done.

    Parameters
    ----------
    arch : str, optional
        Architecture: 'gpu', 'cuda', 'vulkan', 'cpu'.
        Default: try GPU, fall back to CPU.
    """
    global _ti_initialized
    if _ti_initialized:
        return
    if not _TAICHI_AVAILABLE:
        return  # CPU-only mode; kernels will not be available

    if arch is None or arch == "gpu":
        ti.init(arch=ti.gpu, default_fp=ti.f64)
    elif arch == "cuda":
        ti.init(arch=ti.cuda, default_fp=ti.f64)
    elif arch == "vulkan":
        ti.init(arch=ti.vulkan, default_fp=ti.f64)
    elif arch == "cpu":
        ti.init(arch=ti.cpu, default_fp=ti.f64)
    else:
        ti.init(arch=ti.gpu, default_fp=ti.f64)

    _ti_initialized = True


# ——————————————————————————————————————————————————————————————————————————————

NEWTON_MAXITER = 10
BISECT_MAXITER = 30
TOL    = 1e-4
DELTA  = 0.1   # Index perturbation for slope computation
INF_TOF = 1e10  # Sentinel for total internal reflection

# ——————————————————————————————————————————————————————————————————————————————
# Taichi device functions (@ti.func)
# ——————————————————————————————————————————————————————————————————————————————

if _TAICHI_AVAILABLE:

    @ti.func
    def _lin_interp_x(k, xS: ti.template(), n):
        """Linearly interpolate x coordinate at fractional index k."""
        k_clamped = ti.max(0.0, ti.min(k, ti.cast(n - 1, ti.f64) - 0.001))
        k_floor   = ti.cast(k_clamped, ti.i32)
        k_floor   = ti.min(k_floor, n - 2)
        frac      = k_clamped - ti.cast(k_floor, ti.f64)
        return xS[k_floor] + frac * (xS[k_floor + 1] - xS[k_floor])

    @ti.func
    def _lin_interp_z(k, zS: ti.template(), n):
        """Linearly interpolate z coordinate at fractional index k."""
        k_clamped = ti.max(0.0, ti.min(k, ti.cast(n - 1, ti.f64) - 0.001))
        k_floor   = ti.cast(k_clamped, ti.i32)
        k_floor   = ti.min(k_floor, n - 2)
        frac      = k_clamped - ti.cast(k_floor, ti.f64)
        return zS[k_floor] + frac * (zS[k_floor + 1] - zS[k_floor])

    @ti.func
    def _compute_slope(k, xS: ti.template(), zS: ti.template(), n, delta):
        """Compute surface slope dz/dx at index k via finite differences."""
        x_k  = _lin_interp_x(k,         xS, n)
        x_kd = _lin_interp_x(k + delta, xS, n)
        z_kd = _lin_interp_z(k + delta, zS, n)
        dk     = x_kd - x_k
        result = 0.0
        if ti.abs(dk) > 1e-30:
            result = (z_kd - _lin_interp_z(k, zS, n)) / dk
        return result

    @ti.func
    def _compute_tof(k, x_a, z_a, x_o, z_o, x_f, z_f, c1, c2):
        """Compute total time of flight: t = d1/c1 + d2/c2."""
        d1 = ti.sqrt((x_a - x_o) ** 2 + (z_a - z_o) ** 2)
        d2 = ti.sqrt((x_f - x_o) ** 2 + (z_f - z_o) ** 2)
        return d1 / c1 + d2 / c2

    @ti.func
    def _compute_V(k, xS: ti.template(), zS: ti.template(), n,
                   x_a, z_a, x_f, z_f, c1, c2, delta):
        """Compute TOF derivative V_k = dt/dk (Parrilla formulation)."""
        x_k = _lin_interp_x(k, xS, n)
        z_k = _lin_interp_z(k, zS, n)
        Mk  = _compute_slope(k, xS, zS, n, delta)

        d1  = ti.sqrt((x_a - x_k) ** 2 + (z_a - z_k) ** 2)
        d2  = ti.sqrt((x_f - x_k) ** 2 + (z_f - z_k) ** 2)

        V_k = 0.0
        if d1 > 1e-30 and d2 > 1e-30:
            V_k = (
                ((x_k - x_a) + Mk * (z_k - z_a)) / (c1 * d1)
                + ((x_k - x_f) + Mk * (z_k - z_f)) / (c2 * d2)
            )
        return V_k

    @ti.func
    def _find_refraction_single(x_a, z_a, x_f, z_f,
                                xS: ti.template(), zS: ti.template(),
                                n_surf, c1, c2, k_init, tol, delta):
        """Find refraction point for a single (emitter, focus) pair.

        Returns
        -------
        vec2 : (k_index, tof)
            k_index: fractional surface index of refraction point
            tof: total time-of-flight
        """
        k_min = 0.0
        k_max = ti.cast(n_surf - 2, ti.f64)

        # — Newton-Raphson phase
        kappa     = ti.max(k_min, ti.min(k_init, k_max))
        converged = False

        for _ in range(NEWTON_MAXITER):
            V_k0 = _compute_V(kappa,       xS, zS, n_surf, x_a, z_a, x_f, z_f, c1, c2, delta)
            V_k1 = _compute_V(kappa + 1.0, xS, zS, n_surf, x_a, z_a, x_f, z_f, c1, c2, delta)
            dV   = V_k1 - V_k0
            if ti.abs(dV) < 1e-30:
                break

            kappa_new = kappa - V_k0 / dV
            kappa_new = ti.max(k_min, ti.min(kappa_new, k_max))

            if ti.abs(kappa_new - kappa) <= tol:
                converged = True
                break

            kappa = kappa_new

        # — Bisection fallback
        if not converged:
            a   = k_min
            b   = k_max
            V_a = _compute_V(a, xS, zS, n_surf, x_a, z_a, x_f, z_f, c1, c2, delta)
            V_b = _compute_V(b, xS, zS, n_surf, x_a, z_a, x_f, z_f, c1, c2, delta)

            if V_a * V_b <= 0:
                for _ in range(BISECT_MAXITER):
                    mid   = (a + b) / 2.0
                    V_mid = _compute_V(mid, xS, zS, n_surf, x_a, z_a, x_f, z_f, c1, c2, delta)

                    if ti.abs(V_mid) < 1e-30 or (b - a) / 2.0 < tol:
                        break

                    if V_a * V_mid < 0:
                        b = mid
                    else:
                        a   = mid
                        V_a = V_mid

                kappa = (a + b) / 2.0

        # — Compute final TOF
        x_s = _lin_interp_x(kappa, xS, n_surf)
        z_s = _lin_interp_z(kappa, zS, n_surf)
        tof = _compute_tof(kappa, x_a, z_a, x_s, z_s, x_f, z_f, c1, c2)

        return ti.math.vec2(kappa, tof)

    # ——————————————————————————————————————————————————————————————————————————
    # Main GPU kernel
    # ——————————————————————————————————————————————————————————————————————————

    @ti.kernel
    def find_refraction_kernel(
        emitters_x: ti.types.ndarray(),
        emitters_z: ti.types.ndarray(),
        focuses_x:  ti.types.ndarray(),
        focuses_z:  ti.types.ndarray(),
        xS:         ti.types.ndarray(),
        zS:         ti.types.ndarray(),
        n_surf:     int,
        c1:         float,
        c2:         float,
        tol:        float,
        delta:      float,
        k_result:   ti.types.ndarray(),
        tof_result: ti.types.ndarray(),
        k_init_arr: ti.types.ndarray(),
        Ne:         int,
        Nf:         int,
    ):
        """GPU kernel: find refraction points for all (emitter, focus) pairs.

        Each thread handles one (emitter_idx, focus_idx) pair independently.
        The outer for loop is automatically parallelized by Taichi.

        Parameters (all as flat arrays for GPU compatibility)
        ----------
        emitters_x, emitters_z : ndarray[Ne]
            Emitter positions.
        focuses_x, focuses_z : ndarray[Nf]
            Focus positions.
        xS, zS : ndarray[N_surf]
            Discrete surface points.
        n_surf : int
            Number of surface points.
        c1, c2 : float
            Wave speeds.
        tol, delta : float
            Convergence tolerance and slope perturbation.
        k_result : ndarray[Ne * Nf]
            Output: fractional surface indices (flat 2D → 1D).
        tof_result : ndarray[Ne * Nf]
            Output: time-of-flight values (flat 2D → 1D).
        k_init_arr : ndarray[Ne * Nf]
            Initial guesses for each pair (for trajectory tracking).
        Ne, Nf : int
            Dimensions.
        """
        for idx in range(Ne * Nf):
            i = idx // Nf  # emitter index
            j = idx  % Nf  # focus index

            x_a = emitters_x[i]
            z_a = emitters_z[i]
            x_f = focuses_x[j]
            z_f = focuses_z[j]

            k_init = k_init_arr[idx]

            result = _find_refraction_single(
                x_a, z_a, x_f, z_f,
                xS, zS, n_surf,
                c1, c2, k_init,
                tol, delta,
            )

            k_result[idx]   = result[0]
            tof_result[idx] = result[1]

    # ——————————————————————————————————————————————————————————————————————————
    # Coarse search kernel (for initial guesses)
    # ——————————————————————————————————————————————————————————————————————————

    @ti.kernel
    def coarse_search_kernel(
        emitters_x: ti.types.ndarray(),
        emitters_z: ti.types.ndarray(),
        focuses_x:  ti.types.ndarray(),
        focuses_z:  ti.types.ndarray(),
        xS:         ti.types.ndarray(),
        zS:         ti.types.ndarray(),
        n_surf:     int,
        c1:         float,
        c2:         float,
        k_init_arr: ti.types.ndarray(),
        Ne:         int,
        Nf:         int,
        n_coarse:   int,
    ):
        """Find initial k estimates via coarse TOF sampling on GPU.

        For each (emitter, focus) pair, samples n_coarse surface points
        and picks the one with minimum TOF as the initial guess.
        """
        k_max = float(n_surf - 2)

        for idx in range(Ne * Nf):
            i = idx // Nf
            j = idx  % Nf

            x_a = emitters_x[i]
            z_a = emitters_z[i]
            x_f = focuses_x[j]
            z_f = focuses_z[j]

            best_k   = 0.0
            best_tof = 1e30

            for s in range(n_coarse):
                k   = k_max * float(s) / float(n_coarse - 1)
                x_k = _lin_interp_x(k, xS, n_surf)
                z_k = _lin_interp_z(k, zS, n_surf)
                tof = _compute_tof(k, x_a, z_a, x_k, z_k, x_f, z_f, c1, c2)

                if tof < best_tof:
                    best_tof = tof
                    best_k   = k

            k_init_arr[idx] = best_k


# ——————————————————————————————————————————————————————————————————————————————
# Python wrapper functions
# ——————————————————————————————————————————————————————————————————————————————

def find_refraction_points_gpu(
    emitters:  np.ndarray,
    focuses:   np.ndarray,
    xS:        np.ndarray,
    zS:        np.ndarray,
    c1:        float,
    c2:        float,
    tol:       float           = TOL,
    delta:     float           = DELTA,
    n_coarse:  Optional[int]   = None,
    arch:      Optional[str]   = None,
) -> tuple:
    """Compute refraction points and TOF for all emitter-focus pairs on GPU.

    This is the main entry point for GPU ray-tracing. It:
    1. Initializes Taichi (if needed)
    2. Runs a coarse search kernel for initial guesses
    3. Runs the Newton-Raphson kernel with those guesses

    Parameters
    ----------
    emitters : np.ndarray, shape (Na, 2)
        Emitter positions (x, z) in meters.
    focuses : np.ndarray, shape (Nf, 2)
        Focus positions (x, z) in meters.
    xS, zS : np.ndarray, shape (Ns,)
        Discretized surface points.
    c1, c2 : float
        Wave speeds in media 1 and 2 (m/s).
    tol : float
        Convergence tolerance in index space.
    delta : float
        Index perturbation for slope computation.
    n_coarse : int
        Number of coarse samples for initial guess.
    arch : str, optional
        Taichi architecture override.

    Returns
    -------
    k_result : np.ndarray, shape (Na, Nf)
        Fractional surface indices of refraction points.
    tof_result : np.ndarray, shape (Na, Nf)
        Time-of-flight values in seconds.
    """
    ensure_initialized(arch)

    Na = len(emitters)
    Nf = len(focuses)
    Ns = len(xS)

    # Prepare flat arrays (float64)
    emitters = np.ascontiguousarray(emitters, dtype=np.float64)
    focuses  = np.ascontiguousarray(focuses,  dtype=np.float64)
    xS       = np.ascontiguousarray(xS,       dtype=np.float64)
    zS       = np.ascontiguousarray(zS,       dtype=np.float64)

    emitters_x = emitters[:, 0].copy()
    emitters_z = emitters[:, 1].copy()
    focuses_x  = focuses[:, 0].copy()
    focuses_z  = focuses[:, 1].copy()

    k_result   = np.zeros(Na * Nf, dtype=np.float64)
    tof_result = np.zeros(Na * Nf, dtype=np.float64)
    k_init     = np.zeros(Na * Nf, dtype=np.float64)

    if n_coarse is None:
        n_coarse = max(10, Ns // 10)

    # Step 1: Coarse search for initial guesses
    coarse_search_kernel(
        emitters_x, emitters_z,
        focuses_x,  focuses_z,
        xS, zS, Ns,
        c1, c2,
        k_init,
        Na, Nf, n_coarse,
    )

    # Step 2: Newton-Raphson refinement
    find_refraction_kernel(
        emitters_x, emitters_z,
        focuses_x,  focuses_z,
        xS, zS, Ns,
        c1, c2, tol, delta,
        k_result, tof_result, k_init,
        Na, Nf,
    )

    return k_result.reshape(Na, Nf), tof_result.reshape(Na, Nf)


def compute_tof_table(
    transducer_positions: np.ndarray,
    roi_pixels:           np.ndarray,
    xS:                   np.ndarray,
    zS:                   np.ndarray,
    c1:                   float,
    c2:                   float,
    **kwargs,
) -> np.ndarray:
    """Compute travel-time table for TFM: element + pixel TOF on GPU.

    Parameters
    ----------
    transducer_positions : np.ndarray, shape (N_elem, 2)
        Element positions (x, z).
    roi_pixels : np.ndarray, shape (N_pixels, 2)
        Pixel coordinates (x, z).
    xS, zS : np.ndarray
        Discretized surface points.
    c1, c2 : float
        Wave speeds.
    **kwargs :
        Additional arguments for find_refraction_points_gpu.

    Returns
    -------
    tof_table : np.ndarray, shape (N_elem, N_pixels)
        Time-of-flight from each element to each pixel.
    """
    _, tof_table = find_refraction_points_gpu(
        transducer_positions, roi_pixels, xS, zS, c1, c2, **kwargs
    )
    return tof_table
