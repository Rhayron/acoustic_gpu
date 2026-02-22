"""
CPU reference implementation of refraction point search.

Provides a pure NumPy Newton-Raphson implementation of the Parrilla (2007)
refraction algorithm. This serves as ground truth for validating the GPU
(Taichi) kernels.

Algorithm
---------
1. Discretize surface into N equally spaced points (x_k, z_k).
2. For each (emitter, focus) pair, find fractional index k such that
   V_k = dt/dk = 0 (Fermat's principle + Snell's law).
3. Newton-Raphson: k [i+1] = k_i - V (k_i) / V'(k_i)
4. Bisection fallback when NR diverges.

Reference
---------
Parrilla, M. et al. (2007). "Fast technique for ultrasonic ray-tracing
through interfaces." Ultrasonics, 46(1-8), pp. 455-456.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

# — Constants ————————————————————————————————————————————————————————————————
NEWTON_MAXITER = 10
BISECT_MAXITER = 30
TOL   = 1e-4
DELTA = 0.1


@dataclass
class RayResult:
    """Result of a single ray refraction search.

    Attributes
    ----------
    k_index : float
        Fractional surface index of the refraction point.
    x_s : float
        x-coordinate of the refraction point on the surface.
    z_s : float
        z-coordinate of the refraction point on the surface.
    tof : float
        Total time-of-flight (emitter → surface → focus).
    converged : bool
        Whether the Newton-Raphson iteration converged.
    n_iterations : int
        Number of iterations used.
    """

    k_index:      float
    x_s:          float
    z_s:          float
    tof:          float
    converged:    bool
    n_iterations: int


def _lin_interp(k: float, arr: np.ndarray) -> float:
    """Linearly interpolate array value at fractional index k.

    Parameters
    ----------
    k : float
        Fractional index (0 ≤ k ≤ len(arr)-1).
    arr : np.ndarray
        1D array to interpolate.

    Returns
    -------
    float
        Interpolated value.
    """
    n       = len(arr)
    k       = max(0.0, min(k, n - 1 - 0.001))
    k_floor = int(k)
    k_floor = min(k_floor, n - 2)
    frac    = k - k_floor
    return arr[k_floor] + frac * (arr[k_floor + 1] - arr[k_floor])


def _compute_slope(
    k:     float,
    xS:    np.ndarray,
    zS:    np.ndarray,
    delta: float = DELTA,
) -> float:
    """Compute surface slope M_k at fractional index k.

    Uses finite differences:
        M_k = (z_(k+delta) - z_k) / (x_(k+delta) - x_k)

    Parameters
    ----------
    k : float
        Fractional surface index.
    xS, zS : np.ndarray
        Surface point arrays.
    delta : float
        Index perturbation for finite differences.

    Returns
    -------
    float
        Slope dz/dx at index k.
    """
    x_k  = _lin_interp(k,         xS)
    x_kd = _lin_interp(k + delta, xS)
    z_kd = _lin_interp(k + delta, zS)

    dx = x_kd - x_k
    if abs(dx) < 1e-30:
        return 0.0
    return (z_kd - _lin_interp(k, zS)) / dx


def _compute_tof(
    x_k: float,
    z_k: float,
    x_a: float,
    z_a: float,
    x_f: float,
    z_f: float,
    c1:  float,
    c2:  float,
) -> float:
    """Compute total time-of-flight through refraction point.

    T = d1/c1 + d2/c2

    where d1 = distance(emitter, surface point),
          d2 = distance(surface point, focus).

    Parameters
    ----------
    x_k, z_k : float
        Refraction point on surface.
    x_a, z_a : float
        Emitter position.
    x_f, z_f : float
        Focus position.
    c1, c2 : float
        Wave speeds in medium 1 and 2.

    Returns
    -------
    float
        Total time-of-flight.
    """
    d1 = np.sqrt((x_k - x_a) ** 2 + (z_k - z_a) ** 2)
    d2 = np.sqrt((x_f - x_k) ** 2 + (z_f - z_k) ** 2)
    return d1 / c1 + d2 / c2


def _compute_V(
    k:     float,
    xS:    np.ndarray,
    zS:    np.ndarray,
    x_a:   float,
    z_a:   float,
    x_f:   float,
    z_f:   float,
    c1:    float,
    c2:    float,
    delta: float = DELTA,
) -> float:
    """Compute TOF derivative V_k = dt/dk (Parrilla formulation).

    V_k = (1/c1) * [(x_k - x_a) + M_k*(z_k - z_a)] / d1
        + (1/c2) * [(x_k - x_f) + M_k*(z_k - z_f)] / d2

    At the optimal refraction point V_k = 0 (Fermat's principle),
    which is equivalent to Snell's law.

    Parameters
    ----------
    k : float
        Fractional surface index.
    xS, zS : np.ndarray
        Surface points.
    x_a, z_a : float
        Emitter position.
    x_f, z_f : float
        Focus position.
    c1, c2 : float
        Wave speeds.
    delta : float
        Perturbation for slope computation.

    Returns
    -------
    float
        TOF derivative at index k.
    """
    x_k = _lin_interp(k, xS)
    z_k = _lin_interp(k, zS)
    Mk  = _compute_slope(k, xS, zS, delta)

    d1  = np.sqrt((x_k - x_a) ** 2 + (z_a - z_k) ** 2)
    d2  = np.sqrt((x_f - x_k) ** 2 + (z_f - z_k) ** 2)

    if d1 < 1e-30 or d2 < 1e-30:
        return 0.0

    V_k = (
        (1.0 / c1) * ((x_k - x_a) + Mk * (z_k - z_a)) / d1
        + (1.0 / c2) * ((x_k - x_f) + Mk * (z_k - z_f)) / d2
    )
    return V_k


def find_refraction_point(
    x_a:    float,
    z_a:    float,
    x_f:    float,
    z_f:    float,
    xS:     np.ndarray,
    zS:     np.ndarray,
    c1:     float,
    c2:     float,
    k_init: Optional[float] = None,
    tol:    float            = TOL,
    delta:  float            = DELTA,
) -> RayResult:
    """Find the refraction point for a single (emitter, focus) pair.

    Uses Newton-Raphson in index-space with bisection fallback.

    Parameters
    ----------
    x_a, z_a : float
        Emitter position.
    x_f, z_f : float
        Focus point position.
    xS, zS : np.ndarray, shape (Ns,)
        Discrete surface points.
    c1, c2 : float
        Wave speeds in media 1 (above surface) and 2 (below surface).
    k_init : float, optional
        Initial guess for the surface index. If None, uses coarse search.
    tol : float
        Convergence tolerance in index-space (|Δk| ≤ tol).
    delta : float
        Perturbation for slope finite differences.

    Returns
    -------
    RayResult
        Refraction point, TOF, and convergence info.
    """
    n_surf = len(xS)
    k_min  = 0.0
    k_max  = float(n_surf - 2)

    # — Initial guess via coarse search ——————————————————————————————————————
    if k_init is None:
        n_coarse = min(30, n_surf)
        best_k   = 0.0
        best_tof = 1e30
        for s in range(n_coarse):
            k   = k_max * s / (n_coarse - 1)
            x_k = _lin_interp(k, xS)
            z_k = _lin_interp(k, zS)
            tof = _compute_tof(x_k, z_k, x_a, z_a, x_f, z_f, c1, c2)
            if tof < best_tof:
                best_tof = tof
                best_k   = k
        k_init = best_k

    # — Newton-Raphson phase ——————————————————————————————————————————————————
    kappa     = max(k_min, min(k_init, k_max))
    converged = False
    n_iter    = 0

    for i in range(NEWTON_MAXITER):
        n_iter += 1
        V_k0 = _compute_V(kappa,       xS, zS, x_a, z_a, x_f, z_f, c1, c2, delta)
        V_k1 = _compute_V(kappa + 1.0, xS, zS, x_a, z_a, x_f, z_f, c1, c2, delta)

        dV = V_k1 - V_k0
        if abs(dV) < 1e-30:
            break

        kappa_new = kappa - V_k0 / dV
        kappa_new = max(k_min, min(kappa_new, k_max))

        if abs(kappa_new - kappa) <= tol:
            converged = True
            kappa      = kappa_new
            break

        kappa = kappa_new

    # — Bisection fallback ————————————————————————————————————————————————————
    if not converged:
        a   = k_min
        b   = k_max
        V_a = _compute_V(a, xS, zS, x_a, z_a, x_f, z_f, c1, c2, delta)
        V_b = _compute_V(b, xS, zS, x_a, z_a, x_f, z_f, c1, c2, delta)

        if V_a * V_b <= 0:
            for _ in range(BISECT_MAXITER):
                n_iter += 1
                mid   = (a + b) / 2.0
                V_mid = _compute_V(
                    mid, xS, zS, x_a, z_a, x_f, z_f, c1, c2, delta
                )

                if abs(V_mid) < 1e-30 or (b - a) / 2.0 < tol:
                    kappa     = mid
                    converged = True
                    break

                if V_a * V_mid < 0:
                    b = mid
                else:
                    a   = mid
                    V_a = V_mid

            kappa = (a + b) / 2.0

    # — Compute final result ——————————————————————————————————————————————————
    x_s = _lin_interp(kappa, xS)
    z_s = _lin_interp(kappa, zS)
    tof = _compute_tof(x_s, z_s, x_a, z_a, x_f, z_f, c1, c2)

    return RayResult(
        k_index=kappa,
        x_s=x_s,
        z_s=z_s,
        tof=tof,
        converged=converged,
        n_iterations=n_iter,
    )


def find_refraction_points_cpu(
    emitters:  np.ndarray,
    focuses:   np.ndarray,
    xS:        np.ndarray,
    zS:        np.ndarray,
    c1:        float,
    c2:        float,
    tol:       float = TOL,
    delta:     float = DELTA,
    tracking:  bool  = True,
) -> tuple:
    """Compute refraction points for all (emitter, focus) pairs on CPU.

    Parameters
    ----------
    emitters : np.ndarray, shape (Na, 2)
        Emitter positions (x, z).
    focuses : np.ndarray, shape (Nf, 2)
        Focus positions (x, z).
    xS, zS : np.ndarray, shape (Ns,)
        Surface points.
    c1, c2 : float
        Wave speeds.
    tol : float
        Convergence tolerance.
    delta : float
        Slope perturbation.
    tracking : bool
        If True, use previous solution as initial guess for the next
        focus point (trajectory tracking for smoother convergence).

    Returns
    -------
    k_result : np.ndarray, shape (Na, Nf)
        Fractional surface indices.
    tof_result : np.ndarray, shape (Na, Nf)
        Time-of-flight values.
    """
    Na = len(emitters)
    Nf = len(focuses)

    k_result   = np.zeros((Na, Nf), dtype=np.float64)
    tof_result = np.zeros((Na, Nf), dtype=np.float64)

    for i in range(Na):
        x_a, z_a = emitters[i]
        k_prev   = None

        for j in range(Nf):
            x_f, z_f = focuses[j]

            result = find_refraction_point(
                x_a,
                z_a,
                x_f,
                z_f,
                xS,
                zS,
                c1,
                c2,
                k_init=k_prev if tracking else None,
                tol=tol,
                delta=delta,
            )

            k_result[i, j]   = result.k_index
            tof_result[i, j] = result.tof

            if tracking:
                k_prev = result.k_index

    return k_result, tof_result


def compute_tof_table_cpu(
    transducer_positions: np.ndarray,
    roi_pixels:           np.ndarray,
    xS:                   np.ndarray,
    zS:                   np.ndarray,
    c1:                   float,
    c2:                   float,
    **kwargs,
) -> np.ndarray:
    """Compute TOF table on CPU for TFM reconstruction.

    Parameters
    ----------
    transducer_positions : np.ndarray, shape (N_elem, 2)
        Element positions.
    roi_pixels : np.ndarray, shape (N_pixels, 2)
        Pixel coordinates.
    xS, zS : np.ndarray
        Surface points.
    c1, c2 : float
        Wave speeds.

    Returns
    -------
    tof_table : np.ndarray, shape (N_elem, N_pixels)
        Time-of-flight from each element to each pixel.
    """
    _, tof_table = find_refraction_points_cpu(
        transducer_positions, roi_pixels, xS, zS, c1, c2, **kwargs
    )
    return tof_table
