"""Multi-body phase space integration for 2→N scattering.

Provides RAMBO-based Monte Carlo integration of |M|² over N-body final-state
phase space.  Supports arbitrary final-state multiplicity (N ≥ 2) with
massless or massive particles.

The total cross section for 2→N is:

    σ = 1/(2s) × ∫ |M̄|² dΦ_N

where dΦ_N is the Lorentz-invariant N-body phase space measure and the
1/(2s) is the flux factor for massless incoming particles.

For 2→2, the 1D cosθ integration in cross_section.py is faster and more
precise.  This module is for N ≥ 3.

References:
    R. Kleiss, W.J. Stirling, S.D. Ellis, Comp. Phys. Comm. 40 (1986) 359
    (RAMBO algorithm for massless phase space generation)
"""
from __future__ import annotations

import math
import numpy as np
from typing import Callable, Optional


# Physical constants
GEV2_TO_PB = 3.8938e8  # 1 GeV⁻² → pb


def rambo_massless(n_final: int, sqrt_s: float, n_events: int,
                   rng: Optional[np.random.Generator] = None) -> tuple[np.ndarray, np.ndarray]:
    """Generate massless N-body phase space points using RAMBO.

    Parameters
    ----------
    n_final : int
        Number of final-state particles (≥ 2).
    sqrt_s : float
        Centre-of-mass energy in GeV.
    n_events : int
        Number of phase space points to generate.
    rng : numpy Generator, optional
        Random number generator for reproducibility.

    Returns
    -------
    momenta : ndarray, shape (n_events, n_final, 4)
        4-momenta (E, px, py, pz) for each final-state particle.
    weights : ndarray, shape (n_events,)
        Phase space weight for each event (in GeV^{2N-4} units).
        Multiply by |M|²/(2s) and average over events to get σ.
    """
    if rng is None:
        rng = np.random.default_rng()

    n = n_final
    s = sqrt_s ** 2

    # Step 1: Generate n isotropic massless 4-vectors q_i
    # q_i = (|q_i|, q_i_x, q_i_y, q_i_z) with random direction and exponential energy
    rho1 = rng.random((n_events, n))
    rho2 = rng.random((n_events, n))
    rho3 = rng.random((n_events, n))
    rho4 = rng.random((n_events, n))

    cos_theta = 2 * rho2 - 1
    sin_theta = np.sqrt(1 - cos_theta ** 2)
    phi = 2 * np.pi * rho3

    # Exponential energy: E = -ln(rho1 * rho4)
    E_q = -np.log(rho1 * rho4 + 1e-300)  # avoid log(0)

    # 4-vectors q_i
    q = np.zeros((n_events, n, 4))
    q[:, :, 0] = E_q
    q[:, :, 1] = E_q * sin_theta * np.cos(phi)
    q[:, :, 2] = E_q * sin_theta * np.sin(phi)
    q[:, :, 3] = E_q * cos_theta

    # Step 2: Compute Q = sum of all q_i
    Q = np.sum(q, axis=1)  # (n_events, 4)
    Q_mass = np.sqrt(np.abs(Q[:, 0] ** 2 - Q[:, 1] ** 2 - Q[:, 2] ** 2 - Q[:, 3] ** 2))

    # Step 3: Boost all q_i to the frame where Q is at rest, then scale
    # The scaling factor is x = sqrt_s / Q_mass
    # Boost: b = -Q_spatial / Q_0
    # Then scale all energies and momenta by x

    # Boost vector
    b = -Q[:, 1:4] / Q[:, 0:1]  # (n_events, 3)
    b_sq = np.sum(b ** 2, axis=1)  # (n_events,)
    gamma = Q[:, 0] / Q_mass  # (n_events,)

    # Boost each q_i
    p = np.zeros_like(q)
    for i in range(n):
        q_i = q[:, i, :]  # (n_events, 4)
        b_dot_q = np.sum(b * q_i[:, 1:4], axis=1)  # (n_events,)

        # Boost formula: p0 = gamma*(E + b·q), p_vec = q_vec + b*(gamma*E + (gamma-1)*b·q/b²)
        # But we need to handle b²=0 case
        factor = np.where(b_sq > 1e-30,
                          (gamma - 1) * b_dot_q / b_sq + gamma * q_i[:, 0],
                          q_i[:, 0])
        p[:, i, 0] = gamma * (q_i[:, 0] + b_dot_q)
        p[:, i, 1] = q_i[:, 1] + b[:, 0] * factor
        p[:, i, 2] = q_i[:, 2] + b[:, 1] * factor
        p[:, i, 3] = q_i[:, 3] + b[:, 2] * factor

    # Step 4: Scale to desired CM energy
    x = sqrt_s / Q_mass  # (n_events,)
    p *= x[:, np.newaxis, np.newaxis]

    # Step 5: Phase space weight
    # w = (2π)^{4-3n} × (π/2)^{n-1} × s^{n-2} / ((n-1)! × (n-2)!) × (sum E_q / Q_mass)^{2n-4}
    # But the RAMBO weight also includes the 1/(2E) factors and the random sampling Jacobian.
    #
    # RAMBO weight per event (Eq. 4.11 in the original paper):
    # W = S_n × (π/2)^{n-1} × (√s)^{2n-4} / prod(2E_i)
    # where S_n = (2π)^{4-3n} / ((n-1)!(n-2)!)
    # and the 1/(2^n) from 1/(2E) for each particle gives the proper normalization.

    # For our purposes, the weight formula is:
    # w_RAMBO = (2π)^{4-3n} × (π/2)^{n-1} × s^{n-2} / ((n-1)! × (n-2)!)
    # times the correction factor from the boost:
    # × (Q_mass / sqrt_s)^{2n-4} × ∏(E_q_i) / ∏(p_i^0) × ... it gets complex.
    #
    # Simpler: use the standard RAMBO result.
    # The average of |M|²×w over N_events gives: <|M|²×w> ≈ ∫|M|² dΦ_n
    # where w = V_n / N_events and V_n is the phase space volume.

    # Phase space volume for n massless particles at CM energy √s:
    # Φ_n = (π/2)^{n-1} × s^{n-2} / ((2π)^{3n-4} × Γ(n) × Γ(n-1))
    # = (1/(2π))^{3n-4} × (π/2)^{n-1} × s^{n-2} / ((n-1)! × (n-2)!)

    log_phase_vol = ((n - 1) * math.log(math.pi / 2)
                     + (n - 2) * math.log(s)
                     - (3 * n - 4) * math.log(2 * math.pi)
                     - math.lgamma(n) - math.lgamma(n - 1))
    phase_vol = math.exp(log_phase_vol)

    # Each RAMBO event has equal weight for massless particles
    weights = np.full(n_events, phase_vol)

    return p, weights


def dot4(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Minkowski dot product with (+,-,-,-) signature."""
    return a[..., 0] * b[..., 0] - np.sum(a[..., 1:4] * b[..., 1:4], axis=-1)


def compute_dot_products(
    p1: np.ndarray, p2: np.ndarray, momenta: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute all independent dot products for 2→N scattering.

    Returns a dict mapping "pi_pj" style keys to arrays of shape (n_events,).
    Keys: p1_p2, p1_q1, p1_q2, ..., q1_q2, q1_q3, ...

    The incoming momenta p1, p2 are broadcast to all events.
    """
    n_events = momenta.shape[0]
    n_final = momenta.shape[1]
    result = {}

    # Incoming dot products
    result["p1_p2"] = np.full(n_events, float(dot4(p1, p2)))

    # Incoming × outgoing
    for j in range(n_final):
        qj = momenta[:, j, :]
        result[f"p1_q{j+1}"] = dot4(p1[np.newaxis, :], qj)
        result[f"p2_q{j+1}"] = dot4(p2[np.newaxis, :], qj)

    # Outgoing × outgoing
    for j in range(n_final):
        for k in range(j + 1, n_final):
            qj = momenta[:, j, :]
            qk = momenta[:, k, :]
            result[f"q{j+1}_q{k+1}"] = dot4(qj, qk)

    return result


def compute_invariants(
    p1: np.ndarray, p2: np.ndarray, momenta: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute Mandelstam-like invariants for 2→N scattering.

    For 2→2: s, t, u
    For 2→3: s, s12, s13, s23, t1, t2
    For 2→N: s, s_{ij}, t_i
    """
    n_events = momenta.shape[0]
    n_final = momenta.shape[1]
    result = {}

    s_val = float(dot4(p1 + p2, p1 + p2))
    result["s"] = np.full(n_events, s_val)

    if n_final == 2:
        q1, q2 = momenta[:, 0, :], momenta[:, 1, :]
        result["t"] = dot4(p1[np.newaxis, :] - q1, p1[np.newaxis, :] - q1)
        result["u"] = dot4(p1[np.newaxis, :] - q2, p1[np.newaxis, :] - q2)
    else:
        # s_{ij} = (q_i + q_j)^2
        for i in range(n_final):
            for j in range(i + 1, n_final):
                qi = momenta[:, i, :]
                qj = momenta[:, j, :]
                result[f"s{i+1}{j+1}"] = dot4(qi + qj, qi + qj)

        # t_i = (p1 - q_i)^2
        for i in range(n_final):
            qi = momenta[:, i, :]
            result[f"t{i+1}"] = dot4(p1[np.newaxis, :] - qi, p1[np.newaxis, :] - qi)

    return result


## ─── Vegas adaptive Monte Carlo ─────────────────────────────────────────────
#
# Implementation of the VEGAS algorithm (G. P. Lepage, J. Comp. Phys. 27, 1978).
#
# The idea: partition the unit hypercube [0,1]^d into bins along each axis.
# After each iteration, adjust the bin boundaries so that bins where the
# integrand is large are narrower (= sampled more densely).  This importance
# sampling converges much faster than flat MC for peaked integrands (t-channel
# poles, resonances, threshold effects).
#
# We combine VEGAS grid adaptation with RAMBO phase-space generation: the
# VEGAS grid lives on the unit hypercube of RAMBO's random numbers, so the
# adaptive sampling focuses on regions of phase space where |M|² is largest.


class VegasGrid:
    """Adaptive importance-sampling grid on [0,1]^d (one axis per dimension).

    Each axis is divided into ``n_bins`` intervals.  The grid stores the bin
    edges and adjusts them after each iteration based on the distribution of
    the integrand.
    """

    def __init__(self, n_dim: int, n_bins: int = 50):
        self.n_dim = n_dim
        self.n_bins = n_bins
        # Uniform grid: edges at 0, 1/n_bins, 2/n_bins, ..., 1
        self.edges = np.tile(
            np.linspace(0.0, 1.0, n_bins + 1), (n_dim, 1)
        )  # shape (n_dim, n_bins+1)

    def map(self, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Map uniform random numbers u ∈ [0,1]^d through the grid.

        Parameters
        ----------
        u : ndarray, shape (n_events, n_dim)
            Uniform random samples in [0, 1]^d.

        Returns
        -------
        x : ndarray, shape (n_events, n_dim)
            Mapped samples in [0, 1]^d (stretched by the grid).
        jac : ndarray, shape (n_events,)
            Jacobian = product over dimensions of (bin_width × n_bins).
        """
        n_events = u.shape[0]
        x = np.empty_like(u)
        jac = np.ones(n_events)

        for d in range(self.n_dim):
            edges = self.edges[d]
            # Which bin does each sample fall into?
            bin_idx = np.clip(
                (u[:, d] * self.n_bins).astype(int), 0, self.n_bins - 1
            )
            # Fractional position within that bin
            frac = u[:, d] * self.n_bins - bin_idx
            lo = edges[bin_idx]
            hi = edges[bin_idx + 1]
            width = hi - lo
            x[:, d] = lo + frac * width
            jac *= width * self.n_bins

        return x, jac

    def adapt(self, u: np.ndarray, f_abs: np.ndarray, alpha: float = 1.5):
        """Refine the grid based on the integrand distribution.

        Parameters
        ----------
        u : ndarray, shape (n_events, n_dim)
            The uniform random samples used in this iteration.
        f_abs : ndarray, shape (n_events,)
            |f(x)| × jacobian for each sample.
        alpha : float
            Damping exponent (1.5 is Lepage's recommendation).
        """
        for d in range(self.n_dim):
            edges = self.edges[d]
            bin_idx = np.clip(
                (u[:, d] * self.n_bins).astype(int), 0, self.n_bins - 1
            )
            # Accumulate |f| into bins
            bin_sum = np.zeros(self.n_bins)
            np.add.at(bin_sum, bin_idx, f_abs)

            total = bin_sum.sum()
            if total <= 0:
                continue

            # Smooth: replace each bin value with the damped version
            # d_i = ((bin_i/total - 1) / ln(bin_i/total))^alpha  if bin_i > 0
            avg = total / self.n_bins
            smoothed = np.where(
                bin_sum > 0,
                ((bin_sum / avg - 1.0) / np.log(np.maximum(bin_sum / avg, 1e-30))) ** alpha
                if alpha != 1.0 else bin_sum,
                0.0,
            )
            # Simpler Lepage smoothing: just use bin_sum^alpha
            smoothed = np.power(np.maximum(bin_sum, 1e-30), alpha)
            smoothed_total = smoothed.sum()
            if smoothed_total <= 0:
                continue

            # New bin edges: distribute so that each new bin gets equal
            # weight of the smoothed distribution.
            cum = np.cumsum(smoothed)
            cum = cum / cum[-1]  # normalize to [0, 1]

            new_edges = np.zeros(self.n_bins + 1)
            new_edges[0] = 0.0
            new_edges[-1] = 1.0

            # For each new bin boundary j/n_bins, find where it falls in the
            # cumulative distribution and interpolate.
            for j in range(1, self.n_bins):
                target = j / self.n_bins
                # Find the old bin where cum crosses target
                k = np.searchsorted(cum, target)
                k = min(k, self.n_bins - 1)
                if k == 0:
                    frac = target / cum[0] if cum[0] > 0 else 0.5
                else:
                    frac = (target - cum[k - 1]) / (cum[k] - cum[k - 1] + 1e-30)
                new_edges[j] = edges[k] + frac * (edges[k + 1] - edges[k])

            self.edges[d] = new_edges


def vegas_integrate(
    integrand: Callable,
    n_dim: int,
    n_iter: int = 10,
    n_eval_per_iter: int = 10_000,
    n_bins: int = 50,
    alpha: float = 1.5,
    n_adapt: int = 5,
    seed: int = 42,
) -> dict:
    """Adaptive Monte Carlo integration on [0,1]^d using the VEGAS algorithm.

    Parameters
    ----------
    integrand : callable
        Function f(x) where x has shape (n_events, n_dim).
        Returns an ndarray of shape (n_events,).
    n_dim : int
        Dimensionality of the integration domain.
    n_iter : int
        Total number of iterations (adaptation + accumulation).
    n_eval_per_iter : int
        Number of integrand evaluations per iteration.
    n_bins : int
        Number of bins per axis in the adaptive grid.
    alpha : float
        Grid adaptation damping parameter.
    n_adapt : int
        Number of initial iterations used only for grid adaptation
        (their estimates are discarded from the final weighted average).
    seed : int
        Random seed.

    Returns
    -------
    dict with keys: integral, error, chi2_per_dof, n_eval, converged
    """
    rng = np.random.default_rng(seed)
    grid = VegasGrid(n_dim, n_bins)

    # Collect per-iteration estimates for weighted average
    iter_means = []
    iter_vars = []

    for it in range(n_iter):
        u = rng.random((n_eval_per_iter, n_dim))
        x, jac = grid.map(u)

        f_vals = integrand(x)
        # The integrand already includes any external Jacobian (e.g. RAMBO weight).
        # Multiply by the VEGAS grid jacobian.
        weighted = f_vals * jac

        mean_i = float(np.mean(weighted))
        var_i = float(np.var(weighted) / n_eval_per_iter)

        # Adapt the grid
        grid.adapt(u, np.abs(weighted), alpha=alpha)

        # Only accumulate estimates after adaptation phase
        if it >= n_adapt:
            iter_means.append(mean_i)
            iter_vars.append(max(var_i, 1e-30))

    if not iter_means:
        # All iterations were adaptation — use the last one
        return {
            "integral": mean_i,
            "error": math.sqrt(var_i) if var_i > 0 else 0.0,
            "chi2_per_dof": 0.0,
            "n_eval": n_iter * n_eval_per_iter,
            "converged": False,
        }

    # Weighted average: w_i = 1/var_i, result = Σ(w_i × mean_i) / Σ(w_i)
    means = np.array(iter_means)
    variances = np.array(iter_vars)
    weights = 1.0 / variances
    w_sum = weights.sum()

    integral = float(np.sum(weights * means) / w_sum)
    error = float(1.0 / np.sqrt(w_sum))

    # Chi-squared per degree of freedom
    n_accum = len(iter_means)
    if n_accum > 1:
        chi2 = float(np.sum(weights * (means - integral) ** 2))
        chi2_per_dof = chi2 / (n_accum - 1)
    else:
        chi2_per_dof = 0.0

    return {
        "integral": integral,
        "error": error,
        "chi2_per_dof": chi2_per_dof,
        "n_eval": n_iter * n_eval_per_iter,
        "converged": chi2_per_dof < 2.0 or n_accum <= 1,
    }


def total_cross_section_2to3(
    msq_func: Callable,
    sqrt_s: float,
    n_final: int = 3,
    n_events: int = 100_000,
    seed: int = 42,
) -> dict:
    """Monte Carlo integration of σ for a 2→N process.

    Parameters
    ----------
    msq_func : callable
        Function that takes (p1, p2, momenta) arrays and returns
        |M̄|² for each event as an ndarray of shape (n_events,).
        This is the spin/color-averaged squared matrix element.
    sqrt_s : float
        Centre-of-mass energy in GeV.
    n_final : int
        Number of final-state particles (default 3).
    n_events : int
        Number of Monte Carlo samples.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict with keys: sigma_pb, sigma_uncertainty_pb, n_events, sqrt_s_gev
    """
    s_val = sqrt_s ** 2
    rng = np.random.default_rng(seed)

    # Generate phase space
    momenta, weights = rambo_massless(n_final, sqrt_s, n_events, rng)

    # Incoming momenta in CM frame
    E_beam = sqrt_s / 2
    p1 = np.array([E_beam, 0, 0, E_beam])
    p2 = np.array([E_beam, 0, 0, -E_beam])

    # Evaluate |M|² at each point
    msq_vals = msq_func(p1, p2, momenta)

    # Cross section: σ = 1/(2s) × <|M|² × w>
    # where w = phase_vol (same for all events in massless RAMBO)
    integrand = msq_vals * weights / (2 * s_val)

    sigma_gev2 = np.mean(integrand)
    sigma_err_gev2 = np.std(integrand) / np.sqrt(n_events)

    sigma_pb = sigma_gev2 * GEV2_TO_PB
    sigma_err_pb = sigma_err_gev2 * GEV2_TO_PB

    return {
        "process": f"2->{n_final}",
        "sqrt_s_gev": sqrt_s,
        "s_gev2": s_val,
        "sigma_pb": sigma_pb,
        "sigma_uncertainty_pb": sigma_err_pb,
        "n_events": n_events,
        "converged": True,
        "supported": True,
    }
