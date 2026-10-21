"""Closed forms from Phase 1 (T1-T4), SI S1-S2 draft of 2026-10-12.

Conventions (notation.md): d[X]/dt = -k_d2 [X]^2 (consumption convention); cross-deactivation
X + R -> 2D has event rate k_x [X][R]; reservoir depth D = 1 + K_R theta_a; F = c/D is the free
(on-cycle) pool; k_P turnover frequency per on-cycle catalyst; f_X fraction of the pool in the
deactivation source X.

Ported from 03_calculations/01_analytic_derivations/p1_theory/p1lib.py and the T1-T4 scripts.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.special import exp1, i0e, i1e, k0e, k1e

# ----------------------------------------------------------------------------- T1
def ton_single_cycle(p_c):
    """Costentin 2022 single-cycle limit: N is geometric with death probability p_c per turnover,
    TON_lim = <N> = (1 - p_c)/p_c when N counts completed turnovers before death from the start
    of a cycle (= 1/p_c - 1).  For death competing with the turnover step, TON = k_P/k_d."""
    return (1.0 - p_c) / p_c


def lifetime_with_reservoirs(tau0, K, T_attach0):
    """T1 corollary: <tau> = <tau0> + sum_a K_a <T_a0>, T_a0 = mean time spent in the attachment
    state a without reservoirs."""
    return tau0 + float(np.dot(np.atleast_1d(K), np.atleast_1d(T_attach0)))


# ----------------------------------------------------------------------------- T2
def P_inf_closed_batch(S0, c0, k1, k2, kdA, kdB):
    """Closed batch, finite substrate, deactivation kdA from A (branching depends on S) and kdB
    from the resting state B: P_inf solves c0 = (kdA/k1) ln(S0/(S0-P)) + (kdB/k2) P, for any
    pendant reservoir or product-inhibition adduct (T2)."""
    f = lambda P: c0 - (kdA / k1) * np.log(S0 / (S0 - P)) - (kdB / k2) * P
    if f(S0 * (1 - 1e-15)) > 0:
        return S0
    return brentq(f, 0.0, S0 * (1 - 1e-15), xtol=1e-18, rtol=1e-14)


def psi(z):
    """TON reduction by an external exponential clock: TON/TON0 = Psi(z) = z^-1 e^{1/z} E1(1/z),
    z = lambda D tau0. Evaluated stably via the scaled exponential integral."""
    z = np.asarray(z, dtype=float)
    out = np.ones_like(z)
    nz = z > 0
    x = 1.0 / z[nz]
    # e^x E1(x): use exp1 for moderate x, asymptotic series for large x
    big = x > 700
    ex = np.empty_like(x)
    ex[~big] = np.exp(x[~big]) * exp1(x[~big])
    xb = x[big]
    ex[big] = (1 / xb) * (1 - 1 / xb + 2 / xb**2 - 6 / xb**3 + 24 / xb**4)
    out[nz] = x * ex
    return out if out.ndim else float(out)


def ton_poison(kP, kp, g, D=1.0):
    """Poison made at constant rate g (M/s) killing free catalyst with kp (M^-1 s^-1):
    TON(D) = kP sqrt(pi/(2 kp g)) / sqrt(D)."""
    return kP * np.sqrt(np.pi / (2 * kp * g)) / np.sqrt(D)


# ----------------------------------------------------------------------------- T3
def ton_inf_mixed(kP, kd1, kd2, c0, D=1.0, kdR=0.0, kx=0.0, fX=1.0):
    """Catalyst-limited TON for a fast reservoir of depth D on the free pool.
    Deactivation kd1 Xd + kd2 Xd^2 (Xd = fX * free pool), reservoir decay kdR R, cross X + R -> 2D
    with event constant kx. TON = kP D/(kap c0) ln(1 + kap c0/(D a)),
    a = kd1 fX + kdR (D-1), kap = kd2 fX^2 + 2 kx fX (D-1)."""
    a = kd1 * fX + kdR * (D - 1.0)
    kap = kd2 * fX**2 + 2.0 * kx * fX * (D - 1.0)
    if kap == 0:
        return kP / a
    return kP * D / (kap * c0) * np.log1p(kap * c0 / (D * a))


def gain_mixed_ceiling(rho):
    """Mixed order: G(D -> inf) = rho / ln(1 + rho), rho = kd2 fX c0 / kd1 (first-order ceiling)."""
    return rho / np.log1p(rho)


def decaying_reservoir_helps(kdR, kd1_tilde, rho):
    """A decaying reservoir raises TON at small depth iff kdR/kd1~ < (1+rho) ln(1+rho)/rho - 1."""
    return kdR / kd1_tilde < (1 + rho) * np.log1p(rho) / rho - 1


def D_opt_decaying(kP, kd1, kd2, c0, kdR, fX=1.0, Dmax=1e6):
    """Numerical optimum depth for a decaying reservoir (from ton_inf_mixed)."""
    r = minimize_scalar(lambda lD: -ton_inf_mixed(kP, kd1, kd2, c0, D=np.exp(lD), kdR=kdR, fX=fX),
                        bounds=(1e-9, np.log(Dmax)), method="bounded", options={"xatol": 1e-10})
    return float(np.exp(r.x))


def cross_threshold(kd2, fX=1.0):
    """Cross-deactivation X + R -> 2D: the reservoir helps iff kx < kd2 fX / 2."""
    return kd2 * fX / 2.0


def ton_dimer_reservoir(kP, kd1, kd2, Kdim, c0):
    """Dimeric reservoir 2X <-> X2 (K_dim in M^-1, fast), turnover kP X, deactivation kd1 X + kd2 X^2:
    TON = (kP/c0) [4 K/kd2 X0 + (1/kd2)(1 - 4 K kd1/kd2) ln(1 + kd2 X0/kd1)],
    X0 = (-1 + sqrt(1 + 8 K c0))/(4 K)."""
    X0 = (-1 + np.sqrt(1 + 8 * Kdim * c0)) / (4 * Kdim) if Kdim > 0 else c0
    return kP / c0 * (4 * Kdim / kd2 * X0 + (1 / kd2) * (1 - 4 * Kdim * kd1 / kd2) * np.log1p(kd2 * X0 / kd1))


def ton_bimetallic(kP2, kd1, c0, D=1.0):
    """n_p = 2, n_d = 1 (turnover 2X -> 2X + P at kP2 X^2, death kd1 X): TON = (kP2/kd1) c0/(2D)."""
    return kP2 / kd1 * c0 / (2.0 * D)


# ----------------------------------------------------------------------------- T4
def ton_T_mixed(kP, kd1, kd2, c0, T, D=1.0):
    """TON at horizon T, fast reservoir depth D, deactivation kd1 X + kd2 X^2 of the free pool."""
    if kd1 == 0:
        return kP * D / (kd2 * c0) * np.log1p(kd2 * c0 * T / D**2)
    return kP * D / (kd2 * c0) * np.log1p(kd2 * c0 / (kd1 * D) * (-np.expm1(-kd1 * T / D)))


def xstar():
    """x* = y*^-1/2 with y* the root of (1+y) ln(1+y) = 2y. Returns (x*, y*)."""
    y = brentq(lambda y: (1 + y) * np.log1p(y) - 2 * y, 1.0, 10.0, xtol=1e-15)
    return 1.0 / np.sqrt(y), y


def gstar():
    """Efficiency of the best fast reservoir, g* = 2x*/(1+x*^2) = 0.8047."""
    x, _ = xstar()
    return 2 * x / (1 + x**2)


def D_opt_second_order(kd2, c0, T):
    """D* = x* sqrt(kd2 c0 T) (needs beta = kd2 c0 T > y* = 3.92; otherwise D* = 1)."""
    x, y = xstar()
    beta = kd2 * c0 * T
    return max(1.0, x * np.sqrt(beta)) if beta > y else 1.0


def ton_star_second_order(kP, kd2, c0, T):
    """TON at the optimum depth, g* kP sqrt(T/(kd2 c0)) (beta > y*)."""
    return gstar() * kP * np.sqrt(T / (kd2 * c0))


def D_opt_event_convention_prefactor():
    """Prefactor of D* in the event convention d[X]/dt = -2 k_ev X^2: x* sqrt(2) = 0.714."""
    return xstar()[0] * np.sqrt(2.0)


def slow_release_P(kP, kd2, c0, koff, T):
    """Exact product at horizon T for irreversible first-order release (k_on = 0) of c0 from a
    reservoir into a pool dying by second order: dX/dt = koff c0 e^{-koff t} - kd2 X^2.
    P(T) = (kP/kd2) ln[xi0 (I0(xiT) K1(xi0) + K0(xiT) I1(xi0))], xi = 2 sqrt(kd2 c0/koff) e^{-koff t/2}."""
    xi0 = 2.0 * np.sqrt(kd2 * c0 / koff)
    xiT = xi0 * np.exp(-koff * T / 2.0)
    l1 = np.log(i0e(xiT)) + np.log(k1e(xi0)) + (xiT - xi0)
    l2 = np.log(k0e(xiT)) + np.log(i1e(xi0)) + (xi0 - xiT)
    return kP / kd2 * (np.log(xi0) + np.logaddexp(l1, l2))


def slow_release_opt(kP, kd2, c0, T):
    """Optimal koff T for the slow-release reservoir and its P(T)."""
    r = minimize_scalar(lambda lk: -slow_release_P(kP, kd2, c0, 10**lk / T, T), bounds=(-2, 3),
                        method="bounded", options={"xatol": 1e-9})
    return 10**r.x, -r.fun


def const_feed_P(kP, kd2, c0, T):
    """Constant addition of c0 over [0, T] into a second-order pool: P = (kP/kd2) ln cosh sqrt(beta)."""
    b = np.sqrt(kd2 * c0 * T)
    return kP / kd2 * (b + np.log1p(np.exp(-2 * b)) - np.log(2.0))


def dosing_bound_P(kP, kd1, kd2, c0, T):
    """Upper bound on P(T) over every dosing/reservoir/dilution strategy with metal budget c0:
    P <= kP Xi*, kd1 Xi* + kd2 Xi*^2/T = c0."""
    if kd2 == 0:
        return kP * c0 / kd1
    return kP * T / (2 * kd2) * (-kd1 + np.sqrt(kd1**2 + 4 * kd2 * c0 / T))


# ----------------------------------------------------------------------------- P2 additions
def ton_inf_bimetallic_mixed(kP2, kd1, kd2, c0, D=1.0):
    """n_p = 2 (turnover 2X -> 2X + P, d[P]/dt = kP2 F^2) with mixed-order death kd1 F + kd2 F^2 of the
    free pool F = c/D (consumption convention):
        TON = (D kP2/c0) [u/kd2 - (kd1/kd2^2) ln(1 + kd2 u/kd1)],  u = c0/D.
    Limits: kd2 -> 0 gives kP2 c0/(2 D kd1) (= ton_bimetallic); kd1 -> 0 diverges only through the
    first-order channel, and the pure n_p = n_d = 2 network has TON = kP2/kd2 for every D (exact no-go)."""
    u = c0 / D
    if kd2 == 0:
        return kP2 * c0 / (2.0 * D * kd1)
    if kd1 == 0:
        return kP2 / kd2
    x = kd2 * u / kd1
    # u/kd2 - (kd1/kd2^2) ln(1+x) = (kd1/kd2^2)(x - ln(1+x)); series for small x avoids cancellation
    f = x - np.log1p(x) if x > 1e-3 else x * x / 2 - x**3 / 3 + x**4 / 4
    return D * kP2 / c0 * (kd1 / kd2**2) * f


def ton_T_bimetallic_mixed(kP2, kd1, kd2, c0, T, D=1.0):
    """TON(T) for n_p = 2 and mixed-order death of the free pool F = c/D (fast reservoir):
    F(t) obeys dF/dt = -(kd1 F + kd2 F^2)/D  (live catalyst c = D F), P' = kP2 F^2.
    Evaluated by quadrature of the exact F(t) (closed form for F, numerical integral for P)."""
    from scipy.integrate import quad
    F0 = c0 / D
    if kd1 == 0:
        # F = F0/(1 + kd2 F0 t/D);  P = kP2 F0^2 T/(1 + kd2 F0 T/D)... integrate exactly
        a = kd2 * F0 / D
        return kP2 * F0**2 * T / (1.0 + a * T) / c0
    def F(t):
        e = np.exp(-kd1 * t / D)
        return F0 * e / (1.0 + kd2 * F0 / kd1 * (1.0 - e))
    val, _ = quad(lambda t: kP2 * F(t) ** 2, 0.0, T, limit=400, epsabs=0, epsrel=1e-11)
    return val / c0


def loading_slope_mixed(kP, kd1, kd2, c0, D=1.0, kdR=0.0, kx=0.0, fX=1.0, h=1e-4):
    """Local loading slope s = d ln TON_inf / d ln c0 of ton_inf_mixed (central difference in ln c0)."""
    f = lambda c: np.log(ton_inf_mixed(kP, kd1, kd2, c, D=D, kdR=kdR, kx=kx, fX=fX))
    return (f(c0 * np.exp(h)) - f(c0 * np.exp(-h))) / (2 * h)
