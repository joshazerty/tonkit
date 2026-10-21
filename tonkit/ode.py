"""Stiff mass-action ODEs for a Network, and the three TON definitions of notation.md §4.

    ton_inf(net)       catalyst-limited TON: integrate until the live catalyst has died
    ton_T(net, T)      TON at a fixed horizon T
    simulate(net, ...) full trajectory

All quantities are per total metal loaded (c0 = sum over catalyst and dead species of
metal * conc0). Fixed species are held at conc0. Time-dependent rate constants (callables)
are allowed, e.g. a zero-order feed '-> X' with k(t) for slow catalyst addition.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

RTOL = 1e-8
ATOL_REL = 1e-16          # absolute tolerance, relative to the natural scale of each species


class Model:
    """Compiled right-hand side and Jacobian of a Network (mass action)."""

    def __init__(self, net, extra_rhs=None):
        net.validate(strict=True)
        self.net = net
        self.ns = len(net.species)
        Rm, Pm = net.stoich()
        self.S = Pm - Rm
        self.fixed = np.array([s.fixed for s in net.species])
        self.S[self.fixed, :] = 0.0
        self.ks = net.rate_constants()
        self.time_dep = [callable(k) for k in self.ks]
        self.kconst = np.array([0.0 if callable(k) else float(k) for k in self.ks])
        # sparse rate-law exponent lists (stoichiometric unless a reaction sets 'orders')
        E = net.orders_matrix()
        self.reac = [[(i, E[i, j]) for i in np.nonzero(E[:, j])[0]] for j in range(E.shape[1])]
        self.nr = Rm.shape[1]
        self.extra_rhs = extra_rhs
        self.live_w = net.live_metal_vector()
        self.metal_w = net.metal_vector()
        self.iP = net.index(net.product)
        self.S_dense = self.S

    def kvec(self, t):
        if not any(self.time_dep):
            return self.kconst
        k = self.kconst.copy()
        for j, td in enumerate(self.time_dep):
            if td:
                k[j] = self.ks[j](t)
        return k

    def rates(self, t, y):
        k = self.kvec(t)
        r = k.copy()
        for j, lst in enumerate(self.reac):
            for i, nu in lst:
                yi = y[i]
                r[j] *= yi if nu == 1 else (yi * yi if nu == 2 else yi ** nu)
        return r

    def rhs(self, t, y):
        dy = self.S @ self.rates(t, y)
        if self.extra_rhs is not None:
            dy = dy + self.extra_rhs(t, y)
        return dy

    def jac(self, t, y):
        k = self.kvec(t)
        dr = np.zeros((self.nr, self.ns))
        for j, lst in enumerate(self.reac):
            for a, (i, nu) in enumerate(lst):
                v = k[j] * nu * (y[i] ** (nu - 1) if nu != 1 else 1.0)
                for b, (l, mu) in enumerate(lst):
                    if b != a:
                        v *= y[l] ** mu
                dr[j, i] = v
        return self.S @ dr

    def live(self, y):
        return self.live_w @ y


@dataclass
class Result:
    t: np.ndarray
    y: np.ndarray
    names: list
    c0: float
    P: float                 # product formed (M) at the end
    TON: float
    t_half: Optional[float]  # time to half of the final product
    sol: object
    status: str

    def __getitem__(self, name):
        return self.y[self.names.index(name)]


def _atol(net, y0):
    c0 = max(net.c0_total(), 1e-300)
    scale = np.array([c0 if (s.catalyst or s.dead) else max(abs(s.conc0), c0) for s in net.species])
    return ATOL_REL * scale


class _Unscaled:
    """Wrap a dense-output solution of the scaled variables z = y/scale so that sol(t) returns y."""
    def __init__(self, sol, scale):
        self._sol, self._s = sol, scale
    def __call__(self, t):
        z = self._sol(t)
        return z * (self._s if np.ndim(z) == 1 else self._s[:, None])


def simulate(net, t_end, t_eval=None, method="LSODA", rtol=RTOL, atol=None, stop_dead=None,
             dense=False, extra_rhs=None, max_step=np.inf, first_step=None, scale=None, atol_scaled=1e-12):
    """Integrate the network to t_end. stop_dead = f stops when live metal < f * c0.

    scale: None (integrate y as is, atol = ATOL_REL x natural scale) or 'auto' / an array: integrate the
    scaled variables z = y / scale with absolute tolerance atol_scaled. 'auto' takes the initial
    populations (floor 1e-14 c0; c0 for species that start at zero) as the scale, which keeps tiny
    quasi-steady intermediates (rates up to 1e8 s^-1 against deaths at 1e-6 s^-1) well conditioned."""
    m = Model(net, extra_rhs)
    y0 = net.y0()
    c0 = net.c0_total()
    sc = None
    if scale is not None:
        if isinstance(scale, str):
            sc = np.where(y0 > 0, np.maximum(y0, 1e-14 * c0), c0)
        else:
            sc = np.asarray(scale, dtype=float)
    events = None
    if stop_dead is not None:
        if sc is None:
            def ev(t, y):
                return m.live(y) - stop_dead * c0
        else:
            def ev(t, z):
                return m.live(z * sc) - stop_dead * c0
        ev.terminal = True; ev.direction = -1
        events = ev
    if sc is None:
        sol = solve_ivp(m.rhs, (0.0, t_end), y0, method=method, jac=m.jac, rtol=rtol,
                        atol=_atol(net, y0) if atol is None else atol, events=events,
                        t_eval=t_eval, dense_output=dense, max_step=max_step, first_step=first_step)
        yout, solobj = sol.y, sol
    else:
        fun = lambda t, z: m.rhs(t, z * sc) / sc
        jac = lambda t, z: (m.jac(t, z * sc) * sc[None, :]) / sc[:, None]
        sol = solve_ivp(fun, (0.0, t_end), y0 / sc, method=method, jac=jac, rtol=rtol,
                        atol=atol_scaled if atol is None else atol, events=events,
                        t_eval=t_eval, dense_output=dense, max_step=max_step, first_step=first_step)
        yout = sol.y * sc[:, None]
        solobj = sol
        if sol.sol is not None:
            solobj = type('ScaledSol', (), {})()
            solobj.sol = _Unscaled(sol.sol, sc)
    if not sol.success:
        raise RuntimeError(f"integration failed: {sol.message}")
    P = yout[m.iP, -1] - y0[m.iP]
    status = "dead" if (stop_dead is not None and sol.status == 1) else "horizon"
    return Result(sol.t, yout, net.names, c0, P, P / c0 if c0 > 0 else np.nan, None, solobj, status), m


def _t_half(sol, iP, P0, Pend, t0, t1):
    target = P0 + 0.5 * (Pend - P0)
    f = lambda t: sol.sol(t)[iP] - target
    # bracket on a log grid (the curve is monotone for a product that is only formed)
    tg = np.unique(np.concatenate([[t0], np.geomspace(max(t1 * 1e-16, 1e-300), t1, 400), [t1]]))
    vals = np.array([f(t) for t in tg])
    i = np.nonzero(vals >= 0)[0]
    if len(i) == 0:
        return None
    i = i[0]
    if i == 0:
        return tg[0]
    return brentq(f, tg[i - 1], tg[i], xtol=1e-14 * tg[i], rtol=1e-12)


def ton_inf(net, t_max=1e16, live_tol=1e-13, half_time=True, method="LSODA", rtol=RTOL, **kw):
    """Catalyst-limited TON (notation.md: TON_inf): integrate until live metal < live_tol * c0.
    Hold co-substrates 'fixed' for a true TON_inf; with finite substrate this returns TON_cap.
    Returns a Result (TON, P, t_half = time to half of the final product)."""
    res, m = simulate(net, t_max, method=method, rtol=rtol, stop_dead=live_tol, dense=half_time, **kw)
    if res.status != "dead":
        live = m.live(res.y[:, -1]) / res.c0
        if live > 1e-6:
            res.status = f"not dead at t_max (live fraction {live:.2e}); TON is a lower bound"
    if half_time:
        res.t_half = _t_half(res.sol, m.iP, res.y[m.iP, 0], res.y[m.iP, -1], res.t[0], res.t[-1])
    return res


def ton_T(net, T, method="LSODA", rtol=RTOL, **kw):
    """TON at the horizon T (notation.md: TON(T))."""
    res, _ = simulate(net, T, method=method, rtol=rtol, **kw)
    return res.TON


def P_of_t(net, ts, method="LSODA", rtol=RTOL, **kw):
    """Product formed at the times ts (array), M."""
    ts = np.asarray(ts, dtype=float)
    res, m = simulate(net, ts[-1], t_eval=ts, method=method, rtol=rtol, **kw)
    return res.y[m.iP] - net.y0()[m.iP]
