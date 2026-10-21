"""T4 - optimal depth over a horizon T; slow-release reservoir (Bessel); slow addition; dosing bound."""
import numpy as np
import pytest
from scipy.optimize import minimize_scalar, brentq

from tonkit import ton_T
from tonkit import networks as nw
from tonkit import closed_forms as cf

kP = 1.0


def test_constants(check):
    x, y = cf.xstar()
    check("x* = 0.504976", x, 0.504976, tol=2e-6)
    check("y* = 3.92155", y, 3.92155, tol=2e-6)
    check("g* = 0.804742", cf.gstar(), 0.804742, tol=2e-6)
    check("event-convention prefactor 0.714", cf.D_opt_event_convention_prefactor(), 0.71414, tol=1e-4)
    check("(1+y*) ln(1+y*) = 2 y*", (1 + y) * np.log1p(y), 2 * y, tol=1e-12)


def fast_res_ton(D, kd1, kd2, c0, T):
    return ton_T(nw.pool(kP=kP, c0=c0, K_R=D - 1, koff=1e6 / T, kd1=kd1, kd2=kd2), T)


def test_fast_reservoir_ton_T(check):
    c0, kd2, T = 1e-3, 10.0, 1e4          # beta = 100
    Dst = cf.D_opt_second_order(kd2, c0, T)
    for D in (1.0, 2.0, 0.8 * Dst, Dst, 1.25 * Dst, 10.0, 20.0):
        check(f"TON(T; D = {D:.3f}) pure 2nd order", fast_res_ton(D, 0.0, kd2, c0, T),
              cf.ton_T_mixed(kP, 0.0, kd2, c0, T, D=D))


def test_optimum_depth_from_ode(check):
    c0, kd2, T = 1e-3, 10.0, 1e4
    r = minimize_scalar(lambda D: -fast_res_ton(D, 0.0, kd2, c0, T), bounds=(1.0, 30.0), method="bounded",
                        options={"xatol": 1e-5})
    check("ODE optimum D* vs x* sqrt(beta)", r.x, cf.D_opt_second_order(kd2, c0, T))
    check("ODE TON* vs g* kP sqrt(T/(kd2 c0))", -r.fun, cf.ton_star_second_order(kP, kd2, c0, T))


def test_optimum_depth_explicit_cycle(check):
    """Same optimum with an explicit two-state cycle (k1 S = 100, k2 = 1, 2B -> D2, reservoir on B)."""
    c0, kd2, T = 1e-3, 10.0, 1e4
    fB = nw.cycle().meta["fB"]
    Dst = cf.xstar()[0] * np.sqrt(kd2 * fB**2 * c0 * T)
    for D in (1.0, Dst, 3 * Dst):
        net = nw.cycle(K_R=(D - 1) / fB, kd2=kd2, c0=c0)
        check(f"explicit cycle TON(T) D = {D:.3f}", ton_T(net, T), cf.ton_T_mixed(fB, 0.0, kd2 * fB**2, c0, T, D=D))


@pytest.mark.parametrize("gamma", [0.0, 0.3, 3.0, 30.0])
def test_mixed_order_horizon(check, gamma):
    c0, kd2, T = 1e-3, 1e3, 1e4          # beta = 1e4
    kd1 = gamma / T
    rc = minimize_scalar(lambda D: -cf.ton_T_mixed(kP, kd1, kd2, c0, T, D=D), bounds=(1.0, 500.0),
                         method="bounded", options={"xatol": 1e-6})
    for D in (1.0, rc.x, 3 * rc.x):
        check(f"mixed-order TON(T) gamma {gamma:g} D {D:.2f}", fast_res_ton(D, kd1, kd2, c0, T),
              cf.ton_T_mixed(kP, kd1, kd2, c0, T, D=D))
    ro = minimize_scalar(lambda D: -fast_res_ton(D, kd1, kd2, c0, T), bounds=(1.0, 500.0),
                         method="bounded", options={"xatol": 1e-4})
    check(f"mixed-order D* (ODE vs closed form) gamma {gamma:g}", ro.x, rc.x)


@pytest.mark.parametrize("koffT", [0.3, 1.0, 2.5, 10.0, 100.0])
def test_slow_release_bessel(check, koffT):
    c0, kd2, T = 1e-3, 10.0, 1e4
    koff = koffT / T
    net = nw.pool(kP=kP, c0=c0, K_R=0.0, koff=koff, kd2=kd2, start="R")
    net.add_reaction("R -> X", k=koff, id="release")
    check(f"slow release koff T = {koffT:g}", ton_T(net, T), cf.slow_release_P(kP, kd2, c0, koff, T) / c0)


def test_slow_release_optimum(check):
    c0, T = 1e-3, 1e4
    for beta, ref in ((1e2, None), (1e4, None), (1e6, None)):
        kd2 = beta / (c0 * T)
        kT, P = cf.slow_release_opt(kP, kd2, c0, T)
        bound = cf.dosing_bound_P(kP, 0.0, kd2, c0, T)
        if beta >= 1e4:
            check(f"slow-release optimum koff T near 2.5 (beta {beta:g})", kT, 2.5, tol=0.05)
            check(f"slow-release efficiency ~0.90 (beta {beta:g})", P / bound, 0.90, tol=0.01)
    # QSS optimum: koff T = 2 z*, z* root of e^z - 1 = 2z
    zs = brentq(lambda z: np.expm1(z) - 2 * z, 0.5, 3.0)
    kT, _ = cf.slow_release_opt(kP, 1e6 / (c0 * T), c0, T)
    check("large-beta optimum koff T -> 2 z*", kT, 2 * zs, tol=1e-2)


@pytest.mark.parametrize("beta", [1e2, 1e4])
def test_constant_feed(check, beta):
    c0, T = 1e-3, 1e4
    kd2 = beta / (c0 * T)
    net = nw.pool(kP=kP, c0=c0, kd2=kd2, feed=lambda t: c0 / T)
    check(f"constant feed ln cosh (beta {beta:g})", ton_T(net, T, max_step=T / 200), cf.const_feed_P(kP, kd2, c0, T) / c0)


def test_exponential_feed_equals_slow_release(check):
    c0, T, kd2 = 1e-3, 1e4, 1.0
    koff = 2.5 / T
    feed = nw.pool(kP=kP, c0=c0, kd2=kd2, feed=lambda t: koff * c0 * np.exp(-koff * t))
    rel = nw.pool(kP=kP, c0=c0, kd2=kd2, start="R")
    rel.add_reaction("R -> X", k=koff, id="release")
    check("exponential feed == slow-release reservoir", ton_T(feed, T, max_step=T / 200), ton_T(rel, T), tol=1e-5)


def test_dosing_bound_and_efficiencies(check):
    """No strategy beats the bound; fast reservoir -> g* = 0.805, constant feed -> 1 at large beta."""
    c0, T = 1e-3, 1e4
    for beta in (10.0, 1e2, 1e3, 1e4, 1e6):
        kd2 = beta / (c0 * T)
        bound = cf.dosing_bound_P(kP, 0.0, kd2, c0, T)
        Dst = cf.D_opt_second_order(kd2, c0, T)
        for P in (cf.ton_T_mixed(kP, 0.0, kd2, c0, T, D=1.0) * c0, cf.ton_T_mixed(kP, 0.0, kd2, c0, T, D=Dst) * c0,
                  cf.slow_release_opt(kP, kd2, c0, T)[1], cf.const_feed_P(kP, kd2, c0, T)):
            assert P <= bound * (1 + 1e-12)
        if beta >= 1e2:
            check(f"fast-reservoir efficiency = g* (beta {beta:g})",
                  cf.ton_T_mixed(kP, 0.0, kd2, c0, T, D=Dst) * c0 / bound, cf.gstar(), tol=1e-9)
    kd2 = 1e6 / (c0 * T)
    check("constant-feed efficiency -> 1 (beta 1e6)", cf.const_feed_P(kP, kd2, c0, T) / cf.dosing_bound_P(kP, 0.0, kd2, c0, T), 1.0)
    # dosing bound reached by the ODE: constant feed at beta = 1e4 within 1 %
    kd2 = 1e4 / (c0 * T)
    net = nw.pool(kP=kP, c0=c0, kd2=kd2, feed=lambda t: c0 / T)
    check("ODE constant feed vs dosing bound (beta 1e4)", ton_T(net, T, max_step=T / 200) * c0,
          cf.dosing_bound_P(kP, 0.0, kd2, c0, T), tol=1.5e-2)
