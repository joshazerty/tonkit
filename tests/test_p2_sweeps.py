"""P2 sweep support: n_p = 2 closed forms, scaled integration, energetic-span profile networks."""
import numpy as np
import pytest

from tonkit import ton_inf, ton_T, simulate
from tonkit import networks as nw
from tonkit import closed_forms as cf
from tonkit.profile import Profile, span_network, stationary, energetic_span_tof, ton_control

RT = dict(rtol=1e-6, scale="auto", atol_scaled=1e-11)


# ----------------------------------------------------------------------------- n_p = 2
@pytest.mark.parametrize("K", [0.0, 3.0, 30.0, 300.0])
@pytest.mark.parametrize("rho", [0.01, 1.0, 100.0])
def test_bimetallic_mixed_ton_inf(check, K, rho):
    """n_p = 2 with mixed-order death: closed form vs ODE. rho = kd2 c0/kd1 at D = 1."""
    kP2, kd1, c0 = 1e3, 1e-5, 1e-3
    kd2 = rho * kd1 / c0
    net = nw.bimetallic(kP2=kP2, kd1=kd1, kd2=kd2, K_R=K, c0=c0)
    check(f"n_p=2 mixed TON_inf K_R {K:g} rho {rho:g}", ton_inf(net, half_time=False).TON,
          cf.ton_inf_bimetallic_mixed(kP2, kd1, kd2, c0, D=1 + K))


@pytest.mark.parametrize("K", [0.0, 5.0, 100.0])
def test_bimetallic_pure_second_order_no_go(check, K):
    """n_p = n_d = 2: TON_inf = kP2/kd2 for every depth (the order rule's no-go case)."""
    kP2, kd2, c0 = 1e3, 40.0, 1e-3
    net = nw.bimetallic(kP2=kP2, kd1=0.0, kd2=kd2, K_R=K, c0=c0)
    res = simulate(net, 1e12, stop_dead=1e-9)[0]
    # c ~ c0/(1 + ...) never reaches zero in finite time with pure 2nd order; use TON(T) for large T
    ton = ton_T(net, 1e9)
    check(f"n_p=n_d=2 TON(T=1e9) K_R {K:g} (limit kP2/kd2)", ton, cf.ton_inf_bimetallic_mixed(kP2, 0.0, kd2, c0, D=1 + K), tol=2e-2)


@pytest.mark.parametrize("K", [0.0, 10.0])
def test_bimetallic_mixed_horizon(check, K):
    kP2, kd1, kd2, c0, T = 1e3, 1e-4, 50.0, 1e-3, 1e4
    net = nw.bimetallic(kP2=kP2, kd1=kd1, kd2=kd2, K_R=K, c0=c0, koff=1e3)
    check(f"n_p=2 mixed TON(T) K_R {K:g}", ton_T(net, T), cf.ton_T_bimetallic_mixed(kP2, kd1, kd2, c0, T, D=1 + K))


# ----------------------------------------------------------------------------- scaled integration
def test_scaled_equals_unscaled(check):
    net = nw.cycle(K_R=30.0, kd1=1e-5, kd2=1e3, c0=1e-3)
    a = ton_inf(net, half_time=True)
    b = ton_inf(net, half_time=True, scale="auto", atol_scaled=1e-12, live_tol=1e-9)
    check("TON_inf scaled vs unscaled", b.TON, a.TON, tol=1e-5)
    check("t_half scaled vs unscaled", b.t_half, a.t_half, tol=1e-4)


# ----------------------------------------------------------------------------- energetic-span profile
def test_engine_tof_equals_kozuch_shaik(check):
    p = Profile()
    net = span_network(p, reservoir=False)
    _, tof = stationary(net)
    check("engine TOF vs Kozuch-Shaik energetic span", tof, energetic_span_tof(p), tol=1e-6)
    for sh in (-2.0, 3.0):                                     # move the TDI and the TDTS
        q = p.copy(); q.G["I2"] = p.G["I2"] + sh; q.TS["TS23"] = p.TS["TS23"] - sh / 2
        check(f"TOF after shifting I2 by {sh}", stationary(span_network(q, reservoir=False))[1], energetic_span_tof(q), tol=1e-6)


@pytest.mark.parametrize("GR", [-8.0, -10.0, -11.0])
def test_profile_reservoir_slows_tof_by_depth(check, GR):
    p0 = Profile(); p = Profile(G_R=GR)
    pi, tof = stationary(span_network(p))
    pi0, tof0 = stationary(span_network(p0, reservoir=False))
    D = 1 + p.K_R * pi0["I2"]
    check(f"TOF ratio = 1/D (G_R {GR})", tof0 / tof, D, tol=1e-6)
    check(f"R population = K_R theta (G_R {GR})", pi["R"] / (1 - pi["R"]) * (1 + 0), p.K_R * pi0["I2"], tol=1e-6)


def test_profile_first_order_no_go(check):
    """First-order death only: TON_inf is the same with and without a reservoir (Theorem 1), t_half grows by D."""
    p0 = Profile(); res = {}
    for GR in (None, -8.0, -10.0):
        p = Profile(G_R=GR)
        r = ton_inf(span_network(p, second_order=False), **RT, live_tol=1e-9)
        res[GR] = r
    check("TON_inf, reservoir -8 vs none (first order)", res[-8.0].TON, res[None].TON, tol=2e-4)
    check("TON_inf, reservoir -10 vs none (first order)", res[-10.0].TON, res[None].TON, tol=2e-4)
    D8 = 1 + Profile(G_R=-8.0).K_R * stationary(span_network(p0, reservoir=False))[0]["I2"]
    check("t_half ratio = D (first order, G_R -8)", res[-8.0].t_half / res[None].t_half, D8, tol=5e-3)


@pytest.mark.parametrize("GR", [None, -7.0, -9.0, -11.0])
def test_profile_mixed_order_vs_closed_form(check, GR):
    """Energetic-span profile with first- and second-order death vs T3 closed form built from the profile's own
    populations: kP = TOF0, kd1 fX with fX = population of the first-order source, second-order source I2."""
    p0 = Profile()
    pi0, tof0 = stationary(span_network(p0, reservoir=False))
    p = Profile(G_R=GR)
    net = span_network(p)
    meta = net.meta
    # two sources (I1 first order, I2 second order): per-pool constants
    kd1_eff = meta["kd1"] * pi0["I1"]
    kd2_eff = meta["kd2_consumption"] * pi0["I2"] ** 2
    D = 1 + p.K_R * pi0["I2"]
    c0 = 1e-3
    ref = cf.ton_inf_mixed(tof0, kd1_eff, kd2_eff, c0, D=D)
    check(f"profile TON_inf vs closed form (G_R {GR})", ton_inf(net, **RT, live_tol=1e-9).TON, ref, tol=5e-3)


def test_profile_degree_of_ton_control_first_order():
    """Linear (first-order death) profile: X_TON = 0 for every intermediate incl. the reservoir; sum over TS = 0."""
    p = Profile(G_R=-9.0)
    # the engine's integration noise (~1e-5 relative) sets the scale of 'zero' here
    X = ton_control(p, step=0.25, second_order=False, rtol=1e-6)
    for key, v in X.items():
        if key.startswith("G:"):
            assert abs(v) < 5e-3, (key, v)
    assert abs(sum(v for k, v in X.items() if k.startswith("TS:"))) < 1e-2
