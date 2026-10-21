"""T3 - order rule: reservoir-dilution correspondence, mixed order, decaying reservoir,
cross-deactivation, dimeric reservoir, bimetallic turnover (SI S2)."""
import numpy as np
import pytest

from tonkit import ton_inf, simulate, P_of_t
from tonkit import networks as nw
from tonkit import closed_forms as cf

KD1 = 1e-5          # first-order death; ceiling kP/kd1 ~ 1e5 keeps every point at TON >> 1


def cf_for(net, kd1, kd2, c0=1e-3, **kw):
    m = net.meta
    return cf.ton_inf_mixed(m["kP"], kd1, kd2, c0, D=m["D"], fX=m["fX"], **kw)


@pytest.mark.parametrize("src,att", [("B", "B"), ("A", "B"), ("B", "A")])
@pytest.mark.parametrize("rho", [0.1, 10.0, 1e3])
def test_mixed_order_ton_inf(check, src, att, rho):
    c0 = 1e-3
    fX = nw.cycle(src=src).meta["fX"]
    kd1 = KD1 / fX                      # same first-order ceiling kP/(kd1 fX) for both sources
    kd2 = rho * kd1 / (fX * c0)         # rho = kd2 fX c0/kd1, second-order share at D = 1
    for K in (0.0, 1.0, 10.0, 100.0, 1000.0):
        net = nw.cycle(src=src, att=att, K_R=K, kd1=kd1, kd2=kd2, c0=c0)
        check(f"mixed order TON_inf src {src} att {att} rho {rho:g} K_R {K:g}",
              ton_inf(net, half_time=False).TON, cf_for(net, kd1, kd2))


def test_mixed_order_gain_ceiling(check):
    """D -> infinity: G -> rho/ln(1 + rho) (a reservoir can at most reach the first-order ceiling)."""
    c0, rho = 1e-3, 100.0
    fX = nw.cycle().meta["fX"]
    kd2 = rho * KD1 / (fX * c0)
    t1 = ton_inf(nw.cycle(K_R=0.0, kd1=KD1, kd2=kd2), half_time=False).TON
    tD = ton_inf(nw.cycle(K_R=1e6, kd1=KD1, kd2=kd2, koff=1e2), half_time=False).TON
    check("gain at K_R = 1e6 vs rho/ln(1+rho)", tD / t1, cf.gain_mixed_ceiling(rho * fX / fX), tol=1e-2)


@pytest.mark.parametrize("K", [3.0, 30.0])
def test_reservoir_dilution_correspondence(check, K):
    """P_res(t; c0, D) = D * P(t/D; c0/D, no reservoir), exact for a fast reservoir."""
    c0 = 1e-3
    fB = nw.cycle().meta["fB"]
    kd2 = 100 * KD1 / (fB * c0)
    res = nw.cycle(K_R=K, kd1=KD1, kd2=kd2, c0=c0)
    D = res.meta["D"]
    dil = nw.cycle(K_R=0.0, kd1=KD1, kd2=kd2, c0=c0 / D)
    ts = np.array([1e1, 1e2, 1e3, 1e4, 1e5, 1e6])
    Pr = P_of_t(res, ts)
    Pd = P_of_t(dil, ts / D)
    for t, a, b in zip(ts, Pr, D * Pd):
        check(f"correspondence K_R {K:g} t {t:g}", a, b, tol=1e-4)


def test_decaying_reservoir(check):
    c0, kd1, kd2 = 1e-3, 1e-6, 10.0
    fB = nw.cycle().meta["fB"]
    for kdR in (1e-4, 1e-7):
        Dopt = cf.D_opt_decaying(nw.cycle().meta["kP"], kd1, kd2, c0, kdR, fX=fB)
        for D in (1.0, 0.5 * Dopt, Dopt, 2 * Dopt):
            if D < 1:
                continue
            net = nw.cycle(K_R=(D - 1) / fB, kd1=kd1, kd2=kd2, kdR=kdR, c0=c0)
            check(f"decaying reservoir kdR {kdR:g} D {D:.3g}", ton_inf(net, half_time=False).TON,
                  cf_for(net, kd1, kd2, kdR=kdR))
    # approximation D* ~ x* sqrt(kd2 fX^2 c0/kdR) - 1/2 when kdR << kd2 c0 and kd1 -> 0
    x, _ = cf.xstar()
    Dopt = cf.D_opt_decaying(1.0, 1e-12, kd2, c0, 1e-7, fX=fB)
    check("D*_decaying vs 0.505 sqrt(beta_R) - 1/2", Dopt, x * np.sqrt(kd2 * fB**2 * c0 / 1e-7) - 0.5, tol=1e-2)


def test_decaying_reservoir_benefit_condition():
    """Initial-benefit test kdR/kd1~ < (1+rho) ln(1+rho)/rho - 1: sign of dTON/dD at D = 1."""
    c0, kd1, kP = 1e-3, 1e-5, 1.0
    for rho in (1.0, 100.0):
        kd2 = rho * kd1 / c0
        thr = (1 + rho) * np.log1p(rho) / rho - 1
        for f in (0.5, 2.0):
            kdR = f * thr * kd1
            slope = cf.ton_inf_mixed(kP, kd1, kd2, c0, D=1.001, kdR=kdR) - cf.ton_inf_mixed(kP, kd1, kd2, c0, D=1.0, kdR=kdR)
            assert (slope > 0) == cf.decaying_reservoir_helps(kdR, kd1, rho)


@pytest.mark.parametrize("ratio", [0.0, 0.25, 0.5, 0.75])
def test_cross_deactivation(check, ratio):
    c0 = 1e-3
    fB = nw.cycle().meta["fB"]
    kd2 = 1e2 * KD1 / (fB * c0)
    kx = ratio * kd2 * fB
    tons = []
    for K in (0.0, 10.0, 100.0, 1000.0):
        net = nw.cycle(K_R=K, kd1=KD1, kd2=kd2, kx=kx, c0=c0)
        t = ton_inf(net, half_time=False).TON
        tons.append(t)
        check(f"cross-deactivation kx/(kd2 fX) {ratio} K_R {K:g}", t, cf_for(net, KD1, kd2, kx=kx))
    # threshold kx = kd2 fX/2: below it the reservoir helps, above it it hurts
    if ratio < 0.5:
        assert tons[-1] > tons[0]
    elif ratio > 0.5:
        assert tons[-1] < tons[0]


@pytest.mark.parametrize("c0", [1e-4, 1e-3, 1e-2])
def test_dimeric_reservoir(check, c0):
    net = nw.dimer_reservoir(kd1=KD1, kd2=10.0, Kdim=1e5, c0=c0)
    check(f"dimeric reservoir c0 {c0:g}", ton_inf(net, half_time=False).TON,
          cf.ton_dimer_reservoir(1.0, KD1, 10.0, 1e5, c0))


def test_dimeric_reservoir_slope():
    sl = np.diff(np.log([cf.ton_dimer_reservoir(1.0, 1e-12, 10.0, 1e5, c) for c in (1e0, 1e1)]))[0] / np.log(10)
    assert abs(sl + 0.5) < 0.02


@pytest.mark.parametrize("Kdim", [0.0, 1e3, 1e6])
def test_dimeric_reservoir_first_order_nogo(check, Kdim):
    net = nw.dimer_reservoir(kd1=KD1, kd2=0.0, Kdim=Kdim, c0=1e-3)
    check(f"dimeric reservoir, first-order death only, K_dim {Kdim:g}", ton_inf(net, half_time=False).TON, 1.0 / KD1)


@pytest.mark.parametrize("K", [0.0, 9.0])
def test_bimetallic_turnover(check, K):
    net = nw.bimetallic(kP2=1e3, kd1=KD1, K_R=K, c0=1e-3)
    check(f"bimetallic turnover K_R {K:g}", ton_inf(net, half_time=False).TON,
          cf.ton_bimetallic(1e3, KD1, 1e-3, D=1 + K))
