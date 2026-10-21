"""T2 - corrections when branching changes in time: the closed batch stays invariant; external
clocks break the no-go result (Psi(z)); a poison made on an external clock gives TON ~ D^-1/2."""
import numpy as np
import pytest

from tonkit import Network, simulate, ton_inf
from tonkit import networks as nw
from tonkit import closed_forms as cf

PARS = dict(k1=1e3, k2=1.0, kdA=0.1, kdB=2e-4, S0=0.1)


def P_closed(net):
    r = ton_inf(net, half_time=False, live_tol=1e-12)
    P = r["P"][-1] + (r["BP"][-1] if "BP" in r.names else 0.0)
    return P


VARIANTS = [(att, kex, full, K) for att in ("B", "A") for kex in (1e3, 1.0, 1e-3)
            for full in (False, True) for K in (1.0, 100.0)]


@pytest.mark.parametrize("c0", [2e-5, 7e-5, 3e-4])
def test_closed_batch_reservoirs(check, c0):
    pred = cf.P_inf_closed_batch(PARS["S0"], c0, PARS["k1"], PARS["k2"], PARS["kdA"], PARS["kdB"])
    base = P_closed(nw.closed_batch(c0=c0, **PARS))
    check(f"closed batch P_inf, no reservoir, c0 {c0:g}", base, pred)
    for att, kex, full, K in VARIANTS:
        P = P_closed(nw.closed_batch(c0=c0, att=att, K_R=K, kex=kex, full=full, **PARS))
        check(f"closed batch P_inf c0 {c0:g} att {att} kex {kex:g} full {full} K_R {K:g}", P, pred)


@pytest.mark.parametrize("c0", [2e-5, 3e-4])
def test_product_inhibition_adduct(check, c0):
    pred = cf.P_inf_closed_batch(PARS["S0"], c0, PARS["k1"], PARS["k2"], PARS["kdA"], PARS["kdB"])
    for K in (0.0, 10.0):
        P = P_closed(nw.closed_batch(c0=c0, K_R=K, KP=1e4, kpoff=1.0, **PARS))
        check(f"product inhibition, c0 {c0:g}, K_R {K:g}", P, pred)


@pytest.mark.parametrize("ks", [1e-6, 1e-5, 1e-4])
def test_external_clock_psi(check, ks):
    """Substrate lost uncatalysed (S -> W, ks); catalyst-limited (c0 tiny); death from A only.
    TON/TON0 = Psi(z), z = ks * tau_life(D), tau_life = TON0 D/(k2 fB)."""
    k1, k2, kdA, S0, c0 = 1e3, 1.0, 0.1, 0.1, 1e-12
    ton0 = k1 * S0 / kdA
    fB = k1 * S0 / (k1 * S0 + k2)
    for K in (0.0, 10.0, 100.0):
        net = nw.closed_batch(k1=k1, k2=k2, kdA=kdA, kdB=0.0, S0=S0, c0=c0, K_R=K, ks=ks)
        r = ton_inf(net, half_time=False, live_tol=1e-12, t_max=1e12)
        D0 = 1 + K * fB
        z = ks * ton0 * D0 / (k2 * fB)
        check(f"Psi(z) clock ks {ks:g} K_R {K:g} (z = {z:.3g})", r.TON, ton0 * cf.psi(z))


def test_poison_clock(check):
    """Poison made at rate g kills free catalyst (kp): TON(D) = TON(1)/sqrt(D)."""
    kP, kp, g, c0 = 1.0, 1e3, 1e-6, 1e-12
    for K in (0.0, 3.0, 15.0, 99.0):
        net = Network("poison")
        net.add_species("X", catalyst=True, conc0=c0 / (1 + K)).add_species("P").add_species("D", dead=True)
        net.add_species("Pz")
        net.add_reaction("X -> X + P", k=kP)
        net.add_reaction("-> Pz", k=g)
        net.add_reaction("X + Pz -> D + Pz", k=kp)
        if K > 0:
            net.add_species("R", catalyst=True, conc0=c0 * K / (1 + K))
            net.add_reaction("X <-> R", k=K * 1e3, k_rev=1e3)
        r = ton_inf(net, half_time=False, t_max=1e9)
        check(f"poison clock K_R {K:g}", r.TON, cf.ton_poison(kP, kp, g, D=1 + K))
