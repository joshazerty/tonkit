"""T1 - no-go theorem: with every step first order in catalyst, pendant reservoirs leave the whole
law of N (turnovers before death) unchanged and stretch the time only (SI S1, Theorem 1)."""
import numpy as np
import pytest

from tonkit import markov, ton_inf, simulate
from tonkit import networks as nw
from tonkit import closed_forms as cf


def test_exact_N_law_invariant(check):
    ch0 = markov.linear_chain(nw.t1_network(False))
    ch1 = markov.linear_chain(nw.t1_network(True))
    p0 = markov.N_distribution(ch0, 1200)
    p1 = markov.N_distribution(ch1, 1200)
    check("max_n |P0(N=n) - P_res(N=n)|", np.abs(p0 - p1).max(), 0.0, tol=1e-14, mode="abs")
    check("sum_n P(N=n)", p1.sum(), 1.0, tol=1e-12)
    s0, s1 = markov.dead_split(ch0), markov.dead_split(ch1)
    for k in s0:
        check(f"dead-end split {k}", s1[k], s0[k], tol=1e-12)


def test_lifetime_corollary(check):
    """<tau> = <tau0> + sum_a K_a <T_a0>: R1 on C (K1 = 6), R2-R3 chain on B (K_B = 2 (1 + 2) = 6)."""
    ch0 = markov.linear_chain(nw.t1_network(False))
    ch1 = markov.linear_chain(nw.t1_network(True))
    T0 = markov.mean_times(ch0)
    pred = cf.lifetime_with_reservoirs(markov.mean_lifetime(ch0), [6.0, 6.0], [T0["C"], T0["B"]])
    check("<tau> with reservoirs", markov.mean_lifetime(ch1), pred, tol=1e-12)
    check("<tau> = 198.6813 (P1 value)", markov.mean_lifetime(ch1), 198.68131868, tol=1e-9)


@pytest.mark.parametrize("with_res", [False, True])
def test_ode_ton_equals_exact_mean(check, with_res):
    net = nw.t1_network(with_res)
    check(f"ODE TON_inf vs exact <N> (reservoirs={with_res})", ton_inf(net).TON,
          markov.mean_N(markov.linear_chain(net)))


def test_ode_dead_split_matches_markov(check):
    net = nw.t1_network(True)
    r = ton_inf(net)
    split = markov.dead_split(markov.linear_chain(net))
    for k in ("D1", "D2"):
        check(f"ODE dead fraction {k}", r[k][-1] / net.c0_total(), split[k], tol=1e-6)


@pytest.mark.parametrize("src,att", [("A", "A"), ("A", "B"), ("B", "A"), ("B", "B")])
def test_ton_flat_and_time_stretched(check, src, att):
    """Explicit cycle, first-order death from src, reservoir on att: TON_inf = kP/(kd1 fX) for every
    K_R; t_half grows by the depth D = 1 + K_R f_att (fast exchange)."""
    kd1 = 1e-4 / nw.cycle(src=src).meta["fX"]      # same TON (1e4) for both deactivation sources
    base = nw.cycle(src=src, att=att, kd1=kd1)
    ref = base.meta["kP"] / (kd1 * base.meta["fX"])
    th0 = ton_inf(base).t_half
    for K in (1.0, 10.0, 100.0, 1000.0):
        net = nw.cycle(src=src, att=att, K_R=K, kd1=kd1, koff=1e3)
        r = ton_inf(net)
        check(f"TON_inf src {src} att {att} K_R {K:g}", r.TON, ref)
        check(f"t_half/t_half0 = D, src {src} att {att} K_R {K:g}", r.t_half / th0, net.meta["D"])


def test_time_dependent_reservoir_rate(check):
    """Reservoir entry rate k_on(t) = 3 (1 + sin(t/3)): TON_inf still equals the no-reservoir <N>."""
    ref = markov.mean_N(markov.linear_chain(nw.t1_network(False)))
    net = nw.t1_network(True, kon1=lambda t: 3.0 * (1.0 + np.sin(t / 3.0)))
    check("TON_inf with k_on(t)", ton_inf(net, max_step=0.5).TON, ref, tol=1e-6)


def test_single_cycle_geometric(check):
    """Costentin limit: one cycle, death competing with turnover -> N geometric, <N> = (1 - p)/p."""
    net = nw.pool(kP=1.0, kd1=0.01)
    ch = markov.linear_chain(net)
    p = markov.N_distribution(ch, 200)
    pc = 0.01 / 1.01
    n = np.arange(201)
    check("geometric law", np.abs(p - (1 - pc) ** n * pc).max(), 0.0, tol=1e-15, mode="abs")
    check("<N> = (1-p_c)/p_c", markov.mean_N(ch), cf.ton_single_cycle(pc), tol=1e-12)
    check("ODE TON_inf = kP/kd", ton_inf(net).TON, 100.0)
