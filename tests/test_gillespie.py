"""Stochastic engines against exact results (small runs; the production runs live in
03_calculations/02_toy_network_sweeps/p2_gillespie)."""
import numpy as np
from scipy import stats

from tonkit import markov, ton_inf
from tonkit import gillespie as g
from tonkit import networks as nw


def gof_pvalue(N, p_exact, min_expected=20):
    """Chi-square goodness of fit of samples N against an exact law (tail bins pooled)."""
    n = len(N)
    m = len(p_exact)
    h = np.bincount(np.minimum(N, m - 1), minlength=m).astype(float)
    e = p_exact.copy(); e[-1] += max(0.0, 1 - e.sum())
    e *= n
    # pool bins from the right until every expected count >= min_expected
    obs, exp_, acc_o, acc_e = [], [], 0.0, 0.0
    for o, x in zip(h[::-1], e[::-1]):
        acc_o += o; acc_e += x
        if acc_e >= min_expected:
            obs.append(acc_o); exp_.append(acc_e); acc_o = acc_e = 0.0
    if acc_e > 0:
        obs[-1] += acc_o; exp_[-1] += acc_e
    obs, exp_ = np.array(obs), np.array(exp_)
    chi2 = ((obs - exp_) ** 2 / exp_).sum()
    return stats.chi2.sf(chi2, len(obs) - 1)


def test_ssa_linear_matches_exact_law(check):
    for with_res in (False, True):
        ch = markov.linear_chain(nw.t1_network(with_res))
        p = markov.N_distribution(ch)
        out = g.ssa_linear(ch, 20000, seed=11 + with_res)
        N = out["N"]
        mu = markov.mean_N(ch)
        check(f"SSA mean N (reservoirs={with_res}) within 4 SE", abs(N.mean() - mu) / (N.std() / np.sqrt(len(N))),
              0.0, tol=4.0, mode="abs")
        assert gof_pvalue(N, p) > 1e-3
        check(f"SSA mean lifetime (reservoirs={with_res}) within 4 SE",
              abs(out["t"].mean() - markov.mean_lifetime(ch)) / (out["t"].std() / np.sqrt(len(N))), 0.0, tol=4.0, mode="abs")


def test_ssa_linear_thinning():
    """Time-dependent reservoir entry k_on(t) (thinning): law of N still the exact no-reservoir law."""
    ch = markov.linear_chain(nw.t1_network(True))
    p0 = markov.N_distribution(markov.linear_chain(nw.t1_network(False)))
    out = g.ssa_linear(ch, 20000, seed=5, tdep={("C", "R1"): (lambda t: 3.0 * (1 + np.sin(t / 3.0)), 6.0)})
    assert gof_pvalue(out["N"], p0) > 1e-3


def test_agent_ssa_first_order_matches_exact(check):
    """Agent engine on a linear network (reservoir K_R = 10, slow-ish exchange) reproduces <N>."""
    net = nw.cycle(K_R=10.0, koff=0.1, kd1=1e-2)
    N = np.concatenate([g.ssa_agents(net, 4000, seed=20 + i)["N"] for i in range(5)])
    mu = markov.mean_N(markov.linear_chain(net))
    check("agent SSA mean N, first order, within 4 SE", abs(N.mean() - mu) / (N.std() / np.sqrt(len(N))), 0.0,
          tol=4.0, mode="abs")
    assert gof_pvalue(np.rint(N).astype(int), markov.N_distribution(markov.linear_chain(net))) > 1e-3


def test_agent_ssa_second_order_matches_ode(check):
    """Second-order deactivation (2B -> D2): agent-SSA TON tends to the ODE TON_inf (finite-size error
    ~1/n0). Molecules are coupled, so the standard error comes from the spread of independent runs."""
    fB = nw.cycle().meta["fB"]
    for K in (0.0, 10.0):
        net = nw.cycle(K_R=K, koff=0.1, kd1=1e-3, kd2=30 * 1e-3 / (fB * 1e-3))
        runs = np.array([g.ssa_agents(net, 2000, seed=7 + i)["N"].mean() for i in range(8)])
        ode = ton_inf(net).TON
        se = runs.std(ddof=1) / np.sqrt(len(runs))
        check(f"agent SSA TON vs ODE TON_inf, 2nd order, K_R {K:g} (|z|, 8 runs x 2000)",
              abs(runs.mean() - ode) / se, 0.0, tol=4.0, mode="abs")
        check(f"agent SSA TON vs ODE TON_inf, 2nd order, K_R {K:g} (rel)", runs.mean(), ode, tol=2e-2)
