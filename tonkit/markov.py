"""Single-molecule Markov chain of a *linear* network (every step first order in catalyst), T1.

When every catalyst step has exactly one catalyst molecule on each side and every co-reactant is
held fixed (catalyst-limited conditions), one catalyst molecule performs a continuous-time
Markov chain on the catalyst states, absorbed in the dead states. N = number of product molecules
it makes before it dies. This module gives the exact law of N, the dead-end split and the mean
times (T1 corollary), with which the ODE TON_inf (= <N>) and the Gillespie runs are compared.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class NonlinearNetworkError(ValueError):
    pass


@dataclass
class Chain:
    states: list          # catalyst + dead species names
    K: np.ndarray         # K[x, y] = rate x -> y (s^-1), pseudo-first order
    NU: np.ndarray        # product molecules released on the jump x -> y
    p0: np.ndarray        # initial distribution over states (from conc0 of catalyst species)
    absorbing: np.ndarray # bool mask of dead states

    @property
    def transient(self):
        return np.nonzero(~self.absorbing)[0]

    def idx(self, name):
        return self.states.index(name)


def linear_chain(net, t=None):
    """Build the single-molecule chain from a Network. Raises NonlinearNetworkError if a step has
    two catalyst molecules on one side or a non-fixed co-reactant. Time-dependent rate constants are
    evaluated at t (or rejected if t is None)."""
    net.validate(strict=True)
    states = [s.name for s in net.species if s.catalyst or s.dead]
    pos = {n: i for i, n in enumerate(states)}
    n = len(states)
    K = np.zeros((n, n)); NU = np.zeros((n, n))
    ks = net.rate_constants()
    prod = net.product
    for r, k in zip(net.reactions, ks):
        if r.orders:
            raise NonlinearNetworkError(f"{r.id}: non-stoichiometric rate law ('orders') not supported")
        if callable(k):
            if t is None:
                raise NonlinearNetworkError(f"{r.id}: time-dependent k; pass t or use ssa_linear(..., tdep=...)")
            k = k(t)
        cin = [(m, v) for m, v in r.reactants.items() if m in pos]
        cout = [(m, v) for m, v in r.products.items() if m in pos]
        if not cin and not cout:
            continue
        if len(cin) != 1 or cin[0][1] != 1 or len(cout) != 1 or cout[0][1] != 1:
            raise NonlinearNetworkError(f"{r.id}: not first order in catalyst ({r.reactants} -> {r.products})")
        kk = float(k)
        for m, v in r.reactants.items():
            if m in pos:
                continue
            s = net.sp(m)
            if not s.fixed:
                raise NonlinearNetworkError(f"{r.id}: co-reactant {m!r} is not fixed (time-dependent branching)")
            kk *= s.conc0 ** v
        a, b = pos[cin[0][0]], pos[cout[0][0]]
        if net.sp(states[b]).metal != net.sp(states[a]).metal:
            raise NonlinearNetworkError(f"{r.id}: metal count changes")
        K[a, b] += kk
        nu_p = r.products.get(prod, 0) - r.reactants.get(prod, 0)
        if nu_p:
            if K[a, b] > kk and NU[a, b] != nu_p:
                raise NonlinearNetworkError(f"{r.id}: parallel jumps {states[a]}->{states[b]} with different product counts")
            NU[a, b] = nu_p
    # self-loops (e.g. X -> X + P in a reduced model) are kept: they release product without a state change
    absorbing = np.array([net.sp(s).dead for s in states])
    if (K[absorbing].sum(1) > 0).any():
        raise NonlinearNetworkError("a dead species has an outgoing step (reactivation): not absorbing")
    c = np.array([net.sp(s).conc0 if net.sp(s).catalyst else 0.0 for s in states])
    p0 = c / c.sum() if c.sum() > 0 else np.eye(n)[0]
    return Chain(states, K, NU, p0, absorbing)


def _jump_split(ch):
    q = ch.K.sum(1)
    tr = ch.transient
    ab = np.nonzero(ch.absorbing)[0]
    P = np.zeros_like(ch.K)
    P[tr] = ch.K[tr] / q[tr, None]
    M0 = np.where(ch.NU[np.ix_(tr, tr)] == 0, P[np.ix_(tr, tr)], 0.0)
    M1 = np.where(ch.NU[np.ix_(tr, tr)] != 0, P[np.ix_(tr, tr)], 0.0)
    Bm = P[np.ix_(tr, ab)]
    if (ch.NU[np.ix_(tr, tr)] > 1).any() or (ch.NU[np.ix_(tr, ab)] != 0).any():
        raise NotImplementedError("exact law implemented for unit product release on transient jumps")
    return tr, ab, M0, M1, Bm, q


def N_distribution(ch, nmax=None, tail=1e-14):
    """Exact P(N = n), n = 0..nmax: P(N=n) = p0 [(I-M0)^-1 M1]^n (I-M0)^-1 b.
    If nmax is None, continues until the remaining mass is below tail (cap 10^6)."""
    tr, ab, M0, M1, Bm, q = _jump_split(ch)
    F = np.linalg.inv(np.eye(len(tr)) - M0)
    G = F @ M1
    fb = F @ Bm.sum(1)
    w = ch.p0[tr].copy()
    out = []
    mass = 0.0
    nlim = nmax if nmax is not None else 10**6
    for _ in range(nlim + 1):
        p = float(w @ fb)
        out.append(p); mass += p
        if nmax is None and 1 - mass < tail:
            break
        w = w @ G
    return np.array(out)


def mean_N(ch):
    """Exact <N> = TON_inf of one molecule (= ODE TON_inf for linear networks)."""
    tr, ab, M0, M1, Bm, q = _jump_split(ch)
    M = M0 + M1
    reward = M1.sum(1)
    return float(ch.p0[tr] @ np.linalg.solve(np.eye(len(tr)) - M, reward))


def dead_split(ch):
    """Probability of ending in each dead state."""
    tr, ab, M0, M1, Bm, q = _jump_split(ch)
    v = ch.p0[tr] @ np.linalg.inv(np.eye(len(tr)) - M0 - M1) @ Bm
    return {ch.states[a]: float(x) for a, x in zip(ab, v)}


def mean_times(ch):
    """Expected total time spent in each transient state before death."""
    tr = ch.transient
    q = ch.K.sum(1)
    Q = ch.K[np.ix_(tr, tr)] - np.diag(q[tr])
    T = ch.p0[tr] @ np.linalg.inv(-Q)
    return {ch.states[i]: float(x) for i, x in zip(tr, T)}


def mean_lifetime(ch):
    return float(sum(mean_times(ch).values()))
