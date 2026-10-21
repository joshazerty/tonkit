"""Stochastic simulation (Gillespie) of catalyst lifetimes, with per-molecule turnover counts.

Two engines:

ssa_linear(chain, nmol)   independent molecules on the single-molecule chain of a linear network
                          (vectorised numpy). Time-dependent rates by thinning. This is the direct
                          test of the T1 no-go theorem: the law of N with and without reservoirs.

ssa_agents(net, n0)       interacting molecules (direct method) for networks with steps of second
                          order in catalyst (2X -> D2, X + R -> 2D, 2X <-> X2 is NOT supported:
                          live species must carry one metal). Each catalyst molecule is tracked as
                          an agent, so the per-molecule TON distribution is available in the
                          nonlinear regime too. System size Omega = n0/c0 molecules per M.
                          Compiled with numba if available (pure-Python fallback is slow).
"""
from __future__ import annotations

import numpy as np

try:
    import numba
    _njit = numba.njit(cache=True)
except Exception:  # pragma: no cover
    numba = None
    _njit = lambda f: f


# ============================================================================ linear (independent)
def ssa_linear(ch, nmol, seed=0, tdep=None, start=None, max_events=10**7):
    """Simulate nmol independent molecules on a Chain (markov.linear_chain).

    tdep: {(from_state, to_state): (k_of_t, kmax)} time-dependent rates, by thinning.
    start: initial state name (default: drawn from ch.p0).
    Returns dict(N = products per molecule, dead = index of final state, t = death time)."""
    rng = np.random.default_rng(seed)
    K = ch.K.copy()
    thin = []
    if tdep:
        for (a, b), (f, kmax) in tdep.items():
            ia, ib = ch.idx(a), ch.idx(b)
            K[ia, ib] = kmax
            thin.append((ia, ib, f, kmax))
    q = K.sum(1)
    P = np.zeros_like(K)
    nz = q > 0
    P[nz] = K[nz] / q[nz, None]
    cum = np.cumsum(P, axis=1)
    cum[nz, -1] = 1.0 + 1e-12
    NU = ch.NU.astype(np.int64)
    if start is None:
        state = rng.choice(len(ch.states), size=nmol, p=ch.p0)
    else:
        state = np.full(nmol, ch.idx(start))
    N = np.zeros(nmol, dtype=np.int64)
    t = np.zeros(nmol)
    alive = q[state] > 0
    events = 0
    while alive.any():
        ia = np.nonzero(alive)[0]
        s = state[ia]
        t[ia] += rng.exponential(1.0 / q[s])
        new = (rng.random(ia.size)[:, None] < cum[s]).argmax(1)
        for a, b, f, kmax in thin:
            prop = (s == a) & (new == b)
            if prop.any():
                rej = np.nonzero(prop)[0][rng.random(prop.sum()) >= f(t[ia][prop]) / kmax]
                new[rej] = a
        N[ia] += NU[s, new]
        state[ia] = new
        alive[ia] = q[new] > 0
        events += ia.size
        if events > max_events * nmol:
            raise RuntimeError("too many events; is there a dead end?")
    return {"N": N, "dead": state, "t": t, "states": ch.states}


# ============================================================================ agents (interacting)
def compile_agents(net):
    """Translate a Network into arrays for the agent SSA (see module docstring for restrictions)."""
    net.validate(strict=True)
    st = [s.name for s in net.species if s.catalyst or s.dead]
    pos = {n: i for i, n in enumerate(st)}
    for n in st:
        s = net.sp(n)
        if s.catalyst and s.metal != 1:
            raise ValueError(f"agent SSA: live species {n!r} carries {s.metal} metals (unsupported)")
    ks = net.rate_constants()
    prod = net.product
    rows = []
    for r, k in zip(net.reactions, ks):
        if r.orders:
            raise ValueError(f"agent SSA: non-stoichiometric rate law in {r.id!r} not supported")
        if callable(k):
            raise ValueError(f"agent SSA: time-dependent k in {r.id!r} not supported")
        cin = []
        for m, v in r.reactants.items():
            if m in pos:
                cin += [pos[m]] * int(v)
        cout = []
        for m, v in r.products.items():
            if m in pos:
                cout += [pos[m]] * int(v * net.sp(m).metal)
        if not cin and not cout:
            for m in list(r.reactants) + list(r.products):
                if not net.sp(m).fixed and m != prod:
                    raise ValueError(f"agent SSA: catalyst-free step {r.id!r} changes non-fixed {m!r}")
            continue
        if len(cin) not in (1, 2) or len(cout) != len(cin):
            raise ValueError(f"agent SSA: step {r.id!r} must move 1 or 2 metal agents ({cin} -> {cout})")
        keff = float(k)
        for m, v in r.reactants.items():
            if m in pos:
                continue
            s = net.sp(m)
            if not s.fixed:
                raise ValueError(f"agent SSA: co-reactant {m!r} in {r.id!r} must be fixed")
            keff *= s.conc0 ** v
        nu_p = r.products.get(prod, 0) - r.reactants.get(prod, 0)
        rows.append((len(cin), cin[0], cin[-1], cout[0], cout[-1], keff, nu_p / len(cin)))
    R = len(rows)
    arr = lambda i, dt: np.array([row[i] for row in rows], dtype=dt)
    dead = np.array([net.sp(n).dead for n in st])
    c = np.array([net.sp(n).conc0 if net.sp(n).catalyst else 0.0 for n in st])
    return dict(states=st, m=arr(0, np.int64), in0=arr(1, np.int64), in1=arr(2, np.int64),
                out0=arr(3, np.int64), out1=arr(4, np.int64), keff=arr(5, float), credit=arr(6, float),
                dead=dead, c=c, c0=net.c0_total(), R=R)


@_njit
def _agent_loop(m, in0, in1, out0, out1, keff, credit, dead, init_state, Omega, t_max, seed,
                t_grid, live_grid, prod_grid):
    np.random.seed(seed)
    nst = dead.shape[0]
    n0 = init_state.shape[0]
    R = m.shape[0]
    members = np.empty((nst, n0), dtype=np.int64)
    count = np.zeros(nst, dtype=np.int64)
    slot = np.empty(n0, dtype=np.int64)
    state = init_state.copy()
    for i in range(n0):
        s = state[i]
        members[s, count[s]] = i
        slot[i] = count[s]
        count[s] += 1
    N = np.zeros(n0)
    tdeath = np.full(n0, np.inf)
    alive = 0
    for i in range(n0):
        if not dead[state[i]]:
            alive += 1
        else:
            tdeath[i] = 0.0
    a = np.zeros(R)
    t = 0.0
    total_prod = 0.0
    ig = 0
    ng = t_grid.shape[0]
    nev = 0
    while alive > 0:
        a0 = 0.0
        for j in range(R):
            if m[j] == 1:
                a[j] = keff[j] * count[in0[j]]
            elif in0[j] == in1[j]:
                n = count[in0[j]]
                a[j] = keff[j] * n * (n - 1) / Omega
            else:
                a[j] = keff[j] * count[in0[j]] * count[in1[j]] / Omega
            a0 += a[j]
        if a0 <= 0.0:
            break
        dt = -np.log(np.random.random()) / a0
        while ig < ng and t_grid[ig] < t + dt:
            live_grid[ig] = alive
            prod_grid[ig] = total_prod
            ig += 1
        t += dt
        if t > t_max:
            break
        u = np.random.random() * a0
        j = 0
        acc = a[0]
        while acc < u and j < R - 1:
            j += 1
            acc += a[j]
        # pick agents
        s0 = in0[j]
        k0 = np.random.randint(count[s0])
        ag0 = members[s0, k0]
        ag1 = -1
        if m[j] == 2:
            s1 = in1[j]
            if s1 == s0:
                k1 = np.random.randint(count[s0] - 1)
                if k1 >= k0:
                    k1 += 1
                ag1 = members[s0, k1]
            else:
                ag1 = members[s1, np.random.randint(count[s1])]
        # move agents
        for which in range(m[j]):
            ag = ag0 if which == 0 else ag1
            dest = out0[j] if which == 0 else out1[j]
            src = state[ag]
            # remove from src
            k = slot[ag]
            last = members[src, count[src] - 1]
            members[src, k] = last
            slot[last] = k
            count[src] -= 1
            # add to dest
            members[dest, count[dest]] = ag
            slot[ag] = count[dest]
            count[dest] += 1
            state[ag] = dest
            N[ag] += credit[j]
            total_prod += credit[j]
            if dead[dest] and not dead[src]:
                alive -= 1
                tdeath[ag] = t
        nev += 1
    while ig < ng:
        live_grid[ig] = alive
        prod_grid[ig] = total_prod
        ig += 1
    return N, state, tdeath, t, nev


def ssa_agents(net, n0, seed=0, t_max=np.inf, t_grid=None):
    """Agent-based SSA with n0 catalyst molecules (system size Omega = n0/c0).

    Returns dict(N per molecule, dead/final state per molecule, t = death time per molecule,
    states, events, t_end, Omega, and live/prod time series on t_grid if given)."""
    cp = compile_agents(net)
    c = cp["c"]
    frac = c / c.sum()
    counts = np.floor(frac * n0).astype(int)
    counts[np.argmax(frac)] += n0 - counts.sum()
    init = np.repeat(np.arange(len(c)), counts).astype(np.int64)
    Omega = n0 / cp["c0"]
    tg = np.asarray(t_grid if t_grid is not None else [], dtype=float)
    lg = np.zeros(len(tg)); pg = np.zeros(len(tg))
    N, state, td, t_end, nev = _agent_loop(cp["m"], cp["in0"], cp["in1"], cp["out0"], cp["out1"],
                                           cp["keff"], cp["credit"], cp["dead"], init, float(Omega),
                                           float(t_max), int(seed) % (2**31), tg, lg, pg)
    out = {"N": N, "dead": state, "t": td, "states": cp["states"], "events": nev, "t_end": t_end,
           "Omega": Omega, "n0": n0}
    if t_grid is not None:
        out.update(t_grid=tg, live=lg / n0, TON=pg / n0)
    return out
