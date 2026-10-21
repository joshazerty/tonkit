"""Energetic-span profiles with a reservoir and deactivation branches (P2 sweeps, item S5).

A free-energy profile (species G, transition-state G_ts, kcal/mol, Eyring constants, run-mkm style) is
turned into a tonkit Network: a four-intermediate cycle

    I0 + S -> I1 -> I2 -> I3 -> I0 + P            (S held at 1 M; every step reversible)

plus an off-cycle reservoir  att <-> R  (pendant, entered and left only through `att`), a first-order
deactivation  src1 -> D  and a second-order one  2 src2 -> D2.  All energies are relative to I0 + S.

Helpers
    span_network         build the network from a `Profile`
    stationary           stationary populations of the live, death-free network and the TOF per catalyst
    equilibrated_start   set conc0 of the live species to c0 * (stationary populations)
    energetic_span_tof   Kozuch-Shaik closed form (independent of the engine) for the cycle without reservoir
    ton_control          degree of TON control X_TON,i = -RT d ln TON / d G_i by central differences
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from .network import Network, KB_OVER_H, R_J, ENERGY_TO_J
from .ode import ton_inf, ton_T


@dataclass
class Profile:
    """Free-energy profile of the model (kcal/mol, relative to I0 + S)."""
    G: dict = field(default_factory=lambda: dict(I0=0.0, I1=-4.0, I2=-6.0, I3=-2.0))
    TS: dict = field(default_factory=lambda: dict(TS01=8.0, TS12=6.0, TS23=12.0, TS30=4.0))
    dG_rxn: float = -15.0          # free energy of S -> P; sets G(I0 + P) = G(I0) + dG_rxn
    G_R: float | None = None       # reservoir free energy (None = no reservoir)
    att: str = "I2"                # state the reservoir hangs on
    barrier_R: float = 9.0         # barrier for att -> R measured from att (k_on is then depth independent)
    src1: str = "I1"               # first-order deactivation source (None = no first-order channel)
    dG1: float | None = 24.0       # barrier src1 -> D, from src1
    src2: str = "I2"               # second-order deactivation source (None = no second-order channel)
    dGdim: float | None = 20.0     # barrier 2 src2 -> D2 (event convention, from 2 G[src2])
    T: float = 298.15

    def __post_init__(self):
        """Fix the absolute TS energies of the off-cycle channels once (so that moving an intermediate later
        keeps every transition state where it is, as the degree of TON control requires)."""
        if self.dG1 is not None and self.src1 and "TSd1" not in self.TS:
            self.TS["TSd1"] = self.G[self.src1] + self.dG1
        if self.dGdim is not None and self.src2 and "TSdim" not in self.TS:
            self.TS["TSdim"] = 2 * self.G[self.src2] + self.dGdim
        if "TSR" not in self.TS:
            self.TS["TSR"] = self.G[self.att] + self.barrier_R

    def copy(self, **kw):
        return replace(self, G=dict(self.G), TS=dict(self.TS), **kw)

    def with_barriers(self, dG1=None, dGdim=None, barrier_R=None):
        """Copy with the deactivation / exchange barriers reset from the *current* intermediate energies."""
        q = self.copy()
        if dG1 is not None:
            q.dG1 = dG1; q.TS["TSd1"] = q.G[q.src1] + dG1
        if dGdim is not None:
            q.dGdim = dGdim; q.TS["TSdim"] = 2 * q.G[q.src2] + dGdim
        if barrier_R is not None:
            q.barrier_R = barrier_R; q.TS["TSR"] = q.G[q.att] + barrier_R
        return q

    @property
    def RT(self):                  # kcal/mol
        return R_J * self.T / ENERGY_TO_J["kcal/mol"]

    @property
    def K_R(self):
        return 0.0 if self.G_R is None else float(np.exp(-(self.G_R - self.G[self.att]) / self.RT))


_STEPS = [("TS01", "I0", "I1"), ("TS12", "I1", "I2"), ("TS23", "I2", "I3"), ("TS30", "I3", "I0")]


def span_network(p: Profile, c0: float = 1e-3, start: str = "eq", reservoir: bool | None = None,
                 first_order: bool = True, second_order: bool = True, name: str | None = None) -> Network:
    """Build the Network for profile `p` (see module docstring). start: 'eq' (live catalyst at the stationary
    populations of the death-free network, incl. reservoir), 'I0' (all catalyst in I0), 'R' (all in R)."""
    use_res = (p.G_R is not None) if reservoir is None else reservoir
    net = Network(name or "span_profile", T=p.T, energy_unit="kcal/mol", standard_state=1.0)
    for n in ("I0", "I1", "I2", "I3"):
        net.add_species(n, catalyst=True, G=p.G[n])
    if use_res:
        net.add_species("R", catalyst=True, G=p.G_R)
    net.add_species("S", conc0=1.0, fixed=True, G=0.0)
    net.add_species("P", G=p.G["I0"] + p.dG_rxn)
    net.add_species("D", dead=True, G=-30.0)
    net.add_species("D2", dead=True, metal=2, G=-30.0)
    # cycle, each step in both directions with the same TS (Eyring from the reactant side)
    net.add_reaction("I0 + S -> I1", G_ts=p.TS["TS01"], id="c01")
    net.add_reaction("I1 -> I0 + S", G_ts=p.TS["TS01"], id="c01_rev")
    net.add_reaction("I1 -> I2", G_ts=p.TS["TS12"], id="c12")
    net.add_reaction("I2 -> I1", G_ts=p.TS["TS12"], id="c12_rev")
    net.add_reaction("I2 -> I3", G_ts=p.TS["TS23"], id="c23")
    net.add_reaction("I3 -> I2", G_ts=p.TS["TS23"], id="c23_rev")
    net.add_reaction("I3 -> I0 + P", G_ts=p.TS["TS30"], id="c30")
    net.add_reaction("I0 + P -> I3", G_ts=p.TS["TS30"], id="c30_rev")
    if use_res:
        net.add_reaction(f"{p.att} -> R", G_ts=p.TS["TSR"], id="res")
        net.add_reaction(f"R -> {p.att}", G_ts=p.TS["TSR"], id="res_rev")
    if first_order and p.src1 and p.dG1 is not None:
        net.add_reaction(f"{p.src1} -> D", G_ts=p.TS["TSd1"], id="d1")
    if second_order and p.src2 and p.dGdim is not None:
        net.add_reaction(f"2 {p.src2} -> D2", G_ts=p.TS["TSdim"], id="d2")   # event convention
    pi = None
    if start == "eq":
        pi, _ = stationary(net)
        for n, v in pi.items():
            net.sp(n).conc0 = c0 * v
    elif start in ("I0", "R"):
        net.sp(start).conc0 = c0
    else:
        raise ValueError(start)
    net.meta = dict(profile=p, K_R=p.K_R if use_res else 0.0,
                    kd2_consumption=2.0 * _k(net, "d2") if net.has_rx("d2") else 0.0,
                    kd1=_k(net, "d1") if net.has_rx("d1") else 0.0)
    return net


def _k(net, rid):
    j = [r.id for r in net.reactions].index(rid)
    return net.rate_constants()[j]


def stationary(net: Network):
    """Stationary populations pi (dict over live species) and TOF per catalyst of the death-free,
    product-free linear network: reactions with a dead product, order >= 2 in catalyst, or a non-fixed
    non-catalyst reactant (product re-uptake) are dropped; fixed co-reactants are folded into k."""
    live = [s.name for s in net.species if s.catalyst]
    idx = {n: i for i, n in enumerate(live)}
    fixed = {s.name: s.conc0 for s in net.species if s.fixed}
    dead = {s.name for s in net.species if s.dead}
    ks = net.rate_constants()
    n = len(live)
    Q = np.zeros((n, n))
    prod_rate = np.zeros(n)
    for r, k in zip(net.reactions, ks):
        if any(p in dead for p in r.products):
            continue
        cat_re = {a: v for a, v in r.reactants.items() if a in idx}
        other = {a: v for a, v in r.reactants.items() if a not in idx}
        if sum(cat_re.values()) != 1 or any(a not in fixed for a in other):
            continue
        keff = k
        for a, v in other.items():
            keff *= fixed[a] ** v
        i = idx[next(iter(cat_re))]
        Q[i, i] -= keff
        for a in r.products:
            if a in idx:
                Q[idx[a], i] += keff       # column convention: dpi/dt = Q pi
        prod_rate[i] += keff * r.products.get(net.product, 0)
    # null vector of Q
    A = np.vstack([Q, np.ones(n)])
    b = np.zeros(n + 1); b[-1] = 1.0
    pi = np.linalg.lstsq(A, b, rcond=None)[0]
    return dict(zip(live, pi)), float(prod_rate @ pi)


def equilibrated_start(net: Network, c0: float | None = None) -> Network:
    c0 = net.c0_total() if c0 is None else c0
    pi, _ = stationary(net)
    for n, v in pi.items():
        net.sp(n).conc0 = c0 * v
    return net


def energetic_span_tof(p: Profile) -> float:
    """Kozuch-Shaik (2011) TOF of the cycle I0 -> I1 -> I2 -> I3 -> I0, S at 1 M, no reservoir, no death:
        TOF = (kB T/h) (e^{-dGr/RT} - 1) / sum_ij exp[(T_i - I_j - dG'_ij)/RT],  dG'_ij = dGr if i >= j else 0
    where T_i is the TS leaving intermediate i and I_j the intermediate j (i, j in cycle order, I0 first)."""
    RT = p.RT
    I = [p.G["I0"], p.G["I1"], p.G["I2"], p.G["I3"]]
    Tm = [p.TS["TS01"], p.TS["TS12"], p.TS["TS23"], p.TS["TS30"]]
    d = p.dG_rxn
    den = 0.0
    for i in range(4):
        for j in range(4):
            dGp = d if i >= j else 0.0
            den += np.exp((Tm[i] - I[j] - dGp) / RT)
    return KB_OVER_H * p.T * (np.exp(-d / RT) - 1.0) / den


def ton_control(p: Profile, c0: float = 1e-3, step: float = 0.1, keys=None, T_horizon: float | None = None,
                rtol: float = 1e-6, atol_scaled: float = 1e-11, live_tol: float = 1e-9, **net_kw):
    """Degree of TON control X_i = -RT d ln TON / d G_i (central difference, G_i in kcal/mol) for every
    intermediate G[...], the reservoir G_R and every TS (all other energies fixed). TON is TON_inf unless
    T_horizon is given (then TON(T)). Returns dict key -> X_i."""
    def ton(q):
        net = span_network(q, c0=c0, **net_kw)
        if T_horizon is None:
            return ton_inf(net, half_time=False, rtol=rtol, scale="auto", atol_scaled=atol_scaled,
                           live_tol=live_tol).TON
        return ton_T(net, T_horizon, rtol=rtol, scale="auto", atol_scaled=atol_scaled)
    out = {}
    keys = keys or ([f"G:{n}" for n in p.G] + (["G:R"] if p.G_R is not None else []) + [f"TS:{n}" for n in p.TS if n != "TSR" or p.G_R is not None])
    for key in keys:
        kind, n = key.split(":")
        vals = []
        for sgn in (+1, -1):
            q = p.copy()
            if kind == "G" and n == "R":
                q.G_R = p.G_R + sgn * step
            elif kind == "G":
                q.G[n] = p.G[n] + sgn * step
            else:
                q.TS[n] = p.TS[n] + sgn * step
            vals.append(np.log(ton(q)))
        out[key] = -p.RT * (vals[0] - vals[1]) / (2 * step)
    return out
