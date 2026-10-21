"""Toy networks used in the tests and the P2 runs (all built through the general Network spec)."""
from __future__ import annotations

import numpy as np

from .network import Network


def cycle(k1S=100.0, k2=1.0, c0=1e-3, src="B", att="B", K_R=0.0, koff=1e3, kd1=0.0, kd2=0.0,
          kdR=0.0, kx=0.0, start="eq", name=None):
    """Explicit two-state cycle with a pendant reservoir and deactivation (T3/T4 test model).

        A + S -> B (k1, S fixed at 1 M so k1 S = k1S),  B -> A + P (k2)
        att <-> R (kon = K_R koff, koff)               att in {A, B}
        src -> D (kd1);  2 src -> D2 (kd2, consumption convention: d[src]/dt = -kd2 src^2)
        R -> D (kdR);    src + R -> 2 D (event constant kx)

    start='eq': free pool at its quasi-steady state, reservoir at equilibrium (depth
    D = 1 + K_R f_att); start='A': all catalyst in A; start='R': all catalyst in R."""
    net = Network(name or f"cycle_src{src}_att{att}")
    net.add_species("A", catalyst=True).add_species("B", catalyst=True)
    net.add_species("S", conc0=1.0, fixed=True).add_species("P")
    net.add_species("D", dead=True)
    net.add_reaction("A + S -> B", k=k1S, id="r1")
    net.add_reaction("B -> A + P", k=k2, id="r2")
    if K_R > 0 or start == "R" or kdR > 0 or kx > 0:
        net.add_species("R", catalyst=True)
    if K_R > 0:
        net.add_reaction(f"{att} <-> R", k=K_R * koff, k_rev=koff, id="res")
    if kd1 > 0:
        net.add_reaction(f"{src} -> D", k=kd1, id="d1")
    if kd2 > 0:
        net.add_species("D2", dead=True, metal=2)
        net.add_reaction(f"2 {src} -> D2", k=kd2, convention="consumption", id="d2")
    if kdR > 0:
        net.add_reaction("R -> D", k=kdR, id="dR")
    if kx > 0:
        net.add_reaction(f"{src} + R -> 2 D", k=kx, id="dx")
    fA, fB = k2 / (k1S + k2), k1S / (k1S + k2)
    fatt = fA if att == "A" else fB
    D = 1 + K_R * fatt
    if start == "eq":
        F = c0 / D
        net.set(A=F * fA, B=F * fB)
        if K_R > 0:
            net.set(R=F * K_R * fatt)
    elif start == "A":
        net.set(A=c0)
    elif start == "R":
        net.set(R=c0)
    net.meta = dict(fA=fA, fB=fB, D=D, kP=k2 * fB, fX=fA if src == "A" else fB)
    return net


def pool(kP=1.0, c0=1e-3, K_R=0.0, koff=1e6, kd1=0.0, kd2=0.0, start="eq", feed=None, name="pool"):
    """Reduced model of T4: one free pool X (turnover X -> X + P at kP), reservoir X <-> R,
    deactivation kd1 X + kd2 X^2 (consumption convention). feed: callable f(t) (M/s) for slow addition
    (then nothing is loaded at t = 0)."""
    net = Network(name)
    net.add_species("X", catalyst=True).add_species("P").add_species("D", dead=True)
    net.add_reaction("X -> X + P", k=kP, id="turn")
    if K_R > 0 or start == "R":
        net.add_species("R", catalyst=True)
    if K_R > 0:
        net.add_reaction("X <-> R", k=K_R * koff, k_rev=koff, id="res")
    if kd1 > 0:
        net.add_reaction("X -> D", k=kd1, id="d1")
    if kd2 > 0:
        net.add_species("D2", dead=True, metal=2)
        net.add_reaction("2 X -> D2", k=kd2, convention="consumption", id="d2")
    if feed is not None:
        # bookkeeping species: metal still to be added (flagged dead so it counts toward c0 but not as
        # live catalyst); zero-order in Xsrc, so the feed rate is exactly feed(t)
        net.add_species("Xsrc", dead=True)
        net.add_reaction("Xsrc -> X", k=feed, id="feed", orders={"Xsrc": 0})
        net.set(Xsrc=c0)
        net.meta = dict(feed=True)
        return net
    if start == "eq":
        net.set(X=c0 / (1 + K_R))
        if K_R > 0:
            net.set(R=c0 * K_R / (1 + K_R))
    elif start == "X":
        net.set(X=c0)
    elif start == "R":
        net.set(R=c0)
    return net


def t1_network(with_res=True, kon1=None):
    """T1 network of the P1 check (one molecule, all steps first order in catalyst):
        A <-> B (k1 S = 5, km1 = 1), B -> C (4), C -> A + P (1); dead ends A -> D1 (0.1), C -> D2 (0.02)
        pendant reservoirs: R1 <-> C (3, 0.5; K1 = 6); B <-> R2 (2, 1) <-> R3 (0.5, 0.25)."""
    net = Network("t1_network" + ("_res" if with_res else ""))
    for n in ("A", "B", "C"):
        net.add_species(n, catalyst=True)
    net.add_species("S", conc0=1.0, fixed=True).add_species("P")
    net.add_species("D1", dead=True).add_species("D2", dead=True)
    net.add_reaction("A + S <-> B", k=5.0, k_rev=1.0, id="r1")
    net.add_reaction("B -> C", k=4.0, id="r2")
    net.add_reaction("C -> A + P", k=1.0, id="r3")
    net.add_reaction("A -> D1", k=0.10, id="dA")
    net.add_reaction("C -> D2", k=0.02, id="dC")
    if with_res:
        for n in ("R1", "R2", "R3"):
            net.add_species(n, catalyst=True)
        net.add_reaction("C <-> R1", k=3.0 if kon1 is None else kon1, k_rev=0.5, id="res1")
        net.add_reaction("B <-> R2", k=2.0, k_rev=1.0, id="res2")
        net.add_reaction("R2 <-> R3", k=0.5, k_rev=0.25, id="res3")
    net.set(A=1.0)
    return net


def dimer_reservoir(kP=1.0, kd1=1e-5, kd2=10.0, Kdim=1e5, kr=1e3, c0=1e-3):
    """Dimeric reservoir 2X <-> X2 (fast, K_dim = kf/kr in M^-1), turnover kP X, deactivation
    kd1 X + kd2 X^2 (consumption convention). Starts at the monomer-dimer equilibrium."""
    net = Network("dimer_reservoir")
    net.add_species("X", catalyst=True).add_species("X2", catalyst=True, metal=2)
    net.add_species("P").add_species("D", dead=True)
    net.add_reaction("X -> X + P", k=kP, id="turn")
    net.add_reaction("2 X <-> X2", k=Kdim * kr, k_rev=kr, id="dim")
    if kd1 > 0:
        net.add_reaction("X -> D", k=kd1, id="d1")
    if kd2 > 0:
        net.add_species("D2", dead=True, metal=2)
        net.add_reaction("2 X -> D2", k=kd2, convention="consumption", id="d2")
    X0 = (-1 + np.sqrt(1 + 8 * Kdim * c0)) / (4 * Kdim) if Kdim > 0 else c0
    net.set(X=X0, X2=Kdim * X0**2)
    return net


def bimetallic(kP2=1e3, kd1=1e-5, K_R=0.0, koff=1e3, c0=1e-3, kd2=0.0):
    """n_p = 2: turnover 2X -> 2X + P (event rate kP2 X^2), death X -> D (n_d = 1) and, if kd2 > 0,
    2X -> D2 (n_d = 2, consumption convention d[X]/dt = -kd2 X^2), reservoir X <-> R."""
    net = Network("bimetallic")
    net.add_species("X", catalyst=True).add_species("P").add_species("D", dead=True)
    net.add_reaction("2 X -> 2 X + P", k=kP2, id="turn")
    if kd1 > 0:
        net.add_reaction("X -> D", k=kd1, id="d1")
    if kd2 > 0:
        net.add_species("D2", dead=True, metal=2)
        net.add_reaction("2 X -> D2", k=kd2, convention="consumption", id="d2")
    if K_R > 0:
        net.add_species("R", catalyst=True)
        net.add_reaction("X <-> R", k=K_R * koff, k_rev=koff, id="res")
    D = 1 + K_R
    net.set(X=c0 / D)
    if K_R > 0:
        net.set(R=c0 * K_R / D)
    return net


def closed_batch(k1=1e3, k2=1.0, kdA=0.1, kdB=2e-4, S0=0.1, c0=7e-5, att="B", K_R=0.0, kex=1e3,
                 full=False, KP=0.0, kpoff=1.0, ks=0.0):
    """T2 model: finite substrate (S not fixed), A + S -> B, B -> A + P, A -> D, B -> D,
    pendant reservoir att <-> R, product-inhibition adduct B + P <-> BP, uncatalysed S loss (ks).
    full=True starts with all catalyst in R. TON counts free P only; product held in the adduct BP
    must be added for the closed-batch total P + BP (see tests/test_t2_branching.py)."""
    net = Network("closed_batch")
    net.add_species("A", catalyst=True).add_species("B", catalyst=True)
    net.add_species("S", conc0=S0).add_species("P").add_species("D", dead=True)
    net.add_reaction("A + S -> B", k=k1, id="r1")
    net.add_reaction("B -> A + P", k=k2, id="r2")
    net.add_reaction("A -> D", k=kdA, id="dA")
    net.add_reaction("B -> D", k=kdB, id="dB")
    if K_R > 0:
        net.add_species("R", catalyst=True)
        net.add_reaction(f"{att} <-> R", k=K_R * kex, k_rev=kex, id="res")
    if KP > 0:
        net.add_species("BP", catalyst=True)
        net.add_reaction("B + P <-> BP", k=KP * kpoff, k_rev=kpoff, id="inh")
    if ks > 0:
        net.add_species("W")
        net.add_reaction("S -> W", k=ks, id="ks")
    if full and K_R > 0:
        net.set(R=c0)
    else:
        net.set(A=c0)
    return net
