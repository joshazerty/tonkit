"""Engine sanity: parsing, rate conventions, metal balance, Jacobian, IO, analytic toy solutions."""
import json
import os

import numpy as np
import pytest

from tonkit import Network, parse_equation, simulate, ton_inf, ton_T
from tonkit.ode import Model
from tonkit import networks as nw

HERE = os.path.dirname(__file__)


def test_parse_equation():
    assert parse_equation("2 A + S -> B") == ({"A": 2, "S": 1}, {"B": 1}, False)
    assert parse_equation("X <-> R") == ({"X": 1}, {"R": 1}, True)
    assert parse_equation("Xsrc -> X")[2] is False
    assert parse_equation("A + R -> 2 D") == ({"A": 1, "R": 1}, {"D": 2}, False)


def test_metal_balance_is_enforced():
    net = Network("bad")
    net.add_species("X", catalyst=True, conc0=1e-3).add_species("P").add_species("D2", dead=True)  # metal 1
    net.add_reaction("X -> X + P", k=1.0)
    net.add_reaction("2 X -> D2", k=1.0)        # 2 metals -> 1 metal: must fail
    with pytest.raises(ValueError, match="metal not conserved"):
        net.validate()


def test_rate_conventions(check):
    """2X -> D2: event convention d[X]/dt = -2 k X^2; consumption convention d[X]/dt = -k X^2."""
    c0, k, t = 1e-3, 50.0, 300.0
    for conv, fac in (("event", 2.0), ("consumption", 1.0)):
        net = Network(conv)
        net.add_species("X", catalyst=True, conc0=c0).add_species("P").add_species("D2", dead=True, metal=2)
        net.add_reaction("X -> X + P", k=1.0)
        net.add_reaction("2 X -> D2", k=k, convention=conv)
        res, m = simulate(net, t)
        X = res["X"][-1]
        check(f"X(t), {conv} convention", X, c0 / (1 + fac * k * c0 * t))
        # TON(T) = (1/(fac k c0)) ln(1 + fac k c0 T)
        check(f"TON(T), {conv} convention", res.TON, np.log1p(fac * k * c0 * t) / (fac * k * c0))


def test_jacobian_matches_finite_differences():
    net = nw.cycle(K_R=7.0, kd1=1e-3, kd2=30.0, kdR=1e-4, kx=5.0, src="A", att="B")
    m = Model(net)
    rng = np.random.default_rng(0)
    y = np.abs(rng.normal(size=m.ns)) * 1e-3 + 1e-5
    J = m.jac(0.0, y)
    Jfd = np.zeros_like(J)
    for i in range(m.ns):
        h = 1e-7 * max(abs(y[i]), 1e-8)
        yp, ym = y.copy(), y.copy()
        yp[i] += h; ym[i] -= h
        Jfd[:, i] = (m.rhs(0.0, yp) - m.rhs(0.0, ym)) / (2 * h)
    assert np.allclose(J, Jfd, rtol=1e-6, atol=1e-9 * np.abs(J).max())


def test_metal_is_conserved_along_trajectory():
    net = nw.cycle(K_R=5.0, kd1=1e-3, kd2=30.0, kdR=1e-4, kx=5.0)
    res, m = simulate(net, 1e4, t_eval=np.geomspace(1e-3, 1e4, 50))
    tot = net.metal_vector() @ res.y
    assert np.allclose(tot, net.c0_total(), rtol=1e-8)


def test_json_round_trip_and_run_mkm_schema(tmp_path):
    net = nw.cycle(K_R=3.0, kd1=1e-3, kd2=10.0)
    p = tmp_path / "net.json"
    net.save(p)
    net2 = Network.load(p)
    assert abs(ton_inf(net).TON / ton_inf(net2).TON - 1) < 1e-10
    # a run-mkm style network (reactants/products dicts, G and G_ts, no k) loads and gets Eyring constants
    d = {"name": "mkm_style", "T": 298.15, "energy_unit": "kcal/mol",
         "species": [{"name": "cat", "G": 0.0, "conc0": 1e-3, "catalyst": True},
                     {"name": "A", "G": 0.0, "conc0": 1.0, "fixed": True},
                     {"name": "cat_A", "G": -2.0, "catalyst": True},
                     {"name": "P", "G": -20.0},
                     {"name": "D", "G": -30.0, "dead": True}],
         "reactions": [{"id": "r1", "reactants": {"cat": 1, "A": 1}, "products": {"cat_A": 1}, "G_ts": 10.0},
                       {"id": "r2", "reactants": {"cat_A": 1}, "products": {"cat": 1, "P": 1}, "G_ts": 15.0},
                       {"id": "d", "reactants": {"cat_A": 1}, "products": {"D": 1}, "G_ts": 19.0}],
         "objective": {"type": "species", "name": "P"}}
    net3 = Network.from_dict(d)
    ks = net3.rate_constants()
    kT_h = 1.380649e-23 * 298.15 / 6.62607015e-34
    assert abs(ks[1] / (kT_h * np.exp(-17.0 * 4184 / (8.314462618 * 298.15))) - 1) < 1e-12
    # TON = k2/kd exactly (death competes with turnover from the same state)
    assert abs(ton_inf(net3).TON / (ks[1] / ks[2]) - 1) < 1e-6


def test_example_json_files_load():
    ex = os.path.join(HERE, "..", "examples")
    for f in sorted(os.listdir(ex)):
        if f.endswith(".json"):
            net = Network.load(os.path.join(ex, f))
            r = ton_inf(net)
            assert r.status == "dead" and r.TON > 0


def test_t_half_first_order_pool(check):
    """X -> X + P (kP), X -> D (kd): P(t) = (kP c0/kd)(1 - e^{-kd t}); t_half = ln 2 / kd."""
    net = nw.pool(kP=1.0, kd1=2e-3, c0=1e-3)
    r = ton_inf(net)
    check("TON_inf = kP/kd", r.TON, 500.0)
    check("t_half = ln2/kd", r.t_half, np.log(2) / 2e-3)
    check("TON(T) at T = 300 s", ton_T(net, 300.0), 500.0 * (1 - np.exp(-0.6)))
