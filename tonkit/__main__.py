"""python -m tonkit network.json [--T horizon] [--ssa n0]  -> TON_inf, t_half, TON(T), optional SSA."""
import argparse

from . import Network, ton_inf, ton_T, markov, gillespie


def main():
    ap = argparse.ArgumentParser(prog="tonkit", description=__doc__)
    ap.add_argument("network")
    ap.add_argument("--T", type=float, default=None, help="horizon for TON(T), s")
    ap.add_argument("--ssa", type=int, default=0, help="number of molecules for a Gillespie run")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    net = Network.load(a.network)
    for lvl, msg in net.validate(strict=False):
        if lvl != "INFO":
            print(f"{lvl}: {msg}")
    print(net)
    r = ton_inf(net)
    print(f"TON_inf = {r.TON:.6g}   t_half = {r.t_half:.4g} s   ({r.status})")
    if a.T:
        print(f"TON(T = {a.T:g} s) = {ton_T(net, a.T):.6g}")
    try:
        ch = markov.linear_chain(net)
        print(f"linear network: exact <N> = {markov.mean_N(ch):.6g}, mean lifetime = {markov.mean_lifetime(ch):.4g} s")
        if a.ssa:
            o = gillespie.ssa_linear(ch, a.ssa, seed=a.seed)
            print(f"SSA ({a.ssa} independent molecules): <N> = {o['N'].mean():.6g} +/- {o['N'].std()/a.ssa**0.5:.3g}")
    except markov.NonlinearNetworkError as e:
        print(f"nonlinear network ({e})")
        if a.ssa:
            o = gillespie.ssa_agents(net, a.ssa, seed=a.seed)
            print(f"agent SSA ({a.ssa} molecules): TON = {o['N'].mean():.6g}")


if __name__ == "__main__":
    main()
