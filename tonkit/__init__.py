"""tonkit: catalyst-lifetime (TON) kinetics for networks with off-cycle reservoirs (project 22, P2).

    from tonkit import Network, ton_inf, ton_T
    net = Network.load("cycle.json")        # run-mkm-compatible JSON + catalyst/dead flags
    res = ton_inf(net)                       # catalyst-limited TON, t_half
    res.TON, res.t_half

Modules: network (spec), ode (stiff ODEs, TON_inf / TON(T) / t_half), closed_forms (T1-T4),
markov (exact single-molecule law of N, linear networks), gillespie (stochastic runs),
networks (toy models used in the tests and runs), profile (energetic-span networks from G/G_ts),
sweep (grid / parallel map / CSV helpers).
"""
from .network import Network, Species, Reaction, parse_equation
from .ode import simulate, ton_inf, ton_T, P_of_t
from . import closed_forms, markov, gillespie, networks, profile, sweep

__version__ = "0.1.0"
__all__ = ["Network", "Species", "Reaction", "parse_equation", "simulate", "ton_inf", "ton_T",
           "P_of_t", "closed_forms", "markov", "gillespie", "networks", "profile", "sweep"]
