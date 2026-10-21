[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

# tonkit — TON kinetics with off-cycle reservoirs

Small Python package that turns a catalytic network (cycle + reservoirs + dead ends) into the three TON definitions (TON_inf, TON(T), t_half), the exact single-molecule law of the turnover count N, and stochastic (Gillespie) runs. Every closed form from T1–T4 is reproduced through this general engine by the test suite.

## Install and test

```bash
git clone https://github.com/joshazerty/tonkit && cd tonkit
python3 -m pip install -e ".[fast,test,plots]"     # numpy, scipy; numba for the agent SSA; pytest
python3 -m pytest -q                                # about 30 s
```

`tests/deviation_report.csv` is rewritten by every test run: one row per comparison (engine value, closed form or exact value, deviation, tolerance).

## Use

```python
from tonkit import Network, ton_inf, ton_T, markov, gillespie

net = Network("cycle")
net.add_species("A", catalyst=True, conc0=1e-3).add_species("B", catalyst=True).add_species("R", catalyst=True)
net.add_species("S", conc0=1.0, fixed=True).add_species("P")
net.add_species("D2", dead=True, metal=2)
net.add_reaction("A + S -> B", k=100.0)
net.add_reaction("B -> A + P", k=1.0)
net.add_reaction("B <-> R", k=1e4, k_rev=1e3)                         # reservoir, K_R = 10
net.add_reaction("2 B -> D2", k=10.0, convention="consumption")      # d[B]/dt = -k_d2 [B]^2
# declare catalyst/dead species first: species first met in add_reaction are plain (no metal)

r = ton_inf(net)            # catalyst-limited TON_inf, t_half, full trajectory (r.t, r["B"], ...)
ton_T(net, 1e4)             # TON at a horizon
```

or from JSON (`Network.load("examples/....json")`, `python3 -m tonkit net.json --T 1e4 --ssa 10000`).

| Module | Contents |
|---|---|
| `network` | Network spec. JSON is a superset of the run-mkm schema (chem-skills/run-mkm), so a run-mkm network with `G`/`G_ts` loads and gets Eyring constants; extra keys: `catalyst`, `dead`, `metal`, `k`, `k_rev`, `convention`, `orders`. Metal balance is checked per reaction. |
| `ode` | Mass-action ODEs with analytic Jacobian (LSODA, rtol 1e-8). `ton_inf` (stops when live metal < 1e-13 c0), `ton_T`, `P_of_t`, `simulate`; `t_half` = time to half of the final product. Time-dependent rate constants (callables) allowed, e.g. a zero-order feed for slow addition. |
| `closed_forms` | T1–T4 closed forms (ported from `p1lib.py` and the T1–T4 scripts): mixed-order TON_inf with decaying reservoir and cross-deactivation, dimeric reservoir, bimetallic turnover, TON(T), x*, g*, D*, slow-release (Bessel), constant feed, dosing bound, closed-batch P_inf, Ψ(z), poison clock. |
| `markov` | Single-molecule chain of a linear network: exact P(N = n), ⟨N⟩, dead-end split, mean times, lifetime. Raises `NonlinearNetworkError` for steps of order ≥ 2 in catalyst. |
| `gillespie` | `ssa_linear` (independent molecules, vectorised; time-dependent rates by thinning) and `ssa_agents` (interacting molecules, per-molecule turnover counts, numba). |
| `profile` | Energetic-span profiles via run-mkm Eyring route: `Profile`, `span_network`, `stationary`, `energetic_span_tof` (Kozuch-Shaik check), `ton_control` (X_TON). |
| `sweep` | `grid`, `pmap` (fork pool), `write_csv` for parameter sweeps. |
| `networks` | The toy models used in the tests and runs (`cycle`, `pool`, `t1_network`, `dimer_reservoir`, `bimetallic`, `closed_batch`). |

## Conventions

- Rate constants are in the **event** convention by default (`2 X -> D2` at k gives d[X]/dt = −2k[X]²). `convention="consumption"` gives the project convention d[X]/dt = −k_d2[X]²; all toy networks use it for 2X → D2.
- TON is per metal loaded: c0 = Σ metal × conc0 over live and dead species.
- `fixed` species are held constant (catalyst-limited TON_inf). With a finite substrate `ton_inf` returns the conversion-capped TON.
- Agent SSA: system size Ω = n0/c0 molecules per M; propensity of 2X → … is k n(n−1)/Ω (event convention). Live species must carry one metal (a dimeric reservoir needs the ODE engine).

## Solver note

rtol 1e-8 is the default because deep, fast reservoirs (k_on up to 10⁷ s⁻¹ against deactivation at 10⁻⁷ s⁻¹) make LSODA take 10–100× more steps at rtol ≤ 1e-9 without changing TON beyond 10⁻⁷. BDF and Radau were much slower on these problems.

`simulate(..., scale="auto")` integrates scaled variables z = y/scale (atol_scaled) so tiny quasi-steady intermediates stay well conditioned; the profile sweeps use rtol 1e-6, noise floor ~1e-5 relative. New closed forms: `ton_inf_bimetallic_mixed`, `ton_T_bimetallic_mixed`, `loading_slope_mixed`; `bimetallic(kd2=)` adds the 2X -> D2 channel.

## Status

105 tests pass (closed forms T1–T4 reproduced through the general engine; exact single-molecule identities hold to 1e-12).
