"""Reaction-network specification for catalyst-lifetime (TON) kinetics.

The JSON schema is a superset of the one used by the run-mkm skill
(github.com/joshazerty/chem-skills, skills/run-mkm/mkm/network.py), so a network
written for run-mkm loads here unchanged; the extra keys describe catalyst death.

    {
      "name": "cycle_with_reservoir",
      "T": 298.15,                       # K (only used when rates come from free energies)
      "energy_unit": "kcal/mol",         # as run-mkm
      "standard_state": 1.0,             # mol/L, as run-mkm
      "species": [
        {"name": "A", "conc0": 1e-3, "catalyst": true},
        {"name": "B", "catalyst": true},
        {"name": "R", "catalyst": true},
        {"name": "S", "conc0": 1.0, "fixed": true},
        {"name": "P"},
        {"name": "D",  "dead": true},              # dead catalyst, 1 metal
        {"name": "D2", "dead": true, "metal": 2}   # dead dimer, 2 metals
      ],
      "reactions": [
        {"id": "r1", "eq": "A + S -> B", "k": 100.0},
        {"id": "r2", "eq": "B -> A + P", "k": 1.0},
        {"id": "res", "eq": "B <-> R", "k": 1e4, "k_rev": 1e3},
        {"id": "d2",  "eq": "2 B -> D2", "k": 10.0, "convention": "consumption"}
      ],
      "objective": {"type": "species", "name": "P"}
    }

Species keys
  conc0      initial concentration (M); default 0
  fixed      held constant (pseudo-first-order co-reactant; catalyst-limited runs)
  catalyst   live catalyst-bearing species (on-cycle or reservoir)
  dead       dead catalyst (irreversible sink); counts toward metal balance, not live
  metal      metal atoms per molecule (default 1 for catalyst/dead species, 0 otherwise)
  G          free energy (energy_unit), only if rates are built by Eyring (run-mkm style)

Reaction keys
  eq         "2 A + S -> B" or "A <-> R" (alternative: run-mkm style "reactants"/"products" dicts)
  k          forward rate constant (mass action, M^(1-m) s^-1); may be a callable k(t) in the Python API
  k_rev      reverse rate constant for "<->" (or "reversible": true)
  convention "event" (default): event rate r = k prod c^nu  (so 2X -> D gives d[X]/dt = -2 k X^2)
             "consumption": k is defined on consumption of the first reactant with stoichiometry nu,
             r = (k/nu) prod c^nu  (so 2X -> D gives d[X]/dt = -k X^2, the project convention, notation.md §2)
  G_ts       TS free energy (run-mkm style); used only when k is absent (Eyring, thermodynamically
             consistent reverse step from the species G)
  orders     rate-law exponents overriding the stoichiometry ({"Xsrc": 0} makes a zero-order feed);
             ODE engine only

Objective: the species whose formation counts turnovers (default "P").
"""
from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, field
from typing import Callable, Optional, Union

import numpy as np

# --- constants for the Eyring option (CODATA 2018, as run-mkm) ---
KB_OVER_H = 1.380649e-23 / 6.62607015e-34
R_J = 8.314462618
ENERGY_TO_J = {"j/mol": 1.0, "kj/mol": 1e3, "kcal/mol": 4184.0, "ev": 96485.33212, "hartree": 2625499.6}

Rate = Union[float, Callable[[float], float]]


@dataclass
class Species:
    name: str
    conc0: float = 0.0
    fixed: bool = False
    catalyst: bool = False
    dead: bool = False
    metal: Optional[int] = None
    G: Optional[float] = None
    role: str = ""

    def __post_init__(self):
        if self.metal is None:
            self.metal = 1 if (self.catalyst or self.dead) else 0
        if self.catalyst and self.dead:
            raise ValueError(f"species {self.name!r} cannot be both catalyst (live) and dead")


@dataclass
class Reaction:
    id: str
    reactants: dict
    products: dict
    k: Optional[Rate] = None
    convention: str = "event"
    G_ts: Optional[float] = None
    kappa: float = 1.0
    nu_imag: Optional[float] = None
    orders: Optional[dict] = None   # rate-law exponents overriding the stoichiometry, e.g. {"Xsrc": 0}
    note: str = ""

    def event_k(self):
        """Rate constant in the event convention (r = k_event prod c^nu)."""
        if self.k is None:
            raise ValueError(f"reaction {self.id!r} has no rate constant")
        if self.convention == "event":
            return self.k
        if self.convention == "consumption":
            first = next(iter(self.reactants))
            nu = self.reactants[first]
            if callable(self.k):
                kk = self.k
                return lambda t: kk(t) / nu
            return self.k / nu
        raise ValueError(f"unknown rate convention {self.convention!r} in reaction {self.id!r}")


_TERM = re.compile(r"^\s*(\d+(?:\.\d+)?)?\s*\*?\s*([A-Za-z_][\w\[\]\.\'\-]*)\s*$")


def parse_side(side: str) -> dict:
    side = side.strip()
    if side in ("", "0", "∅", "nothing"):
        return {}
    out = {}
    for term in side.split(" + "):
        m = _TERM.match(term)
        if not m:
            raise ValueError(f"cannot parse term {term!r} (separate species with ' + ')")
        nu = float(m.group(1)) if m.group(1) else 1.0
        nu = int(nu) if nu.is_integer() else nu
        out[m.group(2)] = out.get(m.group(2), 0) + nu
    return out


def parse_equation(eq: str):
    """'2 A + S -> B' -> ({'A':2,'S':1}, {'B':1}, reversible=False). '<->' or '<=>' marks reversible."""
    for arrow, rev in (("<->", True), ("<=>", True), ("->", False), ("=>", False)):
        if arrow in eq:
            lhs, rhs = eq.split(arrow, 1)
            return parse_side(lhs), parse_side(rhs), rev
    raise ValueError(f"no arrow in equation {eq!r}")


@dataclass
class Network:
    name: str = "network"
    species: list = field(default_factory=list)
    reactions: list = field(default_factory=list)
    objective: dict = field(default_factory=lambda: {"type": "species", "name": "P"})
    T: float = 298.15
    energy_unit: str = "kcal/mol"
    standard_state: float = 1.0

    # ------------------------------------------------------------------ building
    def add_species(self, name, **kw):
        if any(s.name == name for s in self.species):
            raise ValueError(f"duplicate species {name!r}")
        self.species.append(Species(name, **kw))
        return self

    def add_reaction(self, eq, k=None, k_rev=None, id=None, convention="event", **kw):
        """Add 'A + S -> B' (k) or 'A <-> R' (k, k_rev). A reversible step becomes two reactions,
        ids '<id>' and '<id>_rev'. Unknown species are created as plain (non-catalyst) species."""
        lhs, rhs, rev = parse_equation(eq)
        rid = id or f"r{len(self.reactions) + 1}"
        for name in list(lhs) + list(rhs):
            if not any(s.name == name for s in self.species):
                self.species.append(Species(name))
        self.reactions.append(Reaction(rid, lhs, rhs, k=k, convention=convention, **kw))
        if rev:
            if k_rev is None:
                raise ValueError(f"reversible step {eq!r} needs k_rev")
            self.reactions.append(Reaction(rid + "_rev", dict(rhs), dict(lhs), k=k_rev))
        elif k_rev is not None:
            raise ValueError(f"k_rev given for irreversible step {eq!r}; use '<->'")
        return self

    def set(self, **conc0):
        """Set initial concentrations: net.set(A=1e-3, S=1.0)."""
        for name, c in conc0.items():
            self.sp(name).conc0 = float(c)
        return self

    def set_k(self, rid, k):
        self.rx(rid).k = k
        return self

    def copy(self):
        return copy.deepcopy(self)

    # ------------------------------------------------------------------ lookup
    def sp(self, name) -> Species:
        for s in self.species:
            if s.name == name:
                return s
        raise KeyError(f"no species {name!r}")

    def has_rx(self, rid) -> bool:
        return any(r.id == rid for r in self.reactions)

    def rx(self, rid) -> Reaction:
        for r in self.reactions:
            if r.id == rid:
                return r
        raise KeyError(f"no reaction {rid!r}")

    @property
    def names(self):
        return [s.name for s in self.species]

    def index(self, name):
        return self.names.index(name)

    @property
    def product(self):
        return self.objective.get("name", "P")

    def catalyst_idx(self):
        return np.array([i for i, s in enumerate(self.species) if s.catalyst], dtype=int)

    def dead_idx(self):
        return np.array([i for i, s in enumerate(self.species) if s.dead], dtype=int)

    def metal_vector(self):
        return np.array([s.metal for s in self.species], dtype=float)

    def live_metal_vector(self):
        return np.array([s.metal if s.catalyst else 0 for s in self.species], dtype=float)

    def c0_total(self):
        """Total metal loaded (live + dead at t = 0), M."""
        return float(np.dot(self.metal_vector(), [s.conc0 for s in self.species]))

    def y0(self):
        return np.array([s.conc0 for s in self.species], dtype=float)

    # ------------------------------------------------------------------ matrices
    def stoich(self):
        """(Rmat, Pmat): reactant and product multiplicities, ns x nr."""
        ns, nr = len(self.species), len(self.reactions)
        Rm = np.zeros((ns, nr)); Pm = np.zeros((ns, nr))
        idx = {n: i for i, n in enumerate(self.names)}
        for j, r in enumerate(self.reactions):
            for n, v in r.reactants.items():
                Rm[idx[n], j] += v
            for n, v in r.products.items():
                Pm[idx[n], j] += v
        return Rm, Pm

    def orders_matrix(self):
        """Rate-law exponents (ns x nr): the reactant multiplicities unless a reaction overrides them."""
        Rm, _ = self.stoich()
        idx = {n: i for i, n in enumerate(self.names)}
        for j, r in enumerate(self.reactions):
            if r.orders:
                for n, o in r.orders.items():
                    Rm[idx[n], j] = o
        return Rm

    def rate_constants(self):
        """Event-convention rate constants (floats or callables). Fills Eyring constants
        (run-mkm style) for reactions given by G_ts and species G instead of k."""
        out = []
        for r in self.reactions:
            if r.k is None and r.G_ts is not None:
                out.append(self._eyring(r))
            else:
                out.append(r.event_k())
        return out

    def _eyring(self, r):
        u = ENERGY_TO_J[self.energy_unit.lower()]
        G = {s.name: s.G for s in self.species}
        Greact = sum(v * G[n] for n, v in r.reactants.items())
        dG = max(0.0, (r.G_ts - Greact) * u)
        m = sum(r.reactants.values())
        return r.kappa * KB_OVER_H * self.T * np.exp(-dG / (R_J * self.T)) / self.standard_state ** (m - 1)

    # ------------------------------------------------------------------ checks
    def validate(self, strict=True):
        """Return list of (level, message); raise on FAIL if strict."""
        msgs = []
        names = set(self.names)
        used = set()
        for r in self.reactions:
            used |= set(r.reactants) | set(r.products)
            for n in set(r.reactants) | set(r.products):
                if n not in names:
                    msgs.append(("FAIL", f"{r.id}: unknown species {n!r}"))
            if r.k is None and r.G_ts is None:
                msgs.append(("FAIL", f"{r.id}: no rate constant"))
            # metal balance per reaction
            mv = {s.name: s.metal for s in self.species}
            if all(n in mv for n in list(r.reactants) + list(r.products)):
                m_in = sum(v * mv[n] for n, v in r.reactants.items())
                m_out = sum(v * mv[n] for n, v in r.products.items())
                if abs(m_in - m_out) > 1e-12:
                    msgs.append(("FAIL", f"{r.id}: metal not conserved ({m_in} -> {m_out}); "
                                         f"set 'metal' on species or include the catalyst explicitly"))
            for n in r.reactants:
                if n in names and self.sp(n).dead:
                    msgs.append(("WARN", f"{r.id}: dead species {n!r} is a reactant (reactivation?)"))
        for n in names - used:
            msgs.append(("WARN", f"species {n!r} appears in no reaction"))
        if self.product not in names:
            msgs.append(("FAIL", f"objective species {self.product!r} not found"))
        if len(self.catalyst_idx()) == 0:
            msgs.append(("FAIL", "no catalyst species"))
        if self.c0_total() <= 0:
            msgs.append(("WARN", "no catalyst loaded (conc0 = 0 for every metal-bearing species)"))
        if len(self.dead_idx()) == 0:
            msgs.append(("WARN", "no dead species: TON_inf is infinite unless a step leaves the catalyst pool"))
        if strict and any(l == "FAIL" for l, _ in msgs):
            raise ValueError("invalid network:\n" + "\n".join(f"  {l}: {m}" for l, m in msgs))
        return msgs

    # ------------------------------------------------------------------ IO
    def to_dict(self):
        sp = []
        for s in self.species:
            d = {"name": s.name}
            if s.conc0: d["conc0"] = s.conc0
            for key in ("fixed", "catalyst", "dead"):
                if getattr(s, key): d[key] = True
            if s.metal not in (None, 1 if (s.catalyst or s.dead) else 0): d["metal"] = s.metal
            if s.G is not None: d["G"] = s.G
            sp.append(d)
        rx = []
        for r in self.reactions:
            if callable(r.k):
                raise ValueError(f"reaction {r.id!r} has a time-dependent k; not serialisable")
            d = {"id": r.id, "reactants": r.reactants, "products": r.products}
            if r.k is not None: d["k"] = r.k
            if r.convention != "event": d["convention"] = r.convention
            if r.orders: d["orders"] = r.orders
            if r.G_ts is not None: d["G_ts"] = r.G_ts
            rx.append(d)
        return {"name": self.name, "T": self.T, "energy_unit": self.energy_unit,
                "standard_state": self.standard_state, "species": sp, "reactions": rx,
                "objective": self.objective}

    def save(self, path):
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=1)

    @classmethod
    def from_dict(cls, d):
        strip = lambda x: {k: v for k, v in x.items() if not k.startswith("_")}
        net = cls(name=d.get("name", "network"), objective=d.get("objective", {"type": "species", "name": "P"}),
                  T=float(d.get("T", 298.15)), energy_unit=d.get("energy_unit", "kcal/mol"),
                  standard_state=float(d.get("standard_state", 1.0)))
        for s in d["species"]:
            s = strip(s)
            s.pop("role", None)
            net.species.append(Species(**s))
        for r in d["reactions"]:
            r = strip(r)
            if "eq" in r:
                lhs, rhs, rev = parse_equation(r.pop("eq"))
            else:
                lhs, rhs, rev = r.pop("reactants"), r.pop("products"), bool(r.pop("reversible", False))
            k_rev = r.pop("k_rev", None)
            rid = r.pop("id", f"r{len(net.reactions) + 1}")
            net.reactions.append(Reaction(rid, lhs, rhs, **r))
            if rev or k_rev is not None:
                if k_rev is None:
                    raise ValueError(f"reversible reaction {rid!r} needs k_rev")
                net.reactions.append(Reaction(rid + "_rev", dict(rhs), dict(lhs), k=k_rev))
        return net

    @classmethod
    def load(cls, path):
        with open(path) as fh:
            return cls.from_dict(json.load(fh))

    def __str__(self):
        lines = [f"Network {self.name!r}: {len(self.species)} species, {len(self.reactions)} reactions"]
        for r in self.reactions:
            fmt = lambda d: " + ".join((f"{v:g} " if v != 1 else "") + n for n, v in d.items()) or "0"
            k = "k(t)" if callable(r.k) else f"{r.k:.4g}" if r.k is not None else f"G_ts={r.G_ts}"
            lines.append(f"  {r.id:>10s}: {fmt(r.reactants)} -> {fmt(r.products)}   k = {k} [{r.convention}]")
        return "\n".join(lines)
