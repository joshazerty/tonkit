"""Helpers for the P4 figure scripts: style, colour roles, panel letters, labelled reference
lines, log axes and a save() that writes pdf + 600-dpi png + the numbers json.

Spec: 05_reports/figures/2026-10-13_P4_figure_briefs.md, section 3.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.colors import to_rgb

__all__ = ["style", "C", "ramp", "W1", "W2", "figure", "panel_label", "ref_line", "label_line_end",
           "log_axis", "save", "INK", "MUTED"]

STYLE = Path(__file__).with_name("acs.mplstyle")
W1, W2 = 3.25, 7.0          # ACS single / double column width, inches
INK, MUTED = "#000000", "#666666"

# Colour roles (Okabe-Ito). Never the only carrier of meaning: pair with style, marker or label.
C = {
    "none": "#000000",       # no reservoir / baseline
    "grey": "#666666",
    "light": "#BBBBBB",
    "first": "#0072B2",      # first-order deactivation (n_d = n_p)
    "second": "#D55E00",     # second-order deactivation (n_d > n_p)
    "mixed": "#CC79A7",
    "release": "#009E73",    # slow release
    "feed": "#E69F00",       # slow / constant addition
    "clock": "#56B4E9",      # external clock
}


def style():
    """Apply the house style (call once at the top of a figure script)."""
    plt.style.use(str(STYLE))


def ramp(hex_colour: str, n: int, light: float = 0.45, dark: float = 0.35):
    """n colours of one hue, light -> dark (tint with white, then shade with black)."""
    base = np.array(to_rgb(hex_colour))
    out = []
    for t in np.linspace(0.0, 1.0, n):
        if n == 1:
            t = 1.0
        # t = 0: tint `light` towards white; t = 1: shade `dark` towards black
        mix = -light + t * (light + dark)
        c = base + (1 - base) * (-mix) if mix < 0 else base * (1 - mix)
        out.append(tuple(np.clip(c, 0, 1)))
    return out


def figure(width: float, height: float):
    return plt.figure(figsize=(width, height))


def panel_label(ax, letter: str, dx: float = -0.42, dy: float = 0.06):
    """Bold '(a)' at the top left, outside the axes; offsets in inches."""
    fig = ax.figure
    trans = mpl.transforms.offset_copy(ax.transAxes, fig=fig, x=dx * 72, y=dy * 72, units="points")
    ax.text(0, 1, f"({letter})", transform=trans, fontsize=9.1, fontweight="bold",
            ha="left", va="bottom")


def ref_line(ax, kind: str, value: float, text: str | None = None, where: float = 0.98,
             color: str = "#888888", lw: float = 0.5, ls: str = "-", fontsize: float = 6.5,
             va: str = "bottom", ha: str = "right", zorder: float = 0.5):
    """Thin labelled reference line: kind 'h' (y = value) or 'v' (x = value)."""
    if kind == "h":
        ax.axhline(value, color=color, lw=lw, ls=ls, zorder=zorder)
        if text:
            ax.text(where, value, text, transform=ax.get_yaxis_transform(), color=MUTED,
                    fontsize=fontsize, ha=ha, va=va)
    else:
        ax.axvline(value, color=color, lw=lw, ls=ls, zorder=zorder)
        if text:
            ax.text(value, where, " " + text, transform=ax.get_xaxis_transform(), color=MUTED,
                    fontsize=fontsize, ha="left", va="top", rotation=0)


def label_line_end(ax, x, y, text, color=INK, dx=3, dy=0, fontsize=8, ha="left", va="center"):
    """Direct label just beyond (x, y) in data coordinates; offset in points. Text stays in ink."""
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=fontsize,
                color=INK, ha=ha, va=va)
    # a short colour key next to text keeps identity off the text colour
    return None


def log_axis(ax, which: str = "x", lo=None, hi=None, every: int = 1):
    """Log scale with labels at every `every`-th decade written 10^n."""
    axis = ax.xaxis if which == "x" else ax.yaxis
    (ax.set_xscale if which == "x" else ax.set_yscale)("log")
    if lo is not None:
        (ax.set_xlim if which == "x" else ax.set_ylim)(lo, hi)
    axis.set_major_locator(mticker.LogLocator(base=10, numticks=40))
    axis.set_minor_locator(mticker.LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=40))
    axis.set_minor_formatter(mticker.NullFormatter())

    def fmt(v, _):
        if v <= 0:
            return ""
        e = np.log10(v)
        if abs(e - round(e)) > 1e-6 or (round(e) % every):
            return ""
        e = int(round(e))
        if e == 0:
            return "1"
        if e == 1:
            return "10"
        return rf"$\mathregular{{10^{{{e}}}}}$"
    axis.set_major_formatter(mticker.FuncFormatter(fmt))


def _default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def save(fig, outdir: Path, name: str, numbers: dict | None = None):
    """Write <name>.pdf (vector, TrueType fonts), <name>.png (600 dpi) and <name>_numbers.json."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"{name}.pdf")
    fig.savefig(outdir / f"{name}.png", dpi=600)
    if numbers is not None:
        (outdir / f"{name}_numbers.json").write_text(json.dumps(numbers, indent=1, default=_default))
    plt.close(fig)
