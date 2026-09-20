"""Shared loading/plotting helpers for the BraggGlass-1plus1 diagnostics.

Not meant to be run directly -- see plot_spectrum.py, plot_structure_factor.py,
plot_correlation.py, and plot_all.py in this directory.
"""
from __future__ import annotations

import glob
import os
import re
import warnings
from collections import defaultdict

import numpy as np

# ── Palette (dataviz skill, references/palette.md) ─────────────────────────
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

CAT_BLUE = "#2a78d6"      # categorical slot 1 -- a single simulation curve
CAT_ORANGE = "#eb6834"    # categorical slot 2 -- "Thermal" reference curve
CAT_AQUA = "#1baf7a"      # categorical slot 3 -- "Larkin" reference curve
CAT_VIOLET = "#4a3aa7"    # categorical slot 7 -- "Debye-Waller estimate"

# Sequential blue ramp, light -> dark (ordinal use: start no lighter than
# step 250 on the light surface -- see palette.md).
SEQ_BLUE = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
            "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281"]


def sequential_colors(n: int) -> list[str]:
    """n evenly spaced steps from the ordinal-safe part of the blue ramp,
    used to color an ordered set of temperatures (light = low T)."""
    if n <= 1:
        return [CAT_BLUE]
    idx = np.linspace(0, len(SEQ_BLUE) - 1, n).round().astype(int)
    return [SEQ_BLUE[i] for i in idx]


def style_axes(ax, log_x: bool = True, log_y: bool = True) -> None:
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    ax.grid(True, which="both", color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(BASELINE)
    ax.tick_params(colors=INK_MUTED, labelcolor=INK_SECONDARY)
    ax.xaxis.label.set_color(INK_PRIMARY)
    ax.yaxis.label.set_color(INK_PRIMARY)
    ax.title.set_color(INK_PRIMARY)


def find_replica_files(root: str, stem: str) -> dict[float, list[str]]:
    """Group every *<stem>_replica_<T>.dat under root by temperature T.

    Matches a flat seed scan (seed_<N>_<stem>_replica_<T>.dat, from run.sh),
    a single run's output (<stem>_replica_<T>.dat), and the per-seed
    subdirectories written by slurm/run_array.slurm (results/seed_<N>/...).
    """
    pattern = os.path.join(root, "**", f"*{stem}_replica_*.dat")
    rx = re.compile(rf"{re.escape(stem)}_replica_([0-9.eE+-]+)\.dat$")
    groups: dict[float, list[str]] = defaultdict(list)
    for path in glob.glob(pattern, recursive=True):
        m = rx.search(os.path.basename(path))
        if m:
            groups[float(m.group(1))].append(path)
    return dict(sorted(groups.items()))


def load_averaged(paths: list[str], ncols: int) -> np.ndarray:
    """Load matching-shape .dat files and average column-wise across seeds
    (nan-aware, since B_x/B_y pad with 'nan' past their own axis' range --
    an all-nan row there is expected, not a warning-worthy condition)."""
    arrays = [np.loadtxt(p, comments="#", usecols=range(ncols)) for p in paths]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", r"Mean of empty slice", RuntimeWarning)
        return np.nanmean(np.stack(arrays, axis=0), axis=0)


def plot_spectrum(ax, directory: str, ny: int) -> bool:
    """S_u(qy): q^2<|u(q)|^2>*Ny vs q, against thermal/Larkin scaling."""
    groups = find_replica_files(directory, "displacement_spectra")
    if not groups:
        return False

    qy_min, qy_max = np.inf, -np.inf
    for color, (T, paths) in zip(sequential_colors(len(groups)), groups.items()):
        data = load_averaged(paths, ncols=3)
        qy, su = data[:, 1], data[:, 2]
        qy_min, qy_max = min(qy_min, qy.min()), max(qy_max, qy.max())
        ax.plot(qy, qy**2 * su * ny, "o-", color=color, linewidth=2,
                 markersize=5, label=f"sim, T={T:g} ({len(paths)} seeds)")

    # Range for the reference curves is taken from the data, not ax.get_xlim():
    # the axis is still linear-scaled here (style_axes runs after this), and a
    # linear autoscale can pad down to 0/negative, which breaks geomspace.
    xs = np.geomspace(qy_min, qy_max, 100)
    ax.plot(xs, 0.25 * xs, "--", color=CAT_ORANGE, linewidth=2, label="Thermal ~q")
    ax.plot(xs, 1e-3 / xs**2, "--", color=CAT_AQUA, linewidth=2, label="Larkin ~1/q^2")

    ax.set_xlabel("q")
    ax.set_ylabel(r"$q^2\,\langle|u(q)|^2\rangle$")
    ax.set_title("Displacement spectrum")
    return True


def plot_structure_factor(ax, directory: str, ny: int) -> bool:
    """S_rho(qy), exact, with an optional Debye-Waller estimate overlay
    reconstructed from the matching correlation_replica_<T>.dat file (see
    README.md's Output Files section for the exp(-2 pi^2 B(r)) formula)."""
    groups = find_replica_files(directory, "structure_factor")
    if not groups:
        return False

    corr_groups = find_replica_files(directory, "correlation")

    for color, (T, paths) in zip(sequential_colors(len(groups)), groups.items()):
        data = load_averaged(paths, ncols=3)
        qy, s = data[:, 1], data[:, 2]
        ax.plot(qy, s, "o-", color=color, linewidth=2, markersize=5,
                 label=f"sim, T={T:g} ({len(paths)} seeds)")

        if T in corr_groups:
            b = load_averaged(corr_groups[T], ncols=3)
            by = b[:, 1]  # B_y(r), r = 0 .. Ny/2 - 1 (C++ never computes r = Ny/2)
            n_half = len(by)
            if n_half == ny // 2:
                # Mirror to a full period B_full(r) = B_full(Ny-r); the single
                # missing Nyquist point (r = Ny/2) is approximated by the last
                # known value -- a negligible one-point extrapolation.
                b_full = np.empty(ny)
                b_full[:n_half] = by
                b_full[n_half] = by[-1]
                b_full[n_half + 1:] = by[1:][::-1]
                s_dw = np.fft.fft(np.exp(-2.0 * np.pi**2 * b_full)).real
                qy_dw = 2.0 * np.pi * np.arange(ny) / ny
                mask = qy_dw <= qy.max()
                ax.plot(qy_dw[mask], s_dw[mask], ":", color=CAT_VIOLET, linewidth=2,
                         label=f"Debye-Waller est., T={T:g}")

    ax.set_xlabel("q")
    ax.set_ylabel(r"$S_\rho(q)$")
    ax.set_title("Density structure factor (exact)")
    return True


def plot_correlation(ax, directory: str) -> bool:
    """B(r) along and across chains, against thermal/Larkin roughness scaling."""
    groups = find_replica_files(directory, "correlation")
    if not groups:
        return False

    colors = sequential_colors(len(groups))
    r_min, r_max = np.inf, -np.inf
    for color, (T, paths) in zip(colors, groups.items()):
        data = load_averaged(paths, ncols=3)
        r, by, bx = data[:, 0], data[:, 1], data[:, 2]
        m = r > 0  # log-log can't show r=0
        r_min, r_max = min(r_min, r[m].min()), max(r_max, r[m].max())
        ax.plot(r[m], by[m], "o-", color=color, linewidth=2, markersize=5,
                 label=f"$B_y$, T={T:g} ({len(paths)} seeds)")
        m = (r > 0) & ~np.isnan(bx)
        if m.any():
            ax.plot(r[m], bx[m], "s--", color=color, linewidth=1.5, markersize=4,
                     alpha=0.6, label=f"$B_x$, T={T:g}")

    # See plot_spectrum() for why this is derived from the data, not ax.get_xlim().
    xs = np.geomspace(r_min, r_max, 100)
    ax.plot(xs, 0.25 * xs, "--", color=CAT_ORANGE, linewidth=2,
             label=r"Thermal ~x ($\zeta=1/2$)")
    ax.plot(xs, 0.04 * xs**2, "--", color=CAT_AQUA, linewidth=2,
             label=r"Larkin ~x$^2$ ($\zeta=3/2$)")

    ax.set_xlabel("r")
    ax.set_ylabel("B(r)")
    ax.set_title("Displacement correlations")
    return True
