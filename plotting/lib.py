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
    """Group every [seed_<N>_]<stem>_replica_<T>.dat under root by
    temperature T. The match is anchored to the whole filename (not just a
    suffix) so e.g. stem="structure_factor" doesn't also pick up
    transverse_structure_factor_replica_<T>.dat.

    Matches a flat seed scan (seed_<N>_<stem>_replica_<T>.dat, from run.sh),
    a single run's output (<stem>_replica_<T>.dat), and the per-seed
    subdirectories written by slurm/run_array.slurm (results/seed_<N>/...).
    """
    pattern = os.path.join(root, "**", f"*{stem}_replica_*.dat")
    rx = re.compile(rf"^(?:seed_\d+_)?{re.escape(stem)}_replica_([0-9.eE+-]+)\.dat$")
    groups: dict[float, list[str]] = defaultdict(list)
    for path in glob.glob(pattern, recursive=True):
        m = rx.fullmatch(os.path.basename(path))
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


def plot_structure_factor_q0(ax, directory: str, ny: int) -> bool:
    """S_rho(q) near the origin (the Q=0 / compressional-phonon branch),
    exact but NOT a separate simulation output: for particles at nominal
    position y displaced to y+u(x,y), density conservation gives the
    density fluctuation delta_rho(x,y) = u(x,y+1) - u(x,y) to linear order
    in u -- a finite difference of the already-exported displacement field.
    Its Fourier transform is an exact rescaling of S_u, but note the
    normalization mismatch between how displacement_spectra (divided by
    Ny^2) and structure_factor (divided by Ny^1) are each normalized in
    the C++ code: matching structure_factor's convention (so this is
    directly comparable to the exact Bragg-peak file, not just
    proportional to it) requires an extra factor of Ny, from the discrete
    sum-of-roots-of-unity identity sum_y e^{iqy} = 0 for q != 0 mod 2*pi:
        S_rho(q) = Ny * |e^{iq}-1|^2 * S_u(q) = 4*Ny*sin^2(q/2) * S_u(q)
    which -> Ny*q^2*S_u(q) as q -> 0 (this is exactly ver.gnu's original
    q^2*S_u(q)*Ny convention on the Displacement spectrum panel). Plotted
    here on its own so it's directly comparable in shape/scale to the
    other (Q=2pi Bragg-peak) structure-factor panel, which is a genuinely
    independent, non-linear quantity -- see plot_structure_factor() below."""
    groups = find_replica_files(directory, "displacement_spectra")
    if not groups:
        return False

    for color, (T, paths) in zip(sequential_colors(len(groups)), groups.items()):
        data = load_averaged(paths, ncols=3)
        qy, su = data[:, 1], data[:, 2]
        s_rho0 = 4.0 * ny * np.sin(qy / 2.0)**2 * su
        ax.plot(qy, s_rho0, "o-", color=color, linewidth=2, markersize=5,
                 label=f"sim, T={T:g} ({len(paths)} seeds)")

    ax.set_xlabel("q")
    ax.set_ylabel(r"$S_\rho(q)$")
    ax.set_title(r"$S_\rho$ near $Q{=}0$ (exact, from $S_u$)")
    return True


def plot_structure_factor(ax, directory: str, ny: int) -> bool:
    """S_rho(qy) near the Bragg peak Q=2pi, exact, with an optional
    Debye-Waller estimate overlay reconstructed from the matching
    correlation_replica_<T>.dat file (see README.md's Output Files section
    for the exp(-2 pi^2 B(r)) formula). This is a genuinely independent,
    non-linear measurement -- unlike plot_structure_factor_q0() above, it
    is NOT derivable from S_u alone."""
    groups = find_replica_files(directory, "structure_factor")
    if not groups:
        return False

    corr_groups = find_replica_files(directory, "correlation")
    dw_labeled = False  # every T's DW curve shares one color/style -- label once

    for color, (T, paths) in zip(sequential_colors(len(groups)), groups.items()):
        data = load_averaged(paths, ncols=3)
        qy, s = data[:, 1], data[:, 2]
        ax.plot(qy, s, "o-", color=color, linewidth=2, markersize=5,
                 label=f"sim, T={T:g} ({len(paths)} seeds)")

        if T in corr_groups:
            # correlation_replica_<T>.dat always has max(Nx,Ny)/2 rows, so
            # B_y comes back nan-padded past its own r=Ny/2-1 whenever
            # Nx > Ny -- truncate to the valid part before using its
            # length as Ny/2 (C++ never computes r = Ny/2 either way).
            by = load_averaged(corr_groups[T], ncols=3)[:, 1][:ny // 2]
            n_half = len(by)
            if n_half == ny // 2 and not np.isnan(by).any():
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
                         label=None if dw_labeled else "Debye-Waller estimate (all T)")
                dw_labeled = True

    ax.set_xlabel("q")
    ax.set_ylabel(r"$S_\rho(q)$")
    ax.set_title(r"$S_\rho$ near Bragg peak $Q{=}2\pi$ (exact)")
    return True


def plot_structure_factor_full(ax, directory: str, ny: int) -> bool:
    """S_rho(q) across the whole first Brillouin zone, q in [0, 2*pi],
    stitching the two branches above onto one q-axis instead of two
    disconnected panels: a DFT on integer y only resolves q modulo 2*pi, so
    the Q=0 branch's q in [0, pi] and the Bragg branch's q = 2*pi - qy in
    [pi, 2*pi] (using S_rho(2pi+qy) = S_rho(2pi-qy)) tile the zone exactly
    once, meeting at the zone boundary q = pi.

    The two segments are NOT computed the same way -- the left one is the
    linear-in-u approximation (exact only as q -> 0; already visibly
    breaking down well before q = pi), the right one is the exact
    nonlinear measurement -- so don't expect them to join smoothly. The
    point of this plot is the vertical gap between them: how much
    (or how little) crystalline coherence survives at q ~ 2pi relative to
    the always-small compressional signal at q ~ 0."""
    su_groups = find_replica_files(directory, "displacement_spectra")
    rho_groups = find_replica_files(directory, "structure_factor")
    if not su_groups or not rho_groups:
        return False

    corr_groups = find_replica_files(directory, "correlation")
    dw_labeled = False

    for color, T in zip(sequential_colors(len(su_groups)), su_groups):
        if T not in rho_groups:
            continue
        su_data = load_averaged(su_groups[T], ncols=3)
        q0, s0 = su_data[:, 1], 4.0 * ny * np.sin(su_data[:, 1] / 2.0)**2 * su_data[:, 2]

        rho_data = load_averaged(rho_groups[T], ncols=3)
        qy_rho, s_bragg = rho_data[:, 1], rho_data[:, 2]
        q_bragg = 2.0 * np.pi - qy_rho
        order = np.argsort(q_bragg)

        ax.plot(q0, s0, "o-", color=color, linewidth=2, markersize=4,
                 label=f"T={T:g} ({len(su_groups[T])} seeds)")
        ax.plot(q_bragg[order], s_bragg[order], "o-", color=color, linewidth=2, markersize=4)

        # Same Debye-Waller reconstruction as plot_structure_factor(), folded
        # onto the q = 2pi - qy mapping used for the Bragg segment here.
        if T in corr_groups:
            by = load_averaged(corr_groups[T], ncols=3)[:, 1][:ny // 2]
            n_half = len(by)
            if n_half == ny // 2 and not np.isnan(by).any():
                b_full = np.empty(ny)
                b_full[:n_half] = by
                b_full[n_half] = by[-1]
                b_full[n_half + 1:] = by[1:][::-1]
                s_dw = np.fft.fft(np.exp(-2.0 * np.pi**2 * b_full)).real
                qy_dw = 2.0 * np.pi * np.arange(ny) / ny
                mask = qy_dw <= qy_rho.max()
                q_dw = 2.0 * np.pi - qy_dw[mask]
                order_dw = np.argsort(q_dw)
                ax.plot(q_dw[order_dw], s_dw[mask][order_dw], ":", color=CAT_VIOLET, linewidth=2,
                         label=None if dw_labeled else "Debye-Waller estimate (all T)")
                dw_labeled = True

    ax.axvline(np.pi, color=BASELINE, linewidth=1, linestyle="--", zorder=0)
    ax.set_xlabel("q")
    ax.set_ylabel(r"$S_\rho(q)$")
    ax.set_title(r"$S_\rho(q)$, full zone: $Q{=}0$ (approx) $\to$ Bragg peak $Q{=}2\pi$ (exact)")
    return True


# ─────────────────────────────────────────────────────────────────────────
# Transverse (across-chains, x-axis) diagnostics -- the vortex-LATTICE
# analogues of the three functions above. u(x,y) is a displacement in the
# same direction as x (see the HARDCORE constraint in the kernel, which
# compares u against its x-neighbors at the same y to keep chains from
# crossing), so the actual Bragg-glass translational order -- whether the
# Nx chains stay on a regular lattice -- lives in Fourier transforms along
# x, not y. The functions above (plot_spectrum, plot_structure_factor*)
# instead measure single-chain roughness along its own length; both sets
# are legitimate, complementary diagnostics, just of different physics.
# ─────────────────────────────────────────────────────────────────────────

def plot_transverse_spectrum_q0(ax, directory: str, nx: int) -> bool:
    """S_rho(q) near Q=0 for the vortex lattice: 4*Nx*sin^2(q/2) * S_u^(x)(q),
    an exact rescaling of transverse_spectrum_replica_<T>.dat (the spectrum
    of u along x, averaged over y) -- see plot_structure_factor_q0() for
    the along-y analogue and the derivation of the Nx factor (needed to
    match transverse_structure_factor's normalization convention, not just
    proportionality)."""
    groups = find_replica_files(directory, "transverse_spectrum")
    if not groups:
        return False

    for color, (T, paths) in zip(sequential_colors(len(groups)), groups.items()):
        data = load_averaged(paths, ncols=3)
        qx, su = data[:, 1], data[:, 2]
        s_rho0 = 4.0 * nx * np.sin(qx / 2.0)**2 * su
        ax.plot(qx, s_rho0, "o-", color=color, linewidth=2, markersize=5,
                 label=f"sim, T={T:g} ({len(paths)} seeds)")

    ax.set_xlabel("q")
    ax.set_ylabel(r"$S_\rho(q)$")
    ax.set_title(r"Vortex lattice $S_\rho$ near $Q{=}0$ (exact, from $S_u^{(x)}$)")
    return True


def _dw_estimate_from_bx(corr_groups, T, nx):
    """Debye-Waller estimate S_rho(q) ~ FFT[exp(-2 pi^2 B_x(r))], approaching
    the Bragg peak from below (q = 2*pi - offset). Returns (q, s) or None."""
    if T not in corr_groups:
        return None
    # correlation_replica_<T>.dat always has max(Nx,Ny)/2 rows, so B_x comes
    # back nan-padded past its own r=Nx/2-1 -- truncate before using its
    # length as Nx/2.
    bx = load_averaged(corr_groups[T], ncols=3)[:, 2][:nx // 2]
    n_half = len(bx)
    if n_half != nx // 2 or np.isnan(bx).any():
        return None
    b_full = np.empty(nx)
    b_full[:n_half] = bx
    b_full[n_half] = bx[-1]
    b_full[n_half + 1:] = bx[1:][::-1]
    s_dw = np.fft.fft(np.exp(-2.0 * np.pi**2 * b_full)).real
    q_dw = 2.0 * np.pi - 2.0 * np.pi * np.arange(nx) / nx
    order = np.argsort(q_dw)
    return q_dw[order], s_dw[order]


def plot_transverse_structure_factor(ax, directory: str, nx: int) -> bool:
    """Exact S_rho(q) for the vortex lattice, across the WHOLE first
    Brillouin zone q in [0, 2*pi] in one consistent calculation --
    transverse_structure_factor_replica_<T>.dat, computed from the literal
    phase q*(x+u(x,y)) (no small-q or near-2pi approximation, no
    stitching). q=0 is always exactly Nx (particle-number conservation,
    disorder-independent); q=2*pi is the actual Bragg peak and is NOT
    generally related to the q=0 value by periodicity once u != 0 -- a
    real jump between them is expected physics, not a plotting artifact.
    Includes a Debye-Waller estimate overlay near the Bragg peak from
    B_x(r) (transverse branch of correlation_replica_<T>.dat). See
    plot_structure_factor() for the (different, along-y) analogue."""
    groups = find_replica_files(directory, "transverse_structure_factor")
    if not groups:
        return False

    corr_groups = find_replica_files(directory, "correlation")
    dw_labeled = False

    for color, (T, paths) in zip(sequential_colors(len(groups)), groups.items()):
        data = load_averaged(paths, ncols=3)
        qx, s = data[:, 1], data[:, 2]
        ax.plot(qx, s, "o-", color=color, linewidth=2, markersize=5,
                 label=f"sim, T={T:g} ({len(paths)} seeds)")

        dw = _dw_estimate_from_bx(corr_groups, T, nx)
        if dw is not None:
            q_dw, s_dw = dw
            ax.plot(q_dw, s_dw, ":", color=CAT_VIOLET, linewidth=2,
                     label=None if dw_labeled else "Debye-Waller estimate (all T)")
            dw_labeled = True

    ax.set_xlabel("q")
    ax.set_ylabel(r"$S_\rho(q)$")
    ax.set_title(r"Vortex lattice $S_\rho(q)$, full zone (exact)")
    return True


def plot_transverse_structure_factor_full(ax, directory: str, nx: int) -> bool:
    """Same exact full-zone S_rho(q) as plot_transverse_structure_factor(),
    with the linear/compressional approximation 4 sin^2(q/2) S_u^(x)(q)
    (see plot_transverse_spectrum_q0()) also overlaid near q=0, so you can
    see directly how well it tracks the exact curve there and where it
    stops being a fair comparison."""
    rho_groups = find_replica_files(directory, "transverse_structure_factor")
    su_groups = find_replica_files(directory, "transverse_spectrum")
    if not rho_groups:
        return False

    corr_groups = find_replica_files(directory, "correlation")
    dw_labeled, approx_labeled = False, False

    for color, (T, paths) in zip(sequential_colors(len(rho_groups)), rho_groups.items()):
        data = load_averaged(paths, ncols=3)
        qx, s = data[:, 1], data[:, 2]
        ax.plot(qx, s, "o-", color=color, linewidth=2, markersize=4,
                 label=f"T={T:g} ({len(paths)} seeds)")

        if T in su_groups:
            su_data = load_averaged(su_groups[T], ncols=3)
            q0, s0 = su_data[:, 1], 4.0 * nx * np.sin(su_data[:, 1] / 2.0)**2 * su_data[:, 2]
            ax.plot(q0, s0, "--", color=color, linewidth=1.2, alpha=0.6,
                     label=None if approx_labeled else "linear approx., $4\\sin^2(q/2)S_u^{(x)}(q)$")
            approx_labeled = True

        dw = _dw_estimate_from_bx(corr_groups, T, nx)
        if dw is not None:
            q_dw, s_dw = dw
            ax.plot(q_dw, s_dw, ":", color=CAT_VIOLET, linewidth=2,
                     label=None if dw_labeled else "Debye-Waller estimate (all T)")
            dw_labeled = True

    ax.set_xlabel("q")
    ax.set_ylabel(r"$S_\rho(q)$")
    ax.set_title(r"Vortex lattice $S_\rho(q)$, full zone, vs. both approximations")
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
