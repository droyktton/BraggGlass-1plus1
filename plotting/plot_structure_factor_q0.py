#!/usr/bin/env python3
"""Plot the density structure factor near Q=0 (the compressional/acoustic
branch): S_rho(q) = 4*Ny*sin^2(q/2) * S_u(q), an exact rescaling of the
already-exported displacement spectrum, not a separate simulation output
(the Ny factor matches structure_factor's own normalization convention --
see plot_structure_factor_q0() in lib.py for the derivation). Reduces to
Ny*q^2*S_u(q) as q -> 0 -- compare against plot_spectrum.py's y-axis
(which already includes this same Ny factor) to check that relation
directly. For the genuinely independent Bragg-peak (Q=2pi) structure
factor, computed from the raw configuration, see plot_structure_factor.py
instead.

Usage:
    python plotting/plot_structure_factor_q0.py --ny 64 [--dir .] [--out plots/structure_factor_q0.png] [--show]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt
from lib import SURFACE, plot_structure_factor_q0, style_axes


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=".",
                     help="directory to search recursively for *_displacement_spectra_replica_*.dat")
    ap.add_argument("--ny", type=int, required=True, help="Ny used for the run (params.ini)")
    ap.add_argument("--out", default="plots/structure_factor_q0.png")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(6, 4.5), facecolor=SURFACE)
    if not plot_structure_factor_q0(ax, args.dir, args.ny):
        raise SystemExit(f"No displacement_spectra_replica_*.dat files found under {args.dir}")
    style_axes(ax, log_x=True, log_y=True)
    ax.legend(frameon=False, fontsize=8)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, facecolor=SURFACE)
    print(f"Wrote {args.out}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
