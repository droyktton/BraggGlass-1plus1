#!/usr/bin/env python3
"""Plot S_rho(q) across the whole first Brillouin zone (q in [0, 2*pi]),
combining the Q=0 approximation (small q, from S_u) and the exact Bragg-peak
branch (q near 2*pi) on one q-axis, so you can see the full crossover from
the always-vanishing compressional signal at q->0 to whatever crystalline
coherence survives near the Bragg peak -- instead of two separate panels
that only show each piece on its own.

Usage:
    python plotting/plot_structure_factor_full.py --ny 64 [--dir .] [--out plots/structure_factor_full.png] [--show]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt
from lib import SURFACE, plot_structure_factor_full, style_axes


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=".",
                     help="directory to search recursively for *_displacement_spectra_replica_*.dat and *_structure_factor_replica_*.dat")
    ap.add_argument("--ny", type=int, required=True, help="Ny used for the run (params.ini)")
    ap.add_argument("--out", default="plots/structure_factor_full.png")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor=SURFACE)
    if not plot_structure_factor_full(ax, args.dir, args.ny):
        raise SystemExit(f"Need both displacement_spectra_replica_*.dat and "
                          f"structure_factor_replica_*.dat files under {args.dir}")
    style_axes(ax, log_x=False, log_y=True)
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    print(f"Wrote {args.out}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
