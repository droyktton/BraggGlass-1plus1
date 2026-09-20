#!/usr/bin/env python3
"""Plot the vortex-lattice density structure factor near the Bragg peak
Q=2pi: transverse_structure_factor_replica_<T>.dat, the exact translational
order of the Nx chains (computed along x, averaged over y) -- this is the
one that actually answers "is the lattice a Bragg glass or has it molten,"
unlike plot_structure_factor.py's along-chain (y-axis) analogue. Includes a
Debye-Waller estimate overlay from B_x(r) (correlation_replica_<T>.dat).

Usage:
    python plotting/plot_transverse_structure_factor.py --nx 32 [--dir .] [--out plots/transverse_structure_factor.png] [--show]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt
from lib import SURFACE, plot_transverse_structure_factor, style_axes


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=".",
                     help="directory to search recursively for *_transverse_structure_factor_replica_*.dat")
    ap.add_argument("--nx", type=int, required=True, help="Nx used for the run (params.ini)")
    ap.add_argument("--out", default="plots/transverse_structure_factor.png")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor=SURFACE)
    if not plot_transverse_structure_factor(ax, args.dir, args.nx):
        raise SystemExit(f"No transverse_structure_factor_replica_*.dat files found under {args.dir}")
    style_axes(ax, log_x=False, log_y=True)
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    print(f"Wrote {args.out}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
