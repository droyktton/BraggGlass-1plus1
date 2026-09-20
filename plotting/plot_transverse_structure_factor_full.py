#!/usr/bin/env python3
"""Plot the vortex-lattice S_rho(q) across the whole first Brillouin zone
(q in [0, 2*pi]), stitching the Q=0 approximation (from transverse_spectrum,
i.e. S_u along x) to the exact Bragg-peak branch (transverse_structure_factor,
i.e. translational order of the Nx chains) at q=pi. This is the vortex-lattice
analogue of plot_structure_factor_full.py, and the one that actually answers
whether the lattice remains a Bragg glass or has molten.

Usage:
    python plotting/plot_transverse_structure_factor_full.py --nx 32 [--dir .] [--out plots/transverse_structure_factor_full.png] [--show]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt
from lib import SURFACE, plot_transverse_structure_factor_full, style_axes


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=".",
                     help="directory to search recursively for *_transverse_spectrum_replica_*.dat and *_transverse_structure_factor_replica_*.dat")
    ap.add_argument("--nx", type=int, required=True, help="Nx used for the run (params.ini)")
    ap.add_argument("--out", default="plots/transverse_structure_factor_full.png")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor=SURFACE)
    if not plot_transverse_structure_factor_full(ax, args.dir, args.nx):
        raise SystemExit(f"Need both transverse_spectrum_replica_*.dat and "
                          f"transverse_structure_factor_replica_*.dat files under {args.dir}")
    style_axes(ax, log_x=False, log_y=True)
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    print(f"Wrote {args.out}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
