#!/usr/bin/env python3
"""Check the small-q hydrodynamic approximation to the vortex-lattice
S_rho(qx) against the exact value: overlays S_rho(qx) with
Nx*q^2*S_u^(x)(q) (left panel), and their ratio vs q (right panel), per
temperature. The ratio should sit at ~1 for the smallest accessible q and
depart from 1 as q grows -- faster at higher T, since the approximation
requires q*u << 1 and u fluctuates more at higher T.

Usage:
    python plotting/plot_hydro_check.py --nx 32 [--dir .] [--out plots/hydro_check.png] [--show]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt
from lib import SURFACE, plot_hydro_comparison, plot_hydro_ratio, style_axes


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=".",
                     help="directory to search recursively for *_transverse_structure_factor_replica_*.dat and *_transverse_spectrum_replica_*.dat")
    ap.add_argument("--nx", type=int, required=True, help="Nx used for the run (params.ini)")
    ap.add_argument("--out", default="plots/hydro_check.png")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), facecolor=SURFACE)

    found = plot_hydro_comparison(axes[0], args.dir, args.nx)
    found |= plot_hydro_ratio(axes[1], args.dir, args.nx)
    if not found:
        raise SystemExit(f"Need both transverse_structure_factor_replica_*.dat and "
                          f"transverse_spectrum_replica_*.dat files under {args.dir}")

    style_axes(axes[0], log_x=True, log_y=True)
    style_axes(axes[1], log_x=True, log_y=False)
    axes[1].set_ylim(0, 1.3)
    for ax in axes:
        if ax.lines:
            ax.legend(fontsize=7, frameon=False)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, facecolor=SURFACE)
    print(f"Wrote {args.out}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
